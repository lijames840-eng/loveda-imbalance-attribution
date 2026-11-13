"""Unified training script (model-agnostic) for the course project.

Same recipe for every model so the comparison table in the paper is fair:
AdamW, cosine schedule, 512 crops, AMP + gradient accumulation for the 6 GB GPU.

Usage:
    python code/train.py --config configs/base.yaml --model unet_r34 --run baseline_unet
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loveda_dataset import CLASSES, NUM_CLASSES, IGNORE, LoveDA  # noqa: E402
from metrics import ConfusionMatrix  # noqa: E402
from models import build_model  # noqa: E402
from losses import CEDiceLoss  # noqa: E402
from models.v3 import apply_prior_correction, prior_loss  # noqa: E402

CombinedLoss = CEDiceLoss  # backward-compatible alias


def seed_all(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def validate(model, loader, device, max_batches=0, prior_tau=0.0):
    """Monitor the dev split.  ``prior_tau`` > 0 applies the PAH correction here too.

    Model selection must use the *same* configuration that will be reported, so for
    V3 the checkpoint is chosen on the corrected predictions (prior_tau=1).  The tau
    is pre-registered, and pi comes from the model itself, so this is not a fitted
    hyper-parameter and introduces no selection bias.
    """
    cm = ConfusionMatrix()
    model.eval()
    for i, (img, msk, _) in enumerate(loader):
        img, msk = img.to(device, non_blocking=True), msk.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            out = model(img)
            if isinstance(out, (tuple, list)):
                logit, pi = out[0], (out[1] if len(out) > 1 else None)
            else:
                logit, pi = out, None
        logit = logit.float()
        if prior_tau > 0 and pi is not None:
            logit = apply_prior_correction(logit, pi.float(), tau=prior_tau)
        # same rule as evaluate.predict_tile: argmax over the 7 real classes only
        prob = torch.softmax(logit, 1)[:, 1:]
        prob = prob / prob.sum(1, keepdim=True).clamp_min(1e-8)
        pred = (prob.argmax(1) + 1).cpu().numpy()
        cm.update(pred, msk.cpu().numpy())
        if max_batches and i + 1 >= max_batches:
            break
    return cm.compute(), cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--model", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--limit-batches", type=int, default=0, help="smoke test")
    ap.add_argument("--resume", action="store_true", help="resume from last.pth if present")
    ap.add_argument("--val-tiles", type=int, default=0,
                    help="use only N evenly-spaced val tiles for per-epoch monitoring "
                         "(0 = full val). Final reporting must still use evaluate.py on the full set.")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    epochs = args.epochs or cfg["train"]["epochs"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(cfg["train"].get("seed", 42))

    out_dir = os.path.join(cfg["paths"]["experiments"], args.run)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump({"model": args.model, **cfg}, f, allow_unicode=True)

    train_ds = LoveDA(cfg["paths"]["data"], "train", crop=cfg["data"]["crop"], train_mode=True,
                      scale_range=tuple(cfg["data"]["scale_range"]))
    # Per-epoch monitoring and checkpoint selection use the held-out *dev* split only.
    # The official Val is never touched here -- evaluate.py runs it once at the end.
    val_ds = LoveDA(cfg["paths"]["data"], "dev", crop=cfg["data"]["crop"], train_mode=False)
    if args.val_tiles and args.val_tiles < len(val_ds):
        step = max(len(val_ds) // args.val_tiles, 1)
        idx = list(range(0, len(val_ds), step))[:args.val_tiles]
        from torch.utils.data import Subset
        val_ds = Subset(val_ds, idx)
        print(f"[monitor] per-epoch monitoring uses {len(val_ds)} of the dev tiles")
    nw = cfg["data"]["workers"]
    train_ld = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                          num_workers=nw, drop_last=True, pin_memory=True,
                          persistent_workers=nw > 0)
    val_ld = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                        num_workers=nw, drop_last=False, pin_memory=True)

    model = build_model(args.model, num_classes=NUM_CLASSES).to(device)
    n_param = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[{args.run}] model={args.model} params={n_param:.2f}M train={len(train_ds)} dev(monitor)={len(val_ds)}")

    lc = cfg["train"].get("loss", {})
    share_scale = lc.get("share_scale")
    if isinstance(share_scale, str):
        # path to the JSON emitted by code/calibrate_share.py
        ss = json.load(open(share_scale, encoding="utf-8"))
        share_scale = ss["k_full"]
        print(f"[loss] share_scale <- {ss['k']}")
    criterion = CEDiceLoss(
        num_classes=NUM_CLASSES,
        dice_weight=lc.get("dice_weight", cfg["train"].get("dice_weight", 0.5)),
        per_image=lc.get("per_image", False),
        weight_clip=tuple(lc.get("weight_clip", (0.5, 4.0))),
        class_weights=cfg["train"].get("class_weights"),
        boundary_weight=lc.get("boundary_weight", 0.0),
        share_scale=share_scale,
    ).to(device)
    prior_weight = float(cfg["train"].get("prior_weight", 0.0))
    if prior_weight > 0:
        print(f"[loss] prior auxiliary weight = {prior_weight}")
    # Checkpoint selection uses the same configuration we will report. For V3 that
    # means the PAH correction is ON during dev monitoring (pre-registered tau=1).
    select_tau = float(cfg["train"].get("select_corrected_tau", 0.0))
    if select_tau > 0:
        print(f"[select] dev monitoring applies prior correction (tau={select_tau:g})")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    accum = cfg["train"]["accum_steps"]
    iters_per_epoch = len(train_ld)
    # LambdaLR advances once per *optimizer* update, and an update happens only every
    # `accum` micro-batches (see the sched.step() call below). So the cosine horizon must
    # be counted in optimizer steps, not dataloader iterations. Using iters_per_epoch
    # made any accum>1 run stop annealing at half the schedule: E10 (batch 4 x accum 2)
    # ended at lr 3.0e-4 instead of ~0, with loss still falling and dev still climbing.
    steps_per_epoch = max(iters_per_epoch // accum, 1)
    total = max(epochs * steps_per_epoch, 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: 0.5 * (1 + np.cos(np.pi * min(it, total) / total)))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    log_path = os.path.join(out_dir, "train_log.csv")
    resuming = args.resume and os.path.exists(os.path.join(out_dir, "last.pth"))
    # On resume the log already holds the earlier epochs -- appending keeps them.
    header_needed = not (resuming and os.path.exists(log_path) and os.path.getsize(log_path) > 0)
    with open(log_path, "a" if resuming else "w", newline="") as f:
        if header_needed:
            csv.writer(f).writerow(["epoch", "lr", "train_loss", "dev_mIoU", "dev_OA",
                                    "dev_mF1", "epoch_time_s"])

    best = 0.0
    start_ep = 0
    ckpt_last = os.path.join(out_dir, "last.pth")
    if args.resume and os.path.exists(ckpt_last):
        ck = torch.load(ckpt_last, map_location=device)
        model.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        if "sched" in ck:
            sched.load_state_dict(ck["sched"])
        start_ep = ck.get("epoch", 0)
        best = ck.get("best", ck.get("mIoU", 0.0))
        print(f"[resume] from epoch {start_ep}, best mIoU={best*100:.2f}")

    for ep in range(start_ep, epochs):
        model.train()
        t0 = time.time()
        running, seen = 0.0, 0
        opt.zero_grad(set_to_none=True)
        pbar = tqdm(train_ld, desc=f"ep{ep+1}/{epochs}", ncols=100)
        for step, (img, msk, _) in enumerate(pbar):
            img = img.to(device, non_blocking=True)
            msk = msk.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out = model(img)
                if isinstance(out, (tuple, list)):
                    logit, pi_hat = out[0], out[1]   # V3 style: segmentation + prior head
                else:
                    logit, pi_hat = out, None
                loss = criterion(logit, msk)
                if pi_hat is not None and prior_weight > 0:
                    # supervise the composition predictor; the main loss above uses the
                    # UNCORRECTED logits so the network cannot cancel pi against them
                    loss = loss + prior_weight * prior_loss(pi_hat, msk)
                loss = loss / accum
            scaler.scale(loss).backward()
            if (step + 1) % accum == 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            running += loss.item() * accum
            seen += 1
            if step % 20 == 0:
                pbar.set_postfix(loss=f"{running / max(seen,1):.4f}")
            if args.limit_batches and step + 1 >= args.limit_batches:
                break

        res, _ = validate(model, val_ld, device, max_batches=args.limit_batches,
                          prior_tau=select_tau)
        dt = time.time() - t0
        lr = opt.param_groups[0]["lr"]
        print(f"  ep{ep+1}: loss={running/max(seen,1):.4f} mIoU={res['mIoU']*100:.2f} "
              f"OA={res['OA']*100:.2f} ({dt/60:.1f} min)")
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([ep + 1, f"{lr:.6f}", f"{running/max(seen,1):.4f}",
                                    f"{res['mIoU']:.4f}", f"{res['OA']:.4f}",
                                    f"{res['mF1']:.4f}", f"{dt:.1f}"])
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": ep + 1,
                    "mIoU": res["mIoU"], "best": max(best, res["mIoU"])},
                   ckpt_last)
        if res["mIoU"] > best:
            best = res["mIoU"]
            torch.save({"model": model.state_dict(), "epoch": ep + 1, "mIoU": res["mIoU"]},
                       os.path.join(out_dir, "best.pth"))
        if args.limit_batches:
            break
    print(f"[{args.run}] best DEV mIoU (used for checkpoint selection) = {best*100:.2f}")


if __name__ == "__main__":
    main()
