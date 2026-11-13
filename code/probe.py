"""E-P1: is the minority-class failure recoverable from the frozen features?

Position in the plan
--------------------
`stage0` §1.1 splits the loss into two layers only -- decision and representation.
E-delta (code/fit_threshold.py) probes the decision layer with the *narrowest*
possible intervention: a constant added per class at inference, nothing re-trained.
E-P1 is the next rung on the same ladder: the features stay frozen, but the read-out
is re-fitted from scratch under a **class-balanced** objective -- an objective that
does not itself carry the prior bias we are testing for.

Why the objective has to change
-------------------------------
An earlier version of this probe re-trained the last 1x1 layer with *plain* CE on the
same split and concluded "H2 rejected: the classifier is not the bottleneck".  That
was circular: plain CE on the same data reproduces the same prior bias, so a null
result was guaranteed no matter what the truth was.  `--mode balanced` is the fix.

What each outcome means
-----------------------
* balanced probe recovers minority recall well beyond E-delta -> the information is
  in the features; the loss sits in the decision rule, and a richer read-out can use it.
* balanced probe matches E-delta, and both stay far below the in-domain number ->
  the features themselves do not support the minority classes => representation-limited.

Caveats to state in the paper
-----------------------------
* A linear read-out failing does **not** prove the features are useless -- only that
  they are not linearly separable.
* The frozen features were co-trained with the biased head, so part of the bias may
  live in them; freezing preserves it rather than removing it.
* Head capacity is a possible confound, hence `--head full` (3x3 + 1x1, ~37k params)
  as the control for `--head last` (1x1, 520 params).
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import predict_tile  # noqa: E402
from loveda_dataset import CLASSES, IGNORE, MEAN, NUM_CLASSES, STD, LoveDA, build_index  # noqa: E402
from metrics import ConfusionMatrix  # noqa: E402
from models import build_model  # noqa: E402
from imgio import imread  # noqa: E402


def class_frequencies(ds):
    """Dataset-level class frequencies over the *training* split (not per image)."""
    cnt = np.zeros(NUM_CLASSES, np.float64)
    for _, mpath, _ in ds.items:
        m = imread(mpath, cv2.IMREAD_GRAYSCALE)
        cnt += np.bincount(m.ravel(), minlength=NUM_CLASSES)[:NUM_CLASSES]
    cnt[0] = 0.0
    return cnt / cnt.sum()


def reset_head(head):
    """Re-initialise the read-out: convs by kaiming, the final 1x1 to zero."""
    for m in head.modules():
        if isinstance(m, nn.Conv2d):
            if m.out_channels == NUM_CLASSES and m.kernel_size == (1, 1):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def build_probe(model_name, ckpt, device, head_mode="last"):
    model = build_model(model_name, num_classes=NUM_CLASSES).to(device)
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(sd["model"] if "model" in sd else sd)
    for p in model.parameters():
        p.requires_grad = False

    head = model.head
    if head_mode == "last":
        # keep the 3x3 ConvBNR frozen, replace only the 1x1 classifier (520 params)
        new_head = nn.Conv2d(head[-1].in_channels, head[-1].out_channels, 1).to(device)
        nn.init.zeros_(new_head.weight)
        nn.init.zeros_(new_head.bias)
        head[-1] = new_head
        trainable = list(new_head.parameters())
    else:
        # capacity control: reset and unfreeze the whole head (~37k params)
        reset_head(head)
        for p in head.parameters():
            p.requires_grad = True
        trainable = list(head.parameters())
    return model, trainable


@torch.no_grad()
def eval_items(model, items, device, limit=0):
    cm = ConfusionMatrix()
    model.eval()
    items = items[:limit] if limit else items
    for ipath, mpath, _ in items:
        img = imread(ipath, cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32) / 255.0
        gt = imread(mpath, cv2.IMREAD_GRAYSCALE)
        x = torch.from_numpy(((img - MEAN) / STD).transpose(2, 0, 1).copy())
        pred = predict_tile(model, x, device).cpu().numpy()
        if pred.shape != gt.shape:
            pred = cv2.resize(pred.astype(np.uint8), (gt.shape[1], gt.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        cm.update(pred, gt)
    return cm.compute()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="V1_ce")
    ap.add_argument("--model", default="v1")
    ap.add_argument("--config", default="code/configs/e0_ce.yaml")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--mode", default="balanced", choices=["plain", "balanced"])
    ap.add_argument("--head", default="last", choices=["last", "full"])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.0, help="0 = auto (1e-2 for last, 1e-3 for full)")
    ap.add_argument("--batch-size", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0,
                    help="evaluate only N tiles (dev and val) -- for smoke tests only")
    ap.add_argument("--max-batches", type=int, default=0,
                    help="stop each epoch after N batches -- for smoke tests only")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp = cfg["paths"]["experiments"]
    ckpt = args.ckpt or os.path.join(exp, args.run, "best.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")

    tag = f"probe_{args.mode}_{args.head}"
    out_dir = os.path.join(exp, args.run, tag)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[{tag}] ckpt={ckpt}")
    model, trainable = build_probe(args.model, ckpt, device, args.head)
    n_train = sum(p.numel() for p in trainable)
    print(f"[{tag}] trainable parameters: {n_train}")

    train_ds = LoveDA(cfg["paths"]["data"], "train", crop=512, train_mode=True)
    dev_ds = LoveDA(cfg["paths"]["data"], "dev", crop=512, train_mode=False)
    bs = args.batch_size or cfg["train"]["batch_size"]
    tl = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True,
                    num_workers=cfg["data"]["workers"], pin_memory=True)

    weight = None
    if args.mode == "balanced":
        # A *balanced* objective weights every class equally: w_c = 1 / freq_c,
        # normalised to mean 1.  This is deliberately stronger than the sqrt rule
        # used by E2 -- the point is to remove the prior from the objective entirely.
        p = class_frequencies(train_ds)
        w = np.zeros(NUM_CLASSES)
        pres = p > 0
        w[pres] = 1.0 / p[pres]
        w[pres] = w[pres] / w[pres].mean()
        weight = torch.tensor(w, dtype=torch.float32, device=device)
        print(f"[{tag}] class weights: {np.round(w, 2).tolist()}")

    lr = args.lr or (1e-2 if args.head == "last" else 1e-3)
    opt = torch.optim.AdamW(trainable, lr=lr)

    best_dev, best_state = -1.0, None
    for ep in range(args.epochs):
        # backbone frozen in eval mode (BN statistics must not drift); the read-out
        # itself still trains, which matters for --head full (it contains a BN).
        model.eval()
        model.head.train()
        tot, n = 0.0, 0
        for bi, (img, msk, _) in enumerate(tl):
            if args.max_batches and bi >= args.max_batches:
                break
            img, msk = img.to(device), msk.to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logit = model(img)
                loss = nn.functional.cross_entropy(logit, msk, weight=weight, ignore_index=IGNORE)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item()
            n += 1
        dev = eval_items(model, dev_ds.items, device, args.limit)
        r = dev["Recall"]
        print(f"  ep{ep + 1}: loss={tot / max(n, 1):.4f}  dev mIoU={dev['mIoU'] * 100:5.2f}  "
              f"barren R={r[4] * 100:5.1f}  agri R={r[6] * 100:5.1f}", flush=True)
        if dev["mIoU"] > best_dev:
            best_dev = dev["mIoU"]
            best_state = {k: v.detach().clone() for k, v in model.head.state_dict().items()}

    model.head.load_state_dict(best_state)
    print(f"[{tag}] best dev mIoU={best_dev * 100:.2f}; evaluating official Val ...")
    res = eval_items(model, build_index(cfg["paths"]["data"], "val"), device, args.limit)

    out = dict(mode=args.mode, head=args.head, trainable_params=int(n_train), lr=lr,
               epochs=args.epochs, dev_mIoU=float(best_dev), val_mIoU=float(res["mIoU"]),
               val_OA=float(res["OA"]), val_mF1=float(res["mF1"]),
               per_class={c: dict(IoU=float(res["IoU"][i]), Recall=float(res["Recall"][i]),
                                  Precision=float(res["Precision"][i]))
                          for i, c in enumerate(CLASSES)})
    with open(os.path.join(out_dir, "probe_result.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print(f"Val mIoU={res['mIoU'] * 100:.2f}  OA={res['OA'] * 100:.2f}")
    for i, c in enumerate(CLASSES):
        print(f"  {c:12s} IoU={res['IoU'][i] * 100:5.1f}  R={res['Recall'][i] * 100:5.1f}  "
              f"P={res['Precision'][i] * 100:5.1f}")


if __name__ == "__main__":
    main()
