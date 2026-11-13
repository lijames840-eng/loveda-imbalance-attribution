"""Full-tile evaluation + efficiency benchmark for the paper's comparison table.

Reports the four quantities the course asks for:
    recognition accuracy (mIoU / OA), recall (per-class + mean),
    model size (params, FLOPs), inference speed (ms/img, FPS).

Usage:
    python code/evaluate.py --run baseline_unet --model unet_r34 --split val
    python code/evaluate.py --bench-only --model balnet_v2
"""
from paths import DATA_RAW  # noqa: E402
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loveda_dataset import NUM_CLASSES, MEAN, STD, build_index  # noqa: E402
from metrics import ConfusionMatrix  # noqa: E402
from models import build_model  # noqa: E402
from imgio import imread  # noqa: E402


def load_model(model_name, ckpt=None, device="cuda"):
    model = build_model(model_name, num_classes=NUM_CLASSES).to(device)
    if ckpt:
        # Must fail loudly: silently returning a randomly initialised model produces
        # plausible-looking JSON (it did once -- mIoU 0.57% went straight into a file).
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"checkpoint not found: {ckpt}")
        sd = torch.load(ckpt, map_location=device)
        model.load_state_dict(sd["model"] if "model" in sd else sd)
    model.eval()
    return model


@torch.no_grad()
def predict_tile(model, img, device, window=512, stride=256, return_prob=False,
                 return_prior=False):
    """Sliding-window inference over a full tile, probability averaging.

    Probabilities are renormalised over the **7 real classes** before the argmax.
    Channel 0 is the no-data placeholder that makes ``cross_entropy(ignore_index=0)``
    work during training; at inference every pixel must receive a real label, so
    letting channel 0 win would silently delete hard pixels from the confusion
    matrix -- and would let a threshold correction "improve" mIoU simply by
    predicting no-data.

    Models with a prior head (V3) return ``(logit, pi)``; ``pi`` is averaged over
    the sliding windows and returned as the tile-level composition estimate that
    the inference-time correction consumes.

    Returns labels in 1..7 by default, the 7-channel map if ``return_prob``, or
    ``(prob, pi)`` if ``return_prior`` (``pi`` is None for models without a head).
    """
    _, h, w = img.shape
    prob = torch.zeros((NUM_CLASSES, h, w), device=device)
    cnt = torch.zeros((1, h, w), device=device)
    pi_acc, n_pi = None, 0
    ys = list(range(0, max(h - window, 0) + 1, stride)) or [0]
    xs = list(range(0, max(w - window, 0) + 1, stride)) or [0]
    if ys[-1] + window < h:
        ys.append(h - window)
    if xs[-1] + window < w:
        xs.append(w - window)
    for y in ys:
        for x in xs:
            patch = img[:, y:y + window, x:x + window].unsqueeze(0).to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out = model(patch)
            if isinstance(out, (tuple, list)):
                logit = out[0]
                if return_prior and len(out) > 1 and out[1] is not None:
                    pi_now = out[1][0].float()
                    pi_acc = pi_now.clone() if pi_acc is None else pi_acc + pi_now
                    n_pi += 1
            else:
                logit = out
            prob[:, y:y + window, x:x + window] += torch.softmax(logit.float(), 1)[0]
            cnt[:, y:y + window, x:x + window] += 1
    p = prob / cnt.clamp(min=1)
    p = p[1:] / p[1:].sum(0, keepdim=True).clamp(min=1e-8)   # renormalise over 7 classes
    if return_prior:
        return p, (pi_acc / n_pi if n_pi else None)
    if return_prob:
        return p
    return p.argmax(0) + 1


def evaluate(model_name, ckpt, split="val", root=str(DATA_RAW), device="cuda", limit=0):
    model = load_model(model_name, ckpt, device)
    items = build_index(root, split)
    if limit:
        items = items[:limit]
    cm = ConfusionMatrix()
    per_domain = {"urban": ConfusionMatrix(), "rural": ConfusionMatrix()}
    t0 = time.time()
    for i, (ipath, mpath, domain) in enumerate(items, 1):
        img = imread(ipath, cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32) / 255.0
        msk = imread(mpath, cv2.IMREAD_GRAYSCALE)
        img = torch.from_numpy(((img - MEAN) / STD).transpose(2, 0, 1).copy())
        pred = predict_tile(model, img, device).cpu().numpy()
        pred = cv2.resize(pred.astype(np.uint8), (msk.shape[1], msk.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
        cm.update(pred, msk)
        per_domain[domain].update(pred, msk)
        if i % 50 == 0:
            print(f"  {i}/{len(items)}  ({time.time()-t0:.0f}s)", flush=True)
    res = cm.compute()
    res["per_domain"] = {k: v.compute()["mIoU"] for k, v in per_domain.items()}
    res["n_tiles"] = len(items)
    return res, cm


@torch.no_grad()
def benchmark(model_name, device="cuda", size=512, warmup=20, iters=50):
    model = load_model(model_name, None, device)
    x = torch.randn(1, 3, size, size, device=device)
    for _ in range(warmup):
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms = (time.time() - t0) / iters * 1000
    params = sum(p.numel() for p in model.parameters()) / 1e6
    return dict(model=model_name, params_M=round(params, 2), ms_per_img=round(ms, 1),
                fps=round(1000 / ms, 1), input=size)


def flops(model_name, size=512):
    """GFLOPs via the local hook counter (bench.py) -- no external dependency."""
    try:
        from bench import count_flops
        return round(count_flops(build_model(model_name, num_classes=NUM_CLASSES), size), 2)
    except Exception as e:
        return f"n/a ({e})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run", default="")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--split", default="val")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bench-only", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.bench_only:
        print(json.dumps(benchmark(args.model, device), ensure_ascii=False))
        return

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    ckpt = args.ckpt or os.path.join(cfg["paths"]["experiments"], args.run, "best.pth")
    res, cm = evaluate(args.model, ckpt, args.split, cfg["paths"]["data"], device, args.limit)
    print(cm.table())
    out = os.path.join(cfg["paths"]["experiments"], args.run, f"eval_{args.split}.json")
    res["bench"] = benchmark(args.model, device)
    res["flops_G"] = flops(args.model)
    # numpy arrays are not JSON serialisable
    for k, v in list(res.items()):
        if isinstance(v, np.ndarray):
            res[k] = [float(x) for x in v]
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"mIoU={res['mIoU']*100:.2f} OA={res['OA']*100:.2f} "
          f"urban={res['per_domain']['urban']*100:.2f} rural={res['per_domain']['rural']*100:.2f}")
    print("saved ->", out)


if __name__ == "__main__":
    main()
