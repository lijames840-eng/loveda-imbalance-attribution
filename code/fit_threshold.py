"""E-delta: how much of the loss can be recovered by moving the decision boundary only?

Protocol
--------
 1. Slide a 512/256 window over the split, cache the 7-channel probability maps
    (optionally average-pooled by --ds to keep the search cheap).
 2. Fit a per-class additive bias delta (7 numbers) on **dev** by coordinate ascent.
    Two flavours:
       prior : delta_c = -tau * log(pi_c)   1 free parameter, the Bayes-style form
       free  : delta_c searched independently, 7 free parameters, upper bound
 3. Apply the dev-fitted delta to the official Val split. Nothing is re-fitted
    there -- that is the honest transfer number.
 4. With --cache-val, also fit delta directly on Val and report it as the
    *ceiling* of this family.  That number is fitted on the test set; it is a
    diagnostic only and must never be reported as a result.

Why this is the clean test of "decision-layer headroom"
-------------------------------------------------------
The features never change: same weights, same logits, only a constant added per
class before the argmax.  Contrast with training-time class weighting (E2/E3),
which also moves the representation through the gradient -- that is why those
results cannot be read as decision-layer evidence.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import load_model, predict_tile  # noqa: E402
from loveda_dataset import CLASSES, MEAN, STD, build_index  # noqa: E402
from metrics import ConfusionMatrix  # noqa: E402
from imgio import imread  # noqa: E402

GRID_COARSE = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
GRID_FINE = [-0.5, -0.25, 0.0, 0.25, 0.5]
TAUS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]


def prep(ipath):
    img = imread(ipath, cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32) / 255.0
    return torch.from_numpy(((img - MEAN) / STD).transpose(2, 0, 1).copy())


@torch.no_grad()
def build_cache(model, items, device, npz_path, ds=4):
    """Cache (7,h,w) probability maps + downsampled GT as one npz."""
    if os.path.exists(npz_path):
        d = np.load(npz_path)
        print(f"cache reused -> {npz_path}  {d['prob'].shape}")
        return d["prob"], d["gt"]
    probs, gts = [], []
    for i, (ip, mk, _) in enumerate(items, 1):
        p = predict_tile(model, prep(ip), device, return_prob=True)
        if ds > 1:
            p = torch.nn.functional.avg_pool2d(p.unsqueeze(0), ds, ds)[0]
        gt = imread(mk, cv2.IMREAD_GRAYSCALE)
        if ds > 1:
            gt = cv2.resize(gt, (p.shape[2], p.shape[1]), interpolation=cv2.INTER_NEAREST)
        probs.append(p.cpu().numpy().astype(np.float16))
        gts.append(gt.astype(np.uint8))
        if i % 100 == 0:
            print(f"  cache {i}/{len(items)}", flush=True)
    prob = np.stack(probs)
    gt = np.stack(gts)
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    np.savez(npz_path, prob=prob, gt=gt)
    print(f"cached -> {npz_path}  {prob.shape}")
    return prob, gt


def flatten(prob, gt):
    """(N,7,h,w) + (N,h,w) -> (P,7) float32 **log** probabilities, (P,) class index 0..6.

    The bias must be added in logit space (log p_c + delta_c), which is the
    standard logit adjustment p_c * exp(delta_c); adding it to the probability
    itself would not correspond to moving a decision boundary.
    """
    p = prob.transpose(0, 2, 3, 1).reshape(-1, prob.shape[1]).astype(np.float32)
    p = np.log(np.maximum(p, 1e-8))
    t = gt.reshape(-1)
    keep = t > 0
    return np.ascontiguousarray(p[keep]), (t[keep] - 1).astype(np.int64)


def confusion(flat, tc, delta, buf):
    np.add(flat, delta, out=buf)
    pred = buf.argmax(1)
    return np.bincount(tc * 7 + pred, minlength=49).reshape(7, 7)


def miou(cm):
    tp = np.diag(cm).astype(np.float64)
    den = cm.sum(0) + cm.sum(1) - tp
    iou = np.where(den > 0, tp / np.maximum(den, 1), np.nan)
    return float(np.nanmean(iou)), iou


def coordinate_ascent(flat, tc, buf, rounds=3):
    """Coordinate ascent over the 7 per-class biases.

    Round 0 scans the coarse absolute grid; later rounds refine *around the
    current value* (cur + fine), otherwise a coarse optimum like +2.0 could
    never move further because the fine grid only contains small magnitudes.
    """
    delta = np.zeros(7, np.float32)
    best, _ = miou(confusion(flat, tc, delta, buf))
    start = best
    trace = []
    for r in range(rounds):
        for c in range(7):
            cur, bv, bm = delta[c], delta[c], best
            cands = GRID_COARSE if r == 0 else [cur + g for g in GRID_FINE]
            for v in cands:
                if abs(v - cur) < 1e-9:
                    continue
                delta[c] = v
                m, _ = miou(confusion(flat, tc, delta, buf))
                if m > bm + 1e-9:
                    bv, bm = v, m
            delta[c] = bv
            best = bm
        trace.append(float(best))
        print(f"  round {r + 1}: mIoU={best * 100:6.2f}  delta={np.round(delta, 2).tolist()}",
              flush=True)
    return delta, start, best, trace


def prior_delta(tau, pi):
    return (-tau * np.log(np.maximum(pi, 1e-8))).astype(np.float32)


def fit_tau(flat, tc, buf, pi):
    best_tau, best_m = 0.0, -1.0
    for t in TAUS:
        m, _ = miou(confusion(flat, tc, prior_delta(t, pi), buf))
        print(f"  tau={t:4.2f}  mIoU={m * 100:6.2f}")
        if m > best_m:
            best_tau, best_m = t, m
    return best_tau, best_m


@torch.no_grad()
def eval_pair(model, items, device, delta, tag, limit=0):
    """Full-resolution evaluation: baseline vs. delta-corrected, same pass."""
    cm0, cm1 = ConfusionMatrix(), ConfusionMatrix()
    d = torch.from_numpy(np.asarray(delta, np.float32)).view(7, 1, 1).to(device)
    items = items[:limit] if limit else items
    for i, (ip, mk, _) in enumerate(items, 1):
        p = predict_tile(model, prep(ip), device, return_prob=True)
        gt = imread(mk, cv2.IMREAD_GRAYSCALE)
        pred0 = (p.argmax(0) + 1).cpu().numpy()
        # correction is additive in LOG space: argmax(log p + delta)
        pred1 = ((p.clamp_min(1e-8).log() + d).argmax(0) + 1).cpu().numpy()
        cm0.update(pred0, gt)
        cm1.update(pred1, gt)
        if i % 100 == 0:
            print(f"  {tag} {i}/{len(items)}", flush=True)
    return cm0.compute(), cm1.compute()


def summarise(r):
    return dict(mIoU=float(r["mIoU"]), OA=float(r["OA"]), mF1=float(r["mF1"]),
                per_class={c: float(r["IoU"][i]) for i, c in enumerate(CLASSES)},
                recall={c: float(r["Recall"][i]) for i, c in enumerate(CLASSES)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="V1_ce")
    ap.add_argument("--model", default="v1")
    ap.add_argument("--config", default="code/configs/e0_ce.yaml")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--ds", type=int, default=4, help="search-resolution downsample factor")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cache-val", action="store_true", help="also fit delta on Val (ceiling)")
    ap.add_argument("--freq", default="data/stats/train_class_freq.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    root = cfg["paths"]["data"]
    exp = cfg["paths"]["experiments"]
    out_dir = os.path.join(exp, args.run)
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = args.ckpt or os.path.join(out_dir, "best.pth")
    print(f"checkpoint: {ckpt}  exists={os.path.exists(ckpt)}")
    model = load_model(args.model, ckpt, device)

    pi = np.array([json.load(open(args.freq, encoding="utf-8"))["freq"][c] for c in CLASSES],
                  np.float64)
    pi = pi / pi.sum()
    print("train class freq:", np.round(pi, 3).tolist())

    # ---- 1. dev cache + fit
    dev_items = build_index(root, "dev")
    print(f"\n[1] caching dev probabilities ({len(dev_items)} tiles, ds={args.ds})")
    dprob, dgt = build_cache(model, dev_items, device,
                             os.path.join(out_dir, f"prob_dev_ds{args.ds}.npz"), args.ds)
    dflat, dtc = flatten(dprob, dgt)
    buf = np.empty_like(dflat)
    base_m, base_iou = miou(confusion(dflat, dtc, np.zeros(7, np.float32), buf))
    print(f"  dev baseline mIoU={base_m * 100:.2f}  ({len(dtc) / 1e6:.1f}M px)")

    print("\n[2] fitting tau (prior form, 1 parameter)")
    tau, tau_m = fit_tau(dflat, dtc, buf, pi)
    d_tau = prior_delta(tau, pi)

    print("\n[3] fitting free delta (7 parameters, coordinate ascent)")
    d_free, _, free_m, trace = coordinate_ascent(dflat, dtc, buf, args.rounds)

    # ---- 2. apply on Val (full resolution, single pass)
    val_items = build_index(root, "val")
    n_val = args.limit or len(val_items)
    print(f"\n[4] applying to official Val ({n_val} tiles) -- dev-fitted, nothing re-fitted")
    v_base, v_tau = eval_pair(model, val_items, device, d_tau, "val/tau", args.limit)
    _, v_free = eval_pair(model, val_items, device, d_free, "val/free", args.limit)

    out = dict(run=args.run, model=args.model, ds=args.ds, n_dev=len(dev_items),
               n_val=len(val_items),
               dev=dict(baseline=base_m, tau=float(tau), tau_mIoU=float(tau_m),
                        free_mIoU=float(free_m), delta_free=d_free.tolist(),
                        delta_prior=d_tau.tolist(), trace=trace,
                        per_class_baseline=base_iou.tolist()),
               val=dict(baseline=summarise(v_base), with_tau=summarise(v_tau),
                        with_free=summarise(v_free)))

    # ---- 3. optional ceiling: fit on Val itself
    if args.cache_val:
        print("\n[5] ceiling: fitting delta ON Val (test-set fitting -- diagnostic only)")
        vprob, vgt = build_cache(model, val_items, device,
                                 os.path.join(out_dir, f"prob_val_ds{args.ds}.npz"), args.ds)
        vflat, vtc = flatten(vprob, vgt)
        vbuf = np.empty_like(vflat)
        vbase_m, _ = miou(confusion(vflat, vtc, np.zeros(7, np.float32), vbuf))
        v_delta, _, v_free_m, v_trace = coordinate_ascent(vflat, vtc, vbuf, args.rounds)
        out["val_ceiling"] = dict(baseline=float(vbase_m), fitted_mIoU=float(v_free_m),
                                  delta=v_delta.tolist(), trace=v_trace)

    path = os.path.join(out_dir, "threshold.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print("\n=== summary ===")
    print(f"dev   baseline {base_m * 100:6.2f} | tau({tau}) {tau_m * 100:6.2f} | "
          f"free {free_m * 100:6.2f}")
    print(f"val   baseline {v_base['mIoU'] * 100:6.2f} | dev-tau {v_tau['mIoU'] * 100:6.2f} | "
          f"dev-free {v_free['mIoU'] * 100:6.2f}")
    if args.cache_val:
        print(f"val   ceiling(fit-on-val) {out['val_ceiling']['fitted_mIoU'] * 100:6.2f}")
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
