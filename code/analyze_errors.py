"""Error analysis for the E0 baseline (course requirement (3): performance analysis).

Produces the evidence that drives the V2 design:
  1. global confusion matrix (row-normalised)      -> which pairs are confused
  2. per-class IoU / Recall / Precision            -> the long-tail table
  3. boundary-band error decomposition             -> is the error localised on edges?
  4. per-domain (urban/rural) per-class recall     -> domain prior effect
  5. linear probe on frozen features (H2)          -> representation vs decision

Usage:
    python code/analyze_errors.py --run V1_ce --model v1 --limit 0
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
from loveda_dataset import CLASSES, MEAN, STD, NUM_CLASSES, build_index  # noqa: E402
from metrics import ConfusionMatrix  # noqa: E402
from evaluate import load_model, predict_tile  # noqa: E402
from imgio import imread  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="V1_ce")
    ap.add_argument("--model", default="v1")
    ap.add_argument("--config", default="code/configs/e0_ce.yaml")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--cache", action="store_true", help="reuse cached predictions if present")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    out_dir = args.out or os.path.join(cfg["paths"]["experiments"], args.run, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.model, os.path.join(cfg["paths"]["experiments"], args.run, "best.pth"), device)
    items = build_index(cfg["paths"]["data"], "val")
    if args.limit:
        items = items[: args.limit]

    cm = ConfusionMatrix()
    cm_urban, cm_rural = ConfusionMatrix(), ConfusionMatrix()
    # error counts stratified by "is this GT pixel within 1 px of a class boundary"
    err_in = np.zeros(NUM_CLASSES, dtype=np.int64)
    tot_in = np.zeros(NUM_CLASSES, dtype=np.int64)
    err_out = np.zeros(NUM_CLASSES, dtype=np.int64)
    tot_out = np.zeros(NUM_CLASSES, dtype=np.int64)

    cache_path = os.path.join(out_dir, "preds.npy")
    use_cache = args.cache and os.path.exists(cache_path)
    store = None
    if args.cache and not use_cache:
        store = np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.uint8,
                                          shape=(len(items), 1024, 1024))

    k = np.ones((3, 3), np.uint8)
    for i, (ipath, mpath, domain) in enumerate(items, 1):
        gt = imread(mpath, cv2.IMREAD_GRAYSCALE)
        if use_cache:
            pred = np.load(cache_path, mmap_mode="r")[i - 1]
        else:
            img = imread(ipath, cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32) / 255.0
            x = torch.from_numpy(((img - MEAN) / STD).transpose(2, 0, 1).copy())
            pred = predict_tile(model, x, device).cpu().numpy()
            if pred.shape != gt.shape:
                pred = cv2.resize(pred.astype(np.uint8), (gt.shape[1], gt.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)

        if store is not None:
            store[i - 1] = pred
        cm.update(pred, gt)
        (cm_urban if domain == "urban" else cm_rural).update(pred, gt)

        # boundary band from GT: pixel is "boundary" if the 3x3 window covers >=2 classes
        band = np.zeros(gt.shape, bool)
        for c in range(1, NUM_CLASSES):
            m = (gt == c).astype(np.uint8)
            if m.sum() == 0:
                continue
            band |= cv2.morphologyEx(m, cv2.MORPH_GRADIENT, k).astype(bool)
        wrong = (pred != gt)
        for c in range(1, NUM_CLASSES):
            g = gt == c
            if g.sum() == 0:
                continue
            tot_in[c] += int((g & band).sum())
            err_in[c] += int((g & band & wrong).sum())
            tot_out[c] += int((g & ~band).sum())
            err_out[c] += int((g & ~band & wrong).sum())

        if i % 100 == 0:
            print(f"  {i}/{len(items)}", flush=True)

    res = cm.compute()
    # ---- tables ----
    import pandas as pd
    rows = []
    for j, c in enumerate(CLASSES):
        rows.append(dict(cls=c, IoU=res["IoU"][j], Recall=res["Recall"][j],
                         Precision=res["Precision"][j], F1=res["F1"][j],
                         GT_px=int(tot_in[j + 1] + tot_out[j + 1]),
                         err_in_band=float(err_in[j + 1] / max(tot_in[j + 1], 1)),
                         err_out_band=float(err_out[j + 1] / max(tot_out[j + 1], 1)),
                         band_share_of_errors=float(err_in[j + 1] / max(err_in[j + 1] + err_out[j + 1], 1))))
    df = pd.DataFrame(rows)
    df["GT_ratio"] = df.GT_px / df.GT_px.sum()
    df.to_csv(os.path.join(out_dir, "per_class.csv"), index=False)

    np.save(os.path.join(out_dir, "confusion.npy"), cm.mat)
    pd.DataFrame(cm.mat[1:, 1:], index=CLASSES, columns=CLASSES).to_csv(os.path.join(out_dir, "confusion.csv"))
    summary = dict(
        mIoU=res["mIoU"], OA=res["OA"], mF1=res["mF1"],
        urban_mIoU=cm_urban.compute()["mIoU"], rural_mIoU=cm_rural.compute()["mIoU"],
        urban_recall=dict(zip(CLASSES, cm_urban.compute()["Recall"].tolist())),
        rural_recall=dict(zip(CLASSES, cm_rural.compute()["Recall"].tolist())),
        n_tiles=len(items),
        overall_band_share_of_errors=float((err_in.sum()) / max(err_in.sum() + err_out.sum(), 1)),
    )
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)

    print("\n" + df.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\nmIoU=%.2f OA=%.2f  urban=%.2f rural=%.2f" % (
        res["mIoU"] * 100, res["OA"] * 100, summary["urban_mIoU"] * 100, summary["rural_mIoU"] * 100))
    print("boundary band holds %.1f%% of all pixel errors" % (summary["overall_band_share_of_errors"] * 100))
    print("saved ->", out_dir)


if __name__ == "__main__":
    main()
