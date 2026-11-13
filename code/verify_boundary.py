"""Verify the boundary-band error decomposition from scratch (uses cached preds)."""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_RAW, EXPERIMENTS  # noqa: E402
from loveda_dataset import CLASSES, NUM_CLASSES, build_index  # noqa: E402
from imgio import imread  # noqa: E402

ROOT = str(DATA_RAW)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="V1_ce", help="experiment dir holding analysis/preds.npy")
    args = ap.parse_args()
    cache = f"{EXPERIMENTS}/{args.run}/analysis/preds.npy"
    items = build_index(ROOT, "val")
    preds = np.load(cache, mmap_mode="r")
    k = np.ones((3, 3), np.uint8)
    tot_in = np.zeros(NUM_CLASSES, np.int64)
    tot_out = np.zeros(NUM_CLASSES, np.int64)
    err_in = np.zeros(NUM_CLASSES, np.int64)
    err_out = np.zeros(NUM_CLASSES, np.int64)

    for i, (_, mpath, _) in enumerate(items):
        gt = imread(mpath, cv2.IMREAD_GRAYSCALE)
        pred = preds[i]
        band = np.zeros(gt.shape, bool)
        for c in range(1, NUM_CLASSES):
            m = (gt == c).astype(np.uint8)
            if m.sum():
                band |= cv2.morphologyEx(m, cv2.MORPH_GRADIENT, k).astype(bool)
        wrong = pred != gt
        for c in range(1, NUM_CLASSES):
            g = gt == c
            tot_in[c] += int((g & band).sum())
            tot_out[c] += int((g & ~band).sum())
            err_in[c] += int((g & band & wrong).sum())
            err_out[c] += int((g & ~band & wrong).sum())
        if (i + 1) % 400 == 0:
            print(f"  {i+1}/{len(items)}", flush=True)

    print(f"\n{'class':12s} {'band%ofGT':>10s} {'errRate_in':>11s} {'errRate_out':>12s} "
          f"{'share_of_err':>13s} {'band_err/total_band':>20s}")
    for c in range(1, NUM_CLASSES):
        tot = tot_in[c] + tot_out[c]
        e_in, e_out = err_in[c], err_out[c]
        print(f"{CLASSES[c-1]:12s} {tot_in[c]/max(tot,1)*100:10.1f} "
              f"{e_in/max(tot_in[c],1)*100:11.1f} {e_out/max(tot_out[c],1)*100:12.1f} "
              f"{e_in/max(e_in+e_out,1)*100:13.1f} "
              f"{tot_in[c]/max(tot_in.sum(),1)*100:20.1f}")

    band_all = tot_in.sum()
    err_all = err_in.sum() + err_out.sum()
    print(f"\nband pixels = {band_all/ (tot_in.sum()+tot_out.sum())*100:.1f}% of all GT pixels")
    print(f"band errors = {err_in.sum()/max(err_all,1)*100:.1f}% of all errors")
    print(f"error rate: in-band {err_in.sum()/max(band_all,1)*100:.1f}%  "
          f"out-band {err_out.sum()/max(tot_out.sum(),1)*100:.1f}%")


if __name__ == "__main__":
    main()
