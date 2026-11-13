"""Class frequencies of the actual training split (2269 tiles, dev excluded).

Used to derive the GLOBAL class weights for the E2 ablation, so that E2 (global
re-weighting) and E3 (per-image re-weighting) differ in exactly one factor.
"""
from paths import DATA_RAW, DATA_STATS, SPLITS  # noqa: E402
import json

import cv2
import numpy as np
from imgio import imread  # noqa: E402

ROOT = str(DATA_RAW)
OUT = str(DATA_STATS / "train_class_freq.json")
CLASSES = ["background", "building", "road", "water", "barren", "forest", "agriculture"]


def main():
    ids = json.load(open(SPLITS))["train"]
    cnt = np.zeros(8, np.float64)
    for i, rel in enumerate(ids, 1):
        _, dom, tid = rel.split("/")
        m = imread(f"{ROOT}/Train/{dom}/masks_png/{tid}.png", cv2.IMREAD_GRAYSCALE)
        cnt += np.bincount(m.ravel(), minlength=8)[:8]
        if i % 500 == 0:
            print(f"  {i}/{len(ids)}", flush=True)

    labeled = cnt[1:]
    freq = labeled / labeled.sum()
    # inverse square-root, mean-normalised over the 7 real classes
    w = 1.0 / np.sqrt(freq)
    w = w / w.mean()

    print(f"\ntrain tiles: {len(ids)}   labeled px: {int(labeled.sum())/1e6:.1f} M")
    print(f"{'class':12s} {'freq %':>8s} {'inv-sqrt w':>11s}")
    for c, f, wi in zip(CLASSES, freq, w):
        print(f"{c:12s} {f*100:8.2f} {wi:11.3f}")

    json.dump({"n_tiles": len(ids),
               "freq": dict(zip(CLASSES, freq.tolist())),
               "inv_sqrt_weight": [0.0] + w.tolist()},   # index 0 = ignore class
              open(OUT, "w"), indent=1)
    print("\n(class_weights for yaml, index0=0) ->", [0.0] + [round(x, 3) for x in w.tolist()])
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
