"""Quantitative characterisation of LoveDA for the course report.

Produces the evidence that backs the paper's problem statement:
  1) class_pixels.csv    - pixel frequency per class, per split/domain  -> class imbalance
  2) per_image.csv       - per-tile class mix, boundary ratio           -> heterogeneity
  3) class_geometry.csv  - connected-component area / aspect ratio      -> scale & shape variation
  4) class_adjacency.csv - how often two classes touch along borders    -> prior confusion pairs

Usage:
    python code/dataset_statistics.py --root data/raw/LoveDA \
        --out data/stats --workers 8
"""
from paths import DATA_RAW, DATA_STATS  # noqa: E402
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import pandas as pd
from imgio import imread  # noqa: E402

CLASSES = {1: "background", 2: "building", 3: "road", 4: "water", 5: "barren", 6: "forest", 7: "agriculture"}
SUBSETS = ["Train/Urban", "Train/Rural", "Val/Urban", "Val/Rural"]
FG = [1, 2, 3, 4, 5, 6, 7]  # all seven labelled classes; 0 is no-data and ignored


def list_pairs(root, sub):
    img_dir = os.path.join(root, sub, "images_png")
    msk_dir = os.path.join(root, sub, "masks_png")
    ids = sorted(f[:-4] for f in os.listdir(img_dir) if f.endswith(".png"))
    return [(sub, i, os.path.join(img_dir, i + ".png"), os.path.join(msk_dir, i + ".png")) for i in ids]


def analyse_one(args):
    sub, tid, ipath, mpath = args
    img = imread(ipath, cv2.IMREAD_COLOR)          # BGR
    msk = imread(mpath, cv2.IMREAD_GRAYSCALE)
    h, w = msk.shape

    counts = np.bincount(msk.ravel(), minlength=8)[:8]
    total = counts.sum()

    # --- boundary band: morphological gradient per class, measured on a 2x downsample ---
    small = msk[::2, ::2]
    k = np.ones((3, 3), np.uint8)
    bnd_total, cls_rows = 0, []
    for c in list(CLASSES) + [7]:
        m = (small == c).astype(np.uint8)
        if m.sum() == 0:
            continue
        edge = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, k)
        n_edge = int(edge.sum())
        bnd_total += n_edge
        area = int(m.sum())

        # geometry: connected components on the downsampled mask
        ncomp, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if ncomp > 1:
            a = stats[1:, cv2.CC_STAT_AREA]
            ww = stats[1:, cv2.CC_STAT_WIDTH].astype(float)
            hh = stats[1:, cv2.CC_STAT_HEIGHT].astype(float) + 1e-6
            ar = np.maximum(ww / hh, hh / ww)
            cls_rows.append(dict(split=sub, tile=tid, cls=c, n_objects=int(ncomp - 1),
                                 area_med=float(np.median(a)), area_p90=float(np.percentile(a, 90)),
                                 area_max=int(a.max()), aspect_p90=float(np.percentile(ar, 90)),
                                 edge_px=n_edge, area_px=area))

    # --- adjacency: dilate class i by 3x3, count overlap with class j ---
    adj = {}
    small_fg = {c: (small == c).astype(np.uint8) for c in FG}
    for i in FG:
        if small_fg[i].sum() == 0:
            continue
        dil = cv2.dilate(small_fg[i], k, iterations=1)
        for j in FG:
            if j <= i or small_fg[j].sum() == 0:
                continue
            n = int(np.logical_and(dil, small_fg[j]).sum())
            if n > 0:
                adj[(i, j)] = n

    entropy = 0.0
    p = counts / max(total, 1)
    nz = p[p > 0]
    entropy = float(-(nz * np.log(nz)).sum())

    row = dict(split=sub, tile=tid, height=h, width=w, total_px=int(total),
               boundary_px=int(bnd_total), boundary_frac=float(bnd_total / max(small.size, 1)),
               n_present=int(sum(counts[c] > 0 for c in CLASSES)),
               entropy=entropy,
               dominant=CLASSES[max(CLASSES, key=lambda c: counts[c])],
               mean_rgb_b=float(img[:, :, 0].mean()), mean_rgb_g=float(img[:, :, 1].mean()),
               mean_rgb_r=float(img[:, :, 2].mean()))
    for c, name in CLASSES.items():
        row[f"px_{name}"] = int(counts[c])
    return row, cls_rows, adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DATA_RAW))
    ap.add_argument("--out", default=str(DATA_STATS))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N tiles per subset")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    jobs = []
    for sub in SUBSETS:
        pairs = list_pairs(args.root, sub)
        if args.limit:
            pairs = pairs[: args.limit]
        print(f"{sub:14s} {len(pairs):5d} tiles", flush=True)
        jobs.extend(pairs)

    rows, geo, adj_acc = [], [], {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(analyse_one, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            row, cls_rows, adj = f.result()
            rows.append(row)
            geo.extend(cls_rows)
            for k, v in adj.items():
                adj_acc[(row["split"], k[0], k[1])] = adj_acc.get((row["split"], k[0], k[1]), 0) + v
            if i % 200 == 0 or i == len(futs):
                print(f"  processed {i}/{len(futs)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "per_image.csv"), index=False)
    pd.DataFrame(geo).to_csv(os.path.join(args.out, "class_geometry.csv"), index=False)
    pd.DataFrame([dict(split=a, cls_i=b, cls_j=c, n=n) for (a, b, c), n in adj_acc.items()]) \
        .to_csv(os.path.join(args.out, "class_adjacency.csv"), index=False)

    # class pixel totals
    out = []
    for split in df["split"].unique():
        d = df[df["split"] == split]
        tot = d["total_px"].sum()
        for name in CLASSES.values():
            px = d[f"px_{name}"].sum()
            out.append(dict(split=split, cls=name, pixels=int(px), ratio=float(px / tot)))
    pd.DataFrame(out).to_csv(os.path.join(args.out, "class_pixels.csv"), index=False)

    summary = dict(n_tiles=int(len(df)),
                   boundary_frac_mean=float(df["boundary_frac"].mean()),
                   tiles_by_split=df.groupby("split").size().to_dict())
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("done ->", args.out, summary["n_tiles"], "tiles")


if __name__ == "__main__":
    main()
