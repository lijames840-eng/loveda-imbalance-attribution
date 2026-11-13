"""Calibrate the per-class factors that align per-image class shares.

Why this exists
---------------
`CEDiceLoss._per_image_weights` normalises inverse-frequency weights *inside each
tile*.  That answers "how rare is this class in this scene", but cross-entropy
reweighting needs "how rare is this class across the dataset".  The measured
consequence (report/weighting_diagnosis.md) was background receiving 1.38x its
global share and barren only 0.60x -- which matches exactly where the per-image
run gained and lost IoU.

This script fits one fixed factor k_c per class such that, after multiplying the
per-image weights by k_c, every class receives the **same dataset-level loss mass
as the global scheme**.  Per-image variation is preserved; only the normalisation
bias is removed.

k_c depends on the crop size, the clip range and the sampling distribution, so it
must be recomputed if any of those change -- that is why it is a script and not a
hard-coded constant.

Output: data/stats/per_image_share_scale.json
"""
from paths import ROOT  # noqa: E402
import argparse
import json
import os

import cv2
import numpy as np
from imgio import imread  # noqa: E402

CLASSES = ["background", "building", "road", "water", "barren", "forest", "agriculture"]
NUM = 8


def per_image_w(counts, clip):
    """Exact replica of CEDiceLoss._per_image_weights (before the share scale)."""
    freq = counts / max(counts.sum(), 1)
    w = np.zeros(NUM)
    present = freq > 0
    w[present] = np.sqrt(1.0 / freq[present])
    w[present] = w[present] / w[present].mean()
    w[0] = 0.0
    return np.clip(w, clip[0], clip[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--crops-per-tile", type=int, default=3)
    ap.add_argument("--clip", type=float, nargs=2, default=[0.5, 4.0])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target", default="global", choices=["global"],
                    help="align each class's dataset-level loss mass with the global "
                         "sqrt-inverse-frequency scheme (E2); this is the E3b definition")
    ap.add_argument("--freq", default="data/stats/train_class_freq.json")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--out", default="data/stats/per_image_share_scale.json")
    ap.add_argument("--report", default="report/per_image_share_scale.md")
    args = ap.parse_args()

    root = args.root
    freq_json = json.load(open(os.path.join(root, args.freq), encoding="utf-8"))
    global_w = np.array(freq_json["inv_sqrt_weight"], dtype=np.float64)   # 8-dim, [0]=0
    tiles = json.load(open(os.path.join(root, args.splits), encoding="utf-8"))["train"]
    print(f"train tiles: {len(tiles)}   global weights: {np.round(global_w, 3).tolist()}")

    rng = np.random.default_rng(args.seed)
    counts = []
    for i, key in enumerate(tiles):
        sub, domain, tid = key.split("/")
        mp = os.path.join(root, "data/raw/LoveDA", sub, domain, "masks_png", f"{tid}.png")
        m = imread(mp, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        for _ in range(args.crops_per_tile):
            y = rng.integers(0, max(m.shape[0] - args.crop, 0) + 1)
            x = rng.integers(0, max(m.shape[1] - args.crop, 0) + 1)
            c = m[y:y + args.crop, x:x + args.crop]
            cnt = np.bincount(c.reshape(-1), minlength=NUM).astype(np.float64)[:NUM]
            cnt[0] = 0.0
            if cnt.sum() > 0:
                counts.append(cnt)
        if (i + 1) % 800 == 0:
            print(f"  sample {i + 1}/{len(tiles)}", flush=True)

    C = np.asarray(counts)
    print(f"crops sampled: {C.shape[0]}")

    W_per = np.array([per_image_w(c, args.clip) for c in C])
    mass_per = (W_per * C).sum(0)

    target = (global_w * C).sum(0)
    target_desc = "对齐全局逆平方根方案（E2）的类别份额"

    k = np.ones(NUM)
    for c in range(1, NUM):
        if mass_per[c] > 1e-12:
            k[c] = target[c] / mass_per[c]

    share_per = mass_per / mass_per.sum()
    share_tgt = target / target.sum()
    mass_after = mass_per * k
    share_after = mass_after / mass_after.sum()

    out = dict(crop=args.crop, crops_per_tile=args.crops_per_tile, clip=args.clip,
               seed=args.seed, n_crops=int(C.shape[0]), target=args.target,
               k={CLASSES[i - 1]: float(k[i]) for i in range(1, NUM)},
               k_full=k.tolist(),
               share_before={CLASSES[i - 1]: float(share_per[i]) for i in range(1, NUM)},
               share_after={CLASSES[i - 1]: float(share_after[i]) for i in range(1, NUM)},
               share_target={CLASSES[i - 1]: float(share_tgt[i]) for i in range(1, NUM)})
    os.makedirs(os.path.dirname(os.path.join(root, args.out)), exist_ok=True)
    with open(os.path.join(root, args.out), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"saved -> {args.out}")

    lines = ["# 逐图权重的份额对齐系数\n",
             f"目标：{target_desc}。裁剪 {args.crop}²，每图 {args.crops_per_tile} 个，"
             f"共 {C.shape[0]} 个样本（seed {args.seed}），clip={args.clip}。\n",
             "| 类别 | 对齐前份额 | 对齐后份额 | 目标份额 | 系数 k_c |",
             "|---|---|---|---|---|"]
    for i in range(1, NUM):
        lines.append(f"| {CLASSES[i - 1]} | {share_per[i] * 100:.2f}% | "
                     f"{share_after[i] * 100:.2f}% | {share_tgt[i] * 100:.2f}% | "
                     f"{k[i]:.3f} |")
    lines.append("\n> 用法：`CEDiceLoss(per_image=True, share_scale=<k_full>)`。"
                 "系数只在裁剪尺寸、clip 或采样分布改变时需要重算。\n")
    with open(os.path.join(root, args.report), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"saved -> {args.report}")
    print("\nclass         before    after   target       k")
    for i in range(1, NUM):
        print(f"{CLASSES[i - 1]:<13}{share_per[i] * 100:7.2f}%{share_after[i] * 100:8.2f}%"
              f"{share_tgt[i] * 100:8.2f}%{k[i]:8.3f}")


if __name__ == "__main__":
    main()
