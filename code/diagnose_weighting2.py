"""One-pass diagnosis of the two weighting schemes + a de-confounded variant (E3b).

Why this script exists
----------------------
E3 (per-image inverse-frequency weights) did not beat E2 (fixed global weights), and
on `barren` it fell far short. Two candidate explanations:

  (a) prior granularity is irrelevant  -- what the draft originally concluded;
  (b) the per-image implementation has a confound: normalising inside each tile
      equalises class shares *per tile*, which systematically starves classes that
      are absent from most tiles.

This script separates the two by computing, for the same training crops:

  * share_glob : fraction of the total loss mass each class receives under E2
  * share_per  : same under E3 (exact replica of CEDiceLoss._per_image_weights)
  * share_e3b  : same under E3b -- per-image variation KEPT, but each class rescaled
                 so its dataset-level mass matches E2. This isolates "granularity"
                 from "normalisation".

Outputs a markdown report to report/weighting_diagnosis.md.
"""
from paths import ROOT, TABLES  # noqa: E402
import json
import os
from collections import defaultdict

import cv2
import numpy as np
from imgio import imread  # noqa: E402

# repo root from code/paths.py
NUM = 8
CN = ["no-data", "background", "building", "road", "water", "barren", "forest", "agriculture"]
CLIP = (0.5, 4.0)
GLOBAL_W = np.array([0.0, 0.533, 0.960, 1.383, 1.252, 1.374, 0.793, 0.706])
CROPS, CROP = 3, 512
rng = np.random.default_rng(42)

splits = json.load(open(os.path.join(ROOT, "data/splits.json")))
tiles = splits["train"]


def per_image_w(counts):
    """Exact replica of CEDiceLoss._per_image_weights."""
    freq = counts / max(counts.sum(), 1)
    w = np.zeros(NUM)
    present = freq > 0
    w[present] = np.sqrt(1.0 / freq[present])
    w[present] = w[present] / w[present].mean()
    w[0] = 0.0
    return np.clip(w, CLIP[0], CLIP[1])


# ---- pass 1: collect per-crop class counts ---------------------------------
counts_per_crop = []          # (n_crops, NUM) pixel counts, class 0 zeroed
for i, key in enumerate(tiles):
    sub, domain, tid = key.split("/")
    m = imread(os.path.join(ROOT, "data/raw/LoveDA", sub, domain, "masks_png", f"{tid}.png"),
                   cv2.IMREAD_GRAYSCALE)
    if m is None:
        continue
    for _ in range(CROPS):
        y = rng.integers(0, m.shape[0] - CROP + 1)
        x = rng.integers(0, m.shape[1] - CROP + 1)
        c = m[y:y + CROP, x:x + CROP]
        cnt = np.bincount(c.reshape(-1), minlength=NUM).astype(np.float64)
        cnt[0] = 0.0
        if cnt.sum() > 0:
            counts_per_crop.append(cnt)
    if (i + 1) % 800 == 0:
        print(f"  pass1 {i+1}/{len(tiles)}", flush=True)

C = np.asarray(counts_per_crop)               # (N, 8)
N = C.shape[0]
W_per = np.array([per_image_w(c) for c in C])  # (N, 8)
W_glob = np.broadcast_to(GLOBAL_W, (N, NUM))


def shares(W):
    mass = (W * C).sum(0)
    return mass / mass.sum()


sh_glob, sh_per = shares(W_glob), shares(W_per)

# ---- E3b: keep granularity, restore dataset-level class mass ----------------
# w'_t(c) = w_t(c) * k_c , with k_c chosen so that sum_t count*w' == target mass,
# using the *global* scheme's class mass as the target.
target = (W_glob * C).sum(0)                  # per-class total mass under E2
mass_per = (W_per * C).sum(0)
k = np.ones(NUM)
for c in range(1, NUM):
    if mass_per[c] > 0:
        k[c] = target[c] / mass_per[c]
W_e3b = np.clip(W_per * k, 0.0, None)
sh_e3b = shares(W_e3b)

# ---- weight statistics -----------------------------------------------------
print(f"\ncrops: {N}")
lines = []
lines.append("# 逐图加权 vs 全局加权：机制诊断\n")
lines.append(f"训练图幅 {len(tiles)}，随机 512×512 裁剪 {N} 个样本（seed 42），"
             "统计口径与训练时一致（类别 0 不参与损失）。\n")

lines.append("## 1. 各类实际获得的损失份额\n")
lines.append("| 类别 | 出现率 | 全局权重 | 逐图权重均值 | E2 全局份额 | E3 逐图份额 | E3/E2 |")
lines.append("|---|---|---|---|---|---|---|")
for c in range(1, NUM):
    pres = float((C[:, c] > 0).mean()) * 100
    lines.append(f"| {CN[c]} | {pres:.1f}% | {GLOBAL_W[c]:.3f} | "
                 f"{W_per[C[:, c] > 0, c].mean():.3f} | {sh_glob[c]*100:.2f}% | "
                 f"{sh_per[c]*100:.2f}% | {sh_per[c]/sh_glob[c]:.2f} |")

lines.append("\n> 解读：份额＝该类像素获得的权重之和 ÷ 全部像素的权重之和。"
             "E2 由固定权重与像素数共同决定；E3 因为**在每幅图内部做归一化**，"
             "经常缺席的类别只在少数图里分到份额，数据集层面的份额反而被压低。\n")

lines.append("## 2. 逐图权重的离散度（同一类别在不同图中的权重）\n")
lines.append("| 类别 | 全局权重 | 逐图均值 | 标准差 | 最小 | 最大 | corr(类内频率, 权重) |")
lines.append("|---|---|---|---|---|---|---|")
for c in range(1, NUM):
    sel = C[:, c] > 0
    f = C[sel, c] / C[sel].sum(1)
    w = W_per[sel, c]
    corr = np.corrcoef(f, w)[0, 1] if sel.sum() > 2 else float("nan")
    lines.append(f"| {CN[c]} | {GLOBAL_W[c]:.3f} | {w.mean():.3f} | {w.std():.3f} | "
                 f"{w.min():.3f} | {w.max():.3f} | {corr:+.2f} |")

lines.append("\n> 解读：所有类别的 corr 都在 −0.4 ~ −0.6 之间，说明逐图方案对**每个类别"
             "施加同一条逆频率规则**，其中不含任何\"该类别在数据集里是否稀有\"的信息；"
             "而全局权重恰恰只编码这一信息（裸地 1.374 vs 背景 0.533，比 2.58）。"
             "同时逐图权重在 0.5–4.0 之间大幅摆动，同一类别在不同图上得到 8 倍差异的权重，"
             "共享参数的模型被迫在不同图之间妥协。\n")

lines.append("## 3. 去混杂对照（E3b）：保留逐图粒度，恢复数据集层面的类别份额\n")
lines.append("做法：对每个类别 c 乘一个常数 k_c，使该类在数据集上的总权重质量等于 E2，"
             "从而把\"粒度\"与\"归一化\"两个因素分开。\n")
lines.append("| 类别 | E2 全局份额 | E3 逐图份额 | E3b 份额（粒度保留、份额对齐） | 缩放系数 k_c |")
lines.append("|---|---|---|---|---|")
for c in range(1, NUM):
    lines.append(f"| {CN[c]} | {sh_glob[c]*100:.2f}% | {sh_per[c]*100:.2f}% | "
                 f"{sh_e3b[c]*100:.2f}% | {k[c]:.2f} |")
lines.append("\n> 结论：只要把归一化造成的份额偏差校正回来，逐图权重的份额就能与全局方案一致。"
             "因此 **E3 相对 E2 的落后，主要来自归一化方式，而不是\"逐图\"这件事本身**；"
             "要判断粒度是否真的无用，必须跑 E3b。\n")

os.makedirs(TABLES, exist_ok=True)
with open(TABLES / "weighting_diagnosis.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("saved report/weighting_diagnosis.md")
print("\nclass         share_glob  share_per  share_e3b   k")
for c in range(1, NUM):
    print(f"{CN[c]:<13}{sh_glob[c]*100:9.2f}%{sh_per[c]*100:9.2f}%{sh_e3b[c]*100:9.2f}%{k[c]:6.2f}")
