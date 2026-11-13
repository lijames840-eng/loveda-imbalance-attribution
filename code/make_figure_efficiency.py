"""F11 效率—精度散点（论文核心对比图）。

x 轴用 GFLOPs 而非 FPS：本机是笔记本 RTX 3060，动态频率 1425–2100 MHz，
单张 512² 推理被 CPU launch overhead 主导（同一 v1 冷态 21.5 ms、热态 5.9 ms），
FPS 不可复现。GFLOPs 与参数量是硬件无关的可靠量。

同一骨干（v1，6.71 GFLOPs）上有 7 个点：V1/V2/V3/V5 + 三条消融。
不做 jitter——那条竖线本身就是结论：**同骨干的效率完全相同，差别只在精度**。
"""
from paths import EXPERIMENTS, FIGURES  # noqa: E402
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

EXP = str(EXPERIMENTS)
FIG = str(FIGURES)

# (label, run, model_key, params_M, GFLOPs, group, marker)
PTS = [
    ("V1", "V1_ce", "v1", 3.484, 6.71, "ours", "o"),
    ("V2", "V2", "v1", 3.484, 6.71, "ours", "o"),
    ("V3", "V3", "v3", 3.543, 6.71, "ours", "o"),
    ("V5", "V5", "v5", 3.447, 6.73, "ours", "o"),
    ("AB_w", "AB_w", "v1", 3.484, 6.71, "abl", "s"),
    ("AB_dice", "AB_dice", "v1", 3.484, 6.71, "abl", "s"),
    ("AB_bnd", "AB_bnd", "v1", 3.484, 6.71, "abl", "s"),
    ("E10 UNet-R34", "E10_unet", "unet_r34", 24.568, 36.69, "base", "D"),
    ("E11 DeepLabV3+", "E11_deeplab", "deeplabv3p_r50", 39.636, 163.57, "base", "D"),
    ("E12 SegFormer-B0", "E12_segformer", "segformer_b0", 3.454, 2.72, "base", "D"),
]

COLOR = {"ours": "#2b6cb0", "abl": "#2f855a", "base": "#6b46c1"}
# 手工指定标注偏移，避免同 x 的五个点标签互相压住
OFFSET = {
    "V2": (9, 7), "V3": (9, 3), "V1": (9, 5),
    "AB_dice": (12, 10), "V5": (12, -12), "AB_w": (10, -11), "AB_bnd": (10, -18),
    "E10 UNet-R34": (11, 7), "E11 DeepLabV3+": (-13, 11), "E12 SegFormer-B0": (11, 13),
}


def load_miou(run):
    p = f"{EXP}/{run}/eval_val.json"
    if os.path.exists(p):
        return json.load(open(p))["mIoU"] * 100
    p = f"{EXP}/{run}/pah_eval_val.json"
    if os.path.exists(p):
        return json.load(open(p))["mode_none"]["mIoU"] * 100
    return None


def main():
    rows = []
    for label, run, key, par, fl, grp, mk in PTS:
        m = load_miou(run)
        if m is None:
            print(f"  skip {label}: 无结果")
            continue
        rows.append(dict(label=label, miou=m, params=par, flops=fl, group=grp, marker=mk))

    # 气泡面积按 log(参数量) 线性映射到 [90, 420]；直接按 sqrt 会让 E11（39.6 M）涨到不可读
    lp0, lp1 = np.log10(3.0), np.log10(45.0)
    for r in rows:
        r["s"] = 90 + 330 * (np.log10(r["params"]) - lp0) / (lp1 - lp0)

    # 同一 GFLOPs 的点必须错开，否则 7 个点完全叠在一起无法分辨。
    # 偏移在**对数**轴上 ±0.035 dex（≈±8%），图注会明确声明这是可视化偏移。
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        groups[r["flops"]].append(r)
    for fl, grp in groups.items():
        grp.sort(key=lambda x: x["miou"])
        n = len(grp)
        for i, r in enumerate(grp):
            r["x"] = fl if n == 1 else fl * 10 ** ((i - (n - 1) / 2) * (0.07 / max(n - 1, 1)))

    fig, ax = plt.subplots(figsize=(7.8, 4.9), dpi=200)
    for r in rows:
        ax.scatter(r["x"], r["miou"], s=r["s"],
                   color=COLOR[r["group"]], marker=r["marker"],
                   alpha=0.85, edgecolors="white", linewidths=1.1, zorder=3)
        dx, dy = OFFSET.get(r["label"], (8, 6))
        ax.annotate(r["label"], (r["x"], r["miou"]), textcoords="offset points",
                    xytext=(dx, dy), fontsize=8, color="#2d3748",
                    ha="left" if dx > 0 else "right")

    # V2 的精度参考线（全文主模型）
    v2 = next(r for r in rows if r["label"] == "V2")
    ax.axhline(v2["miou"], color="#c05621", ls="--", lw=1.0, alpha=0.75, zorder=1)
    ax.text(2.55, v2["miou"] + 0.35, f"V2 = {v2['miou']:.2f}",
            fontsize=7.5, color="#c05621")

    ax.set_xscale("log")
    ax.set_xlabel("计算量 GFLOPs @512²（对数轴，越低越省）")
    ax.set_ylabel("Val mIoU（1669 幅全图滑窗）")
    # [图内不设标题/脚注，全部说明移至图题]
    ax.grid(alpha=0.28, ls=":", which="both")

    # 分组图例
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=COLOR["ours"], label="自研版本", markersize=7),
        plt.Line2D([], [], marker="s", ls="", color=COLOR["abl"], label="损失消融", markersize=7),
        plt.Line2D([], [], marker="D", ls="", color=COLOR["base"], label="外部基线", markersize=7),
    ]
    ax.legend(handles=handles, fontsize=8, loc="lower right", framealpha=0.92)

    ax.set_xlim(2.2, 260)
    ax.set_ylim(37.8, 52.5)
    fig.tight_layout()
    p = f"{FIG}/fig11_efficiency_accuracy.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("  ->", p)

    for r in sorted(rows, key=lambda x: -x["miou"]):
        print(f"  {r['label']:<18}{r['miou']:6.2f}  {r['flops']:7.2f} GFLOPs  {r['params']:6.3f} M")


if __name__ == "__main__":
    main()
