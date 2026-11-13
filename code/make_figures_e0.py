"""Figures from analyze_errors.py output (problem-exposure section of the paper)."""
from paths import EXPERIMENTS, FIGURES  # noqa: E402
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CLS_CN = {"background": "背景", "building": "建筑", "road": "道路", "water": "水体",
          "barren": "裸地", "forest": "林地", "agriculture": "农田"}
ORDER = list(CLS_CN)


def fig_longtail(an_dir, out):
    df = pd.read_csv(f"{an_dir}/per_class.csv").set_index("cls").loc[ORDER]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.scatter(df.GT_ratio * 100, df.Recall * 100, s=90, c="#d64545", zorder=3)
    for c in ORDER:
        ax.annotate(CLS_CN[c], (df.loc[c, "GT_ratio"] * 100, df.loc[c, "Recall"] * 100),
                    textcoords="offset points", xytext=(7, 5), fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("该类别占测试集像素比例 (%)  — 对数坐标")
    ax.set_ylabel("逐类召回率 (%)")
    ax.set_title("类别频率与召回率：被压制的少数类")
    ax.grid(alpha=.3, which="both")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_confusion(an_dir, out):
    cm = np.load(f"{an_dir}/confusion.npy").astype(float)[1:, 1:]
    cm = cm / cm.sum(1, keepdims=True).clip(min=1) * 100
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(cm, cmap="YlOrRd", vmin=0, vmax=100)
    ax.set_xticks(range(7), [CLS_CN[c] for c in ORDER], rotation=35, ha="right")
    ax.set_yticks(range(7), [CLS_CN[c] for c in ORDER])
    for i in range(7):
        for j in range(7):
            if cm[i, j] >= 1:
                ax.text(j, i, f"{cm[i,j]:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > 55 else "#333")
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title("混淆矩阵（按行归一化，单位 %）")
    fig.colorbar(im, ax=ax, fraction=.046, pad=.03)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_boundary(an_dir, out):
    df = pd.read_csv(f"{an_dir}/per_class.csv").set_index("cls").loc[ORDER]
    x = np.arange(len(ORDER))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(x - w / 2, df.err_in_band * 100, w, label="边界带内像素错误率", color="#d64545")
    ax.bar(x + w / 2, df.err_out_band * 100, w, label="区域内部像素错误率", color="#3a7bd5")
    ax.set_xticks(x, [CLS_CN[c] for c in ORDER])
    ax.set_ylabel("像素错误率 (%)")
    ax.set_title("错误的空间分布：边界带 vs 区域内部")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="V1_ce", help="experiment dir under experiments/")
    ap.add_argument("--out", default=str(FIGURES))
    args = ap.parse_args()
    an = f"{EXPERIMENTS}/{args.run}/analysis"
    out = args.out
    os.makedirs(out, exist_ok=True)
    fig_longtail(an, f"{out}/fig4_longtail_v1.png")
    fig_confusion(an, f"{out}/fig5_confusion_v1.png")
    fig_boundary(an, f"{out}/fig6_boundary_error_v1.png")
    print("figures written to", out)
