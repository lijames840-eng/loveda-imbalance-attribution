"""F4/F5/F6/F13 —— 不依赖 GPU 的四张分析图。

数据来源全部是已有产物，不重新推理：
  F4  长尾分布      <- data/stats/train_class_freq.json + class_geometry.csv
  F5  混淆矩阵      <- experiments/{V1_ce,V2}/cm_val.json（行归一化）
  F6  类间邻接      <- data/stats/class_adjacency.csv
  F13 落差分解      <- experiments/gap_decompose.json

诚实标注：F6 画的是"类间邻接强度"，不是"误差随边界距离的分层"——后者需要逐像素
预测与距离变换，本轮未重算。图注里写明，避免与 stage0 里"边界带仅承载 4.0% 误差"
那句话混淆（那是旧协议下的统计）。
"""
from paths import FIGURES, ROOT  # noqa: E402
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# repo root from code/paths.py
FIG = str(FIGURES)
CLS = ["背景", "建筑", "道路", "水体", "裸地", "林地", "农田"]
CLS_EN = ["background", "building", "road", "water", "barren", "forest", "agriculture"]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
C_MAIN, C_ALT, C_GRAY = "#2b6cb0", "#c05621", "#718096"


def fig4():
    freq = json.load(open(f"{ROOT}/data/stats/train_class_freq.json"))["freq"]
    geo = pd.read_csv(f"{ROOT}/data/stats/class_geometry.csv")
    geo = geo[geo["split"].str.startswith("Train/")]

    order = sorted(CLS_EN, key=lambda c: -freq[c])
    vals = [freq[c] * 100 for c in order]
    # 每类的平均目标面积中位数（像素）
    area = [geo[geo["cls"] == i + 1]["area_med"].median() for i in range(7)]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    xs = np.arange(len(order))
    bars = ax.bar(xs, vals, color=[C_MAIN if v > 5 else C_ALT for v in vals])
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([CLS[CLS_EN.index(c)] for c in order], fontsize=9)
    ax.set_ylabel("像素占比 %（对数轴）")
    # [图内不设标题/脚注，全部说明移至图题]
    for x, v in zip(xs, vals):
        ax.text(x, v * 1.15, f"{v:.1f}", ha="center", fontsize=8)
    ax.grid(axis="y", alpha=0.3, ls=":")
    ax.set_ylim(0.5, max(vals) * 2.2)
    ax.text(0.98, 0.92, f"最多/最少 = {max(vals)/min(vals):.1f}×",
            transform=ax.transAxes, fontsize=8.5, color=C_GRAY, ha="right")

    ax = axes[1]
    # 箱线图版本没有区分度（每个类跨越 1~1e5 像素，所有箱几乎重合），改用聚合后的
    # 平均单目标面积 = 该类总像素 / 该类目标数，才能看出尺度跨度。
    g = geo.groupby("cls").agg(area=("area_px", "sum"), n=("n_objects", "sum"))
    avg = (g["area"] / g["n"].replace(0, np.nan)).reindex(range(1, 8))
    xs2 = [freq[c] * 100 for c in CLS_EN]
    ax.scatter(xs2, avg.values, s=70, c=[C_MAIN if v > 5 else C_ALT for v in xs2], zorder=3)
    for x_, y_, cname in zip(xs2, avg.values, CLS):
        ax.annotate(cname, (x_, y_), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("像素占比 %（对数轴）", fontsize=9)
    ax.set_ylabel("平均单目标面积 / 像素（对数轴）", fontsize=9)
    # [图内不设标题/脚注，全部说明移至图题]
    ax.grid(alpha=0.3, ls=":")
    # [图内不设标题/脚注，全部说明移至图题]
    fig.tight_layout()
    p = f"{FIG}/fig4_longtail.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("  ->", p)


def fig5():
    cms = {}
    for run, lb in (("V1_ce", "V1（朴素 CE）"), ("V2", "V2（三件套）")):
        cm = np.array(json.load(open(f"{ROOT}/experiments/{run}/cm_val.json")), float)[1:, 1:]
        cms[lb] = cm / np.maximum(cm.sum(1, keepdims=True), 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, (lb, m) in zip(axes, cms.items()):
        im = ax.imshow(m, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(7))
        ax.set_xticklabels(CLS, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(7))
        ax.set_yticklabels(CLS, fontsize=8)
        ax.set_xlabel("预测", fontsize=9)
        ax.set_ylabel("真值", fontsize=9)
        ax.set_title(lb, fontsize=10)
        for i in range(7):
            for j in range(7):
                v = m[i, j]
                if v >= 0.01:
                    ax.text(j, i, f"{v*100:.0f}", ha="center", va="center", fontsize=7,
                            color="white" if v > 0.55 else "black")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="行归一化比例")
    # [图内不设标题/脚注，全部说明移至图题]
    p = f"{FIG}/fig5_confusion.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("  ->", p)


def fig6():
    adj = pd.read_csv(f"{ROOT}/data/stats/class_adjacency.csv")
    adj = adj[adj["split"].str.startswith("Train/")]
    m = np.zeros((7, 7))
    for _, r in adj.iterrows():
        i, j = int(r["cls_i"]) - 1, int(r["cls_j"]) - 1
        m[i, j] += r["n"]
        m[j, i] += r["n"]
    m = m / m.sum()
    m = np.where(np.eye(7, dtype=bool), np.nan, m)   # 类内不算邻接，置灰

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#e2e8f0")
    im = ax.imshow(m * 100, cmap=cmap)
    ax.set_xticks(range(7)); ax.set_xticklabels(CLS, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(7)); ax.set_yticklabels(CLS, fontsize=8)
    for i in range(7):
        for j in range(7):
            v = m[i, j] * 100
            if not np.isnan(v) and v >= 0.4:
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 4 else "black")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02, label="共边像素占全部邻接的 %")
    # [图内不设标题/脚注，全部说明移至图题]
    fig.tight_layout()
    p = f"{FIG}/fig6_adjacency.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("  ->", p)


def fig7():
    """训练动态。loss 只画纯 CE 的四个（V1/E10/E11/E12），因为 V2/V5 用的是 CE+Dice，
    量纲不同；dev mIoU 是同一监控口径（center-crop 512），四条都可以比。"""
    import csv
    pure = [("V1_ce", "V1（3.48M 从零）", "#2b6cb0"),
            ("E10_unet", "E10 UNet-R34（24.57M）", "#6b46c1"),
            ("E11_deeplab", "E11 DeepLabV3+（39.64M）", "#c05621"),
            ("E12_segformer", "E12 SegFormer-B0（3.45M）", "#2f855a")]
    mixed = [("V2", "V2（CE+Dice）", "#d69e2e"), ("V5", "V5（ASPP-Lite）", "#e53e3e")]

    def load(run):
        rows = list(csv.DictReader(open(f"{ROOT}/experiments/{run}/train_log.csv")))
        return ([int(r["epoch"]) for r in rows],
                [float(r["train_loss"]) for r in rows],
                [float(r["dev_mIoU"]) * 100 for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4))
    styles = ["-", "--", "-.", ":"]
    ax = axes[0]
    for i, (run, lb, c) in enumerate(pure):
        e, l, _ = load(run)
        ax.plot(e, l, color=c, lw=1.8, ls=styles[i], label=lb)
    ax.set_xlabel("epoch"); ax.set_ylabel("训练损失（纯 CE）")
    ax.set_title("(a) 纯 CE 模型的训练损失：四条轨迹几乎重合", fontsize=10)
    ax.grid(alpha=0.3, ls=":")
    ax.legend(fontsize=7.5, loc="upper right")
    ax.annotate("V1 / E10 / E12 三条重叠\n→ 损失值无法区分架构高低", xy=(20, 0.82),
                xytext=(11, 1.12), fontsize=7.5, color="#4a5568",
                arrowprops=dict(arrowstyle="->", color="#a0aec0", lw=0.8))

    ax = axes[1]
    for i, (run, lb, c) in enumerate(pure + mixed):
        e, _, d = load(run)
        ax.plot(e, d, color=c, lw=1.7, ls=styles[i % 4], label=lb)
    ax.set_xlabel("epoch"); ax.set_ylabel("dev mIoU（center-crop 512 监控）")
    ax.set_title("(b) dev 监控曲线：同一口径，六个模型可比", fontsize=10)
    ax.grid(alpha=0.3, ls=":"); ax.set_ylim(18, 78)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=3, frameon=False)
    ax.text(0.02, 0.95, "V3 未画：其监控口径是逐图校正后的 mIoU（35.02），不可比",
            transform=ax.transAxes, fontsize=7, color="#718096")
    fig.suptitle("F7 训练动态", fontsize=11)
    fig.tight_layout()
    p = f"{FIG}/fig7_training_dynamics.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("  ->", p)


def fig13():
    d = json.load(open(f"{ROOT}/experiments/gap_decompose.json"))
    labels, gaps, priors = [], [], []
    for tag, run in (("V1", "V1_ce"), ("V2", "V2")):
        r = d[run]
        labels.append(f"{tag}\n全量"); gaps.append(r["gap"]); priors.append(r["gap_from_prior"])
        for dom in ("urban", "rural"):
            bd = r["by_domain"][dom]
            labels.append(f"{tag}\n{dom}"); gaps.append(bd["gap"]); priors.append(bd["from_prior"])

    fig, ax = plt.subplots(figsize=(9.2, 4.3))
    x = np.arange(len(labels)); w = 0.36
    ax.bar(x - w / 2, gaps, w, label="实测落差（dev − Val）", color=C_ALT)
    ax.bar(x + w / 2, priors, w, label="其中可由类别先验漂移解释", color=C_MAIN)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("点数")
    # [图内不设标题/脚注，全部说明移至图题]
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.92)
    ax.grid(axis="y", alpha=0.3, ls=":")
    for xi, (g_, p_) in enumerate(zip(gaps, priors)):
        ax.text(xi - w / 2, g_ + (0.5 if g_ >= 0 else -0.5), f"{g_:+.1f}",
                ha="center", va="bottom" if g_ >= 0 else "top", fontsize=7.5)
        ax.text(xi + w / 2, p_ + (0.5 if p_ >= 0 else -0.5), f"{p_:+.1f}",
                ha="center", va="bottom" if p_ >= 0 else "top", fontsize=7.5, color=C_MAIN)
    lo = min(min(gaps), min(priors)); hi = max(max(gaps), max(priors))
    ax.set_ylim(lo - 2.5, hi + 2.5)
    fig.tight_layout()
    p = f"{FIG}/fig13_gap_decompose.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("  ->", p)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    print("生成 F4 / F5 / F6 / F7 / F13 ...")
    fig4(); fig5(); fig6(); fig7(); fig13()
    print("done")
