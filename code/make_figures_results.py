"""从已有产物生成论文图表（不需要 GPU 的那几张）。

只读 JSON，不碰模型，因此可以在训练占用 GPU 时安全运行。

当前生成
--------
F8   干预强度阶梯柱状图（V1 / +δ / E-P1 / V2 / V3 / V5）——论文核心图
F9   π 预测构成 vs 真实构成（253 个 dev tile × 7 类）
F10  τ 曲线（pred / global / oracle × dev/Val）

待 GPU 出数的（需要前瞻预测或混淆矩阵）：
F4 长尾曲线 / F5 混淆矩阵 / F6 边界分层 / F11 效率-精度散点 / F12 预测可视化 / F13 落差分解

用法
----
    python code/make_figures_results.py
"""
from paths import EXPERIMENTS, FIGURES  # noqa: E402
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP = str(EXPERIMENTS)
FIG = str(FIGURES)
CLASSES = ["background", "building", "road", "water", "barren", "forest", "agriculture"]
CN = dict(zip(CLASSES, ["背景", "建筑", "道路", "水体", "裸地", "林地", "农田"]))

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 200


def load(p):
    with open(os.path.join(EXP, p), encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
def fig_f8():
    """干预强度阶梯：Val mIoU，标注相对 V1 的差。"""
    v1 = load("V1_ce/eval_val.json")["mIoU"] * 100
    v2 = load("V2/eval_val.json")["mIoU"] * 100
    v3 = load("V3/pah_eval_val.json")["mode_none"]["mIoU"] * 100
    v5 = load("V5/eval_val.json")["mIoU"] * 100
    th = load("V1_ce/threshold.json")
    e_delta = th["val"]["with_free"]["mIoU"] * 100
    e_p1 = load("V1_ce/probe_balanced_last/probe_result.json")["val_mIoU"] * 100

    labels = ["V1\n基线", "+δ\n决策层", "E-P1\n读出层", "V2\n损失层",
              "V3\nPAH\n(vs V2 −0.60)", "V5\n上下文\n(vs V2 −1.62)"]
    vals = [v1, e_delta, e_p1, v2, v3, v5]
    colors = ["#9aa0a6", "#4c72b0", "#4c72b0", "#c0392b", "#dd8452", "#dd8452"]
    deltas = [v - v1 for v in vals]

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    bars = ax.bar(range(len(vals)), vals, color=colors, width=0.62)
    ax.axhline(v1, ls="--", lw=0.9, color="#555", zorder=0)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Val mIoU (%)")
    ax.set_ylim(32.5, 45.6)
    # [图内不设标题/脚注，全部说明移至图题]
    for b, v, d in zip(bars, vals, deltas):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.25, f"{v:.2f}", ha="center", fontsize=8.5)
        if abs(d) > 1e-9:
            ax.text(b.get_x() + b.get_width() / 2, v - 1.1, f"{d:+.2f}",
                    ha="center", fontsize=7.5, color="white", weight="bold")
    ax.grid(axis="y", alpha=0.25)
    # [图内不设标题/脚注，全部说明移至图题]
    fig.tight_layout()
    p = os.path.join(FIG, "fig8_intervention_ladder.png")
    fig.savefig(p)
    plt.close(fig)
    print("  F8 ->", p)
    return dict(v1=v1, edelta=e_delta, ep1=e_p1, v2=v2, v3=v3, v5=v5)


# --------------------------------------------------------------------------- #
def fig_f9():
    """π 预测构成 vs 真实构成（V3 的先验头质量）。"""
    d = load("V3/pah_eval_dev.json")
    S = d["pi_samples"]
    P = np.array([r["pred"] for r in S])
    G = np.array([r["gt"] for r in S])

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.1))
    ax = axes[0]
    cmap = plt.get_cmap("tab10")
    for c in range(7):
        ax.scatter(G[:, c], P[:, c], s=7, alpha=0.45, color=cmap(c), label=CN[CLASSES[c]],
                   edgecolors="none")
    lim = max(G.max(), P.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=0.9)
    ax.set_xlabel("真实构成 π_gt")
    ax.set_ylabel("预测构成 π")
    ax.set_title(f"π vs 真值（dev {len(S)} 幅，MAE={d['prior_mae']:.4f}）", fontsize=10)
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.grid(alpha=0.2)

    ax = axes[1]
    mae = np.abs(P - G).mean(0)
    bars = ax.barh(range(7), mae, color=[cmap(c) for c in range(7)])
    ax.set_yticks(range(7))
    ax.set_yticklabels([CN[c] for c in CLASSES], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("MAE（π 与真值构成）")
    ax.set_title("逐类误差：稀有类误差更大", fontsize=10)
    for b, v in zip(bars, mae):
        ax.text(v + 0.002, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=7.5)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    p = os.path.join(FIG, "fig9_prior_scatter.png")
    fig.savefig(p)
    plt.close(fig)
    print("  F9 ->", p)


# --------------------------------------------------------------------------- #
def fig_f10():
    """τ 曲线：三种先验 × 两个域。"""
    dev = load("V3/pah_eval_dev.json")
    val = load("V3/pah_eval_val.json")
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9), sharey=False)
    styles = {"pred": ("#c0392b", "o", "逐图 π（V3）"),
              "global": ("#4c72b0", "s", "全局 π_train"),
              "oracle": ("#6d6d6d", "^", "oracle 真值构成")}
    for ax, d, name in ((axes[0], dev, "dev（253 幅）"), (axes[1], val, "Val（1669 幅）")):
        taus = d["tau_list"]
        for mode, (c, m, lab) in styles.items():
            ys = [d["sweep"][mode][f"{t:g}"]["mIoU"] * 100 for t in taus]
            ax.plot(taus, ys, marker=m, ms=3.6, lw=1.4, color=c, label=lab)
        base = d["mode_none"]["mIoU"] * 100
        ax.axhline(base, ls="--", lw=1.0, color="#111")
        ax.annotate(f"不校正 {base:.2f}", xy=(taus[0], base),
                    xytext=(0.05, 0.06), textcoords="axes fraction",
                    fontsize=7.5, color="#111",
                    bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi * 1.04)
        ax.set_xlabel("τ")
        ax.set_title(name, fontsize=10)
        ax.grid(alpha=0.22)
        ax.legend(fontsize=7.5, loc="upper right", framealpha=0.92)
    axes[0].set_ylabel("mIoU (%)")
    # [图内不设标题/脚注，全部说明移至图题]
    fig.tight_layout()
    p = os.path.join(FIG, "fig10_tau_curves.png")
    fig.savefig(p)
    plt.close(fig)
    print("  F10 ->", p)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    print("生成不依赖 GPU 的图表：")
    info = fig_f8()
    fig_f9()
    fig_f10()
    print("\nF8 数据：", {k: round(v, 2) for k, v in info.items()})
