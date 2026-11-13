"""F14 V5 相对 V2 的预测类别份额变化（两域并排）。

图要传达的只有一件事：**裸地在两个域上都被抽掉约 1.6 个百分点**，而农田被抬高。
两域方向一致 → 说明 V5 的作用是单一的（压低裸地倾向），差别只在于落点。
"""
from paths import EXPERIMENTS, FIGURES  # noqa: E402
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

EXP = str(EXPERIMENTS)
FIG = str(FIGURES)
CLS = ["background", "building", "road", "water", "barren", "forest", "agriculture"]
CN = ["背景", "建筑", "道路", "水体", "裸地", "林地", "农田"]


def share(run, split):
    d = json.load(open(f"{EXP}/{run}/eval_{split}.json"))
    return np.array([d["per_class"][c]["IoU"] for c in CLS])  # placeholder, not used


def main():
    # 用机制脚本的输出（预测份额），不是 eval 里的 IoU
    out = {}
    for sp in ("dev", "val"):
        d = json.load(open(f"{EXP}/v5_mechanism_{sp}.json"))
        out[sp] = {k: np.array(v["pred_share"][1:8]) * 100 for k, v in d.items()}

    dev_delta = out["dev"]["V5/dev"] - out["dev"]["V2/dev"]
    val_delta = out["val"]["V5/val"] - out["val"]["V2/val"]

    fig, ax = plt.subplots(figsize=(8.4, 4.2), dpi=200)
    x = np.arange(7)
    w = 0.38
    b1 = ax.bar(x - w / 2, dev_delta, w, label="dev（253 幅）", color="#2b6cb0", alpha=0.9)
    b2 = ax.bar(x + w / 2, val_delta, w, label="Val（1669 幅）", color="#c05621", alpha=0.9)
    ax.axhline(0, color="k", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(CN, fontsize=9)
    ax.set_ylabel("预测份额变化（V5 − V2，百分点）")
    # [图内不设标题/脚注，全部说明移至图题]
    ax.grid(axis="y", alpha=0.25, ls=":")
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.92)

    for bars in (b1, b2):
        for b in bars:
            v = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, v + (0.12 if v >= 0 else -0.12),
                    f"{v:+.1f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=7)

    # 圈出裸地
    ax.axvspan(3.5, 4.5, color="#fde68a", alpha=0.35, zorder=0)
    ax.annotate("两域都降 1.6–1.7 pp", xy=(4.42, -1.15), xytext=(4.75, -3.2),
                fontsize=8, color="#7b341e",
                arrowprops=dict(arrowstyle="->", color="#c05621", lw=1.0))

    lo = min(dev_delta.min(), val_delta.min()); hi = max(dev_delta.max(), val_delta.max())
    ax.set_ylim(lo - 0.55, hi + 0.9)
    fig.tight_layout()
    p = f"{FIG}/fig14_v5_share_shift.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("  ->", p)
    for i, c in enumerate(CN):
        print(f"  {c:<4} dev {dev_delta[i]:+6.2f}  val {val_delta[i]:+6.2f}")


if __name__ == "__main__":
    main()
