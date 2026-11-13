"""F0 网络结构示意图（论文方法节）。

结构直接照 models/v1.py 的真实定义画，不按文档转述：
  stem  : conv3x3/s2 + DS-ResBlock          → 64  @ /2
  s1..s4: 2/3/3/4 个 DS-ResBlock             → 128@/4, 256@/8, 320@/16, 448@/32
  gc    : GlobalContext(448)                 ← V5 把它换成 ASPP-Lite
  FPN   : lat4→fuse3→proj3→fuse2→proj2→fuse1→head，自顶向下融合到 /4
  head  : ConvBNR(64,64) + 1×1 → 8 类，再双线性 ×4 回原分辨率

上下两排对齐，是为了让三条跳连走直线——图上一眼能看出"哪一级送到哪一级"。
"""
from paths import FIGURES  # noqa: E402
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

FIG = str(FIGURES)

C_ENC = "#dbeafe"   # 编码器
C_DEC = "#dcfce7"   # 解码器
C_GC = "#fef3c7"    # 上下文模块（V5 替换点）
C_V3 = "#fed7aa"    # V3 的 PAH
C_EDGE = "#4a5568"


def box(ax, x, y, w, h, text, fc, ls="-", lw=1.0, fs=6.6):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.004,rounding_size=0.012",
                                fc=fc, ec=C_EDGE, lw=lw, ls=ls, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=3, linespacing=1.35)


def arrow(ax, p0, p1, ls="-", color=C_EDGE, lw=1.0, rad=0.0, style="-|>"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=9,
                                 lw=lw, color=color, ls=ls, zorder=1,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=1.5, shrinkB=1.5))


def main():
    fig, ax = plt.subplots(figsize=(11.6, 3.0), dpi=200)
    ax.set_xlim(0, 1); ax.set_ylim(0.43, 1.01); ax.axis("off")

    Y_ENC, Y_DEC = 0.815, 0.505
    H, W = 0.088, 0.112
    X = dict(stem=0.165, s1=0.305, s2=0.442, s3=0.579, s4=0.716, gc=0.856)

    # ---------- 输入 ----------
    ax.text(0.045, Y_ENC, "输入\n512×512×3", ha="center", va="center", fontsize=7.2)
    arrow(ax, (0.083, Y_ENC), (X["stem"] - W / 2, Y_ENC))

    # ---------- 编码器 ----------
    box(ax, X["stem"], Y_ENC, W, H, "Stem\n3×3 / s2 + DS-Res\n64 @ /2", C_ENC)
    box(ax, X["s1"], Y_ENC, W, H, "Stage 1\n2 × DS-Res\n128 @ /4", C_ENC)
    box(ax, X["s2"], Y_ENC, W, H, "Stage 2\n3 × DS-Res\n256 @ /8", C_ENC)
    box(ax, X["s3"], Y_ENC, W, H, "Stage 3\n3 × DS-Res\n320 @ /16", C_ENC)
    box(ax, X["s4"], Y_ENC, W, H, "Stage 4\n4 × DS-Res\n448 @ /32", C_ENC)
    box(ax, X["gc"], Y_ENC, W, H, "上下文模块\nV1: SE-Gate (101 k)\nV5: ASPP-Lite (64 k)",
        C_GC, ls="--", lw=1.3)

    for a, b in [("stem", "s1"), ("s1", "s2"), ("s2", "s3"), ("s3", "s4"), ("s4", "gc")]:
        arrow(ax, (X[a] + W / 2, Y_ENC), (X[b] - W / 2, Y_ENC))

    # ---------- 解码器（左向） ----------
    Xd = dict(lat4=X["gc"], fuse3=X["s3"], fuse2=X["s2"], fuse1=X["s1"], head=X["stem"])
    box(ax, Xd["lat4"], Y_DEC, W, H, "lat4\n1×1 → 128", C_DEC)
    box(ax, Xd["fuse3"], Y_DEC, W, H, "fuse3\n上采样+相加\n128 → 96", C_DEC)
    box(ax, Xd["fuse2"], Y_DEC, W, H, "fuse2\n上采样+相加\n96 → 64", C_DEC)
    box(ax, Xd["fuse1"], Y_DEC, W, H, "fuse1\n上采样+相加\n64", C_DEC)
    box(ax, Xd["head"], Y_DEC, W, H, "Head\n3×3 + 1×1\n8 类 @ /4", C_DEC)

    # 自上而下的主通路：gc → lat4
    arrow(ax, (X["gc"], Y_ENC - H / 2), (Xd["lat4"], Y_DEC + H / 2))
    # 解码器内部（向左）
    for a, b in [("lat4", "fuse3"), ("fuse3", "fuse2"), ("fuse2", "fuse1"), ("fuse1", "head")]:
        arrow(ax, (Xd[a] - W / 2, Y_DEC), (Xd[b] + W / 2, Y_DEC))

    # ---------- 三条跳连（竖直，直线） ----------
    skips = [("s1", "fuse1", "/4"), ("s2", "fuse2", "/8"), ("s3", "fuse3", "/16")]
    for enc, dec, tag in skips:
        x = X[enc]
        arrow(ax, (x, Y_ENC - H / 2), (x, Y_DEC + H / 2), ls="--", color="#2b6cb0", lw=1.1)
        ax.text(x + 0.009, (Y_ENC + Y_DEC) / 2, tag, fontsize=6.8,
                color="#2b6cb0", ha="left", va="center")

    # ---------- 输出 ----------
    arrow(ax, (Xd["head"] - W / 2, Y_DEC), (0.083, Y_DEC))
    ax.text(0.045, Y_DEC - 0.055, "双线性 ×4\n512×512×8", ha="center", va="center", fontsize=7.2)

    # ---------- V3 的 PAH 挂载点 ----------
    arrow(ax, (X["gc"], Y_ENC + H / 2), (X["gc"], 0.906), ls="--", color="#c05621", lw=1.1)
    box(ax, X["gc"] - 0.078, 0.948, 0.165, 0.098,
        "V3: PAH（仅此版本）\nGAP → 128 → 7 → softmax\n58.4 k，仅推理期校正",
        C_V3, ls="--", lw=1.3, fs=5.7)
    ax.text(0.60, 0.952, "π = 本图类别构成\n（辅助监督 β=0.1）", fontsize=6.2,
            color="#c05621", ha="right", va="center", linespacing=1.4)

    # ---------- 图例与说明 ----------
    # [图内不设标题/脚注，全部说明移至图题]

    # [图内不设标题/脚注，全部说明移至图题]
    fig.tight_layout()
    p = f"{FIG}/fig0_architecture.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("  ->", p)


if __name__ == "__main__":
    main()
