"""Figures for the dataset-analysis section (stage 1 deliverable)."""
from paths import DATA_RAW, DATA_STATS, FIGURES  # noqa: E402
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imgio import imread  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CLS_CN = {"background": "背景", "building": "建筑", "road": "道路", "water": "水体",
          "barren": "裸地", "forest": "林地", "agriculture": "农田"}
CLS_ID = {"background": 1, "building": 2, "road": 3, "water": 4, "barren": 5, "forest": 6, "agriculture": 7}
PALETTE = {1: (0, 0, 0), 2: (255, 0, 0), 3: (255, 255, 0), 4: (0, 0, 255),
           5: (159, 129, 183), 6: (0, 255, 0), 7: (255, 195, 128)}

STATS = str(DATA_STATS)
OUT = str(FIGURES)
os.makedirs(OUT, exist_ok=True)


def fig_class_ratio():
    cp = pd.read_csv(f"{STATS}/class_pixels.csv")
    piv = cp.pivot(index="split", columns="cls", values="ratio").loc[
        ["Train/Urban", "Train/Rural", "Val/Urban", "Val/Rural"]]
    order = ["background", "agriculture", "forest", "building", "water", "road", "barren"]
    piv = piv[order] * 100

    fig, ax = plt.subplots(figsize=(9, 3.6))
    left = np.zeros(len(piv))
    colors = ["#4d4d4d", "#f2b134", "#2e8b57", "#d64545", "#3a7bd5", "#9b59b6", "#b08d57"]
    for c, col in zip(order, colors):
        ax.barh(range(len(piv)), piv[c], left=left, label=CLS_CN[c], color=col, edgecolor="white", linewidth=.6)
        for i, v in enumerate(piv[c]):
            if v > 6:
                ax.text(left[i] + v / 2, i, f"{v:.1f}", ha="center", va="center",
                        color="white", fontsize=8)
        left += piv[c].values
    ax.set_yticks(range(len(piv)))
    ax.set_yticklabels([s.replace("Train", "训练").replace("Val", "验证")
                        .replace("/Urban", "-城市").replace("/Rural", "-乡村") for s in piv.index])
    ax.set_xlabel("像素占比 (%)")
    # [图内不设标题/脚注，全部说明移至图题]
    ax.legend(ncol=7, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
    ax.set_xlim(0, 100)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_class_ratio.png", dpi=200)
    plt.close(fig)


def fig_object_scale():
    g = pd.read_csv(f"{STATS}/class_geometry.csv")
    g["cls_name"] = g["cls"].map({v: k for k, v in CLS_ID.items()})
    order = ["building", "road", "forest", "water", "barren", "background", "agriculture"]
    data = [np.log10(g.loc[g.cls_name == c, "area_med"].clip(lower=1)) for c in order]
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    bp = ax.boxplot(data, labels=[CLS_CN[c] for c in order], patch_artist=True, showfliers=False)
    for patch, col in zip(bp["boxes"], ["#d64545", "#9b59b6", "#2e8b57", "#3a7bd5",
                                        "#b08d57", "#4d4d4d", "#f2b134"]):
        patch.set_facecolor(col)
        patch.set_alpha(.75)
    ax.set_ylabel("单个连通域面积 log10(像素)")
    # [图内不设标题/脚注，全部说明移至图题]
    ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_object_scale.png", dpi=200)
    plt.close(fig)


def _colourise(mask):
    out = np.zeros((*mask.shape, 3), np.uint8)
    for k, c in PALETTE.items():
        out[mask == k] = c
    return out


def fig_samples(root=str(DATA_RAW)):
    picks = [("Val/Urban", "3514"), ("Val/Urban", "3600"), ("Val/Rural", "2522"), ("Val/Rural", "2600")]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.4))
    for j, (sub, tid) in enumerate(picks):
        ip = f"{root}/{sub}/images_png/{tid}.png"
        mp = f"{root}/{sub}/masks_png/{tid}.png"
        if not os.path.exists(ip):
            continue
        img = cv2.cvtColor(imread(ip), cv2.COLOR_BGR2RGB)
        msk = imread(mp, cv2.IMREAD_GRAYSCALE)
        axes[0, j].imshow(img)
        axes[0, j].set_title(f"{sub.split('/')[1]}  #{tid}", fontsize=9)
        axes[1, j].imshow(_colourise(msk))
        for r in (0, 1):
            axes[r, j].axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=np.array(c) / 255) for c in PALETTE.values()]
    fig.legend(handles, [CLS_CN[k] for k in
                         ["background", "building", "road", "water", "barren", "forest", "agriculture"]],
               ncol=7, loc="lower center", fontsize=9, frameon=False)
    # [图内不设标题/脚注，全部说明移至图题]
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(f"{OUT}/fig3_samples.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    fig_class_ratio()
    fig_object_scale()
    fig_samples()
    print("figures ->", OUT, os.listdir(OUT))
