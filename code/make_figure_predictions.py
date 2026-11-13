"""F12 预测可视化：同一张图上五个模型的输出对比。

选图标准（两条同时满足）：
  ① GT 含全部 7 类 —— 否则 mIoU 会退化成单类 IoU。曾选 Urban#3600（V1 14.7 → V2 44.4），
     后来发现它的 GT 只有"水体"一类，E11 直接拿到 100.0，这类图不能当示例。
  ② V2 相对 V1 的逐图 mIoU 提升显著（Urban#3520 +10.7、Rural#2577 +7.6）。
     曾经选过 Rural#2571（+13.2），但它亮度只有 47，原图几乎全黑看不清地物，换掉。
这是刻意的"典型成功案例"，**不代表平均水平**——同批 60 张 rural 的 V2−V1 平均只有 +0.4。
图注必须写明这一点，否则会被读成 cherry-pick。

每个面板标注该图的 per-tile mIoU（只统计 gt≠0 的像素，与全图评测口径一致）。
"""
from paths import ROOT  # noqa: E402
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import load_model, predict_tile  # noqa: E402
from loveda_dataset import MEAN, STD  # noqa: E402
from imgio import imread  # noqa: E402

# repo root from code/paths.py
DATA = f"{ROOT}/data/raw/LoveDA"
FIG = f"{ROOT}/report/figures"
PALETTE = {1: (0, 0, 0), 2: (255, 0, 0), 3: (255, 255, 0), 4: (0, 0, 255),
           5: (159, 129, 183), 6: (0, 255, 0), 7: (255, 195, 128)}
CLS = {1: "背景", 2: "建筑", 3: "道路", 4: "水体", 5: "裸地", 6: "林地", 7: "农田"}

TILES = [("Val/Urban", "3520"), ("Val/Rural", "2577")]
MODELS = [("V1_ce", "v1", "V1"), ("V2", "v1", "V2"), ("V3", "v3", "V3"),
          ("V5", "v5", "V5"), ("E11_deeplab", "deeplabv3p_r50", "E11 DeepLab")]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def colourise(mask):
    """0 = no-data（图像黑边），必须与 1 = 背景区分开，否则读者会把无效区当成模型漏判。"""
    out = np.zeros((*mask.shape, 3), np.uint8)
    out[mask == 0] = (150, 150, 150)
    for k, c in PALETTE.items():
        out[mask == k] = c
    return out


def tile_iou(pred, gt):
    """只统计 gt≠0 的像素，与全图评测的 7 类口径一致。"""
    ious = []
    for c in range(1, 8):
        tp = np.sum((pred == c) & (gt == c))
        fp = np.sum((pred == c) & (gt != c) & (gt != 0))
        fn = np.sum((pred != c) & (gt == c))
        den = tp + fp + fn
        ious.append(tp / den if den else np.nan)
    return float(np.nanmean(ious)) * 100


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ncol = 2 + len(MODELS)
    fig, axes = plt.subplots(2, ncol, figsize=(2.55 * ncol, 6.9))

    for r, (sub, tid) in enumerate(TILES):
        ip = f"{DATA}/{sub}/images_png/{tid}.png"
        mp = f"{DATA}/{sub}/masks_png/{tid}.png"
        img = imread(ip, cv2.IMREAD_COLOR)[:, :, ::-1]
        gt = imread(mp, cv2.IMREAD_GRAYSCALE)
        x = torch.from_numpy(
            (((img.astype(np.float32) / 255.0) - MEAN) / STD).transpose(2, 0, 1).copy())

        axes[r, 0].imshow(img)
        axes[r, 0].set_title(f"{sub.split('/')[1]} #{tid}\n原图", fontsize=8.5)
        axes[r, 1].imshow(colourise(gt))
        nodata = float((gt == 0).mean()) * 100
        axes[r, 1].set_title("GT", fontsize=8.5)
        for c in (0, 1):
            axes[r, c].axis("off")

        for j, (run, mname, lb) in enumerate(MODELS, start=2):
            ckpt = f"{ROOT}/experiments/{run}/best.pth"
            if not os.path.exists(ckpt):
                axes[r, j].text(0.5, 0.5, "缺 checkpoint", ha="center", va="center",
                                fontsize=8, transform=axes[r, j].transAxes)
                axes[r, j].axis("off")
                continue
            model = load_model(mname, ckpt, device)
            with torch.no_grad():
                pred = predict_tile(model, x, device).cpu().numpy()
            del model
            torch.cuda.empty_cache()
            axes[r, j].imshow(colourise(pred))
            axes[r, j].set_title(f"{lb}\ntile mIoU {tile_iou(pred, gt):.1f}", fontsize=8.5)
            axes[r, j].axis("off")
            print(f"  {sub}#{tid} {lb:<12} {tile_iou(pred, gt):.2f}")

    handles = [plt.Rectangle((0, 0), 1, 1, color=np.array(PALETTE[k]) / 255)
               for k in range(1, 8)] + [plt.Rectangle((0, 0), 1, 1, color=(150/255,)*3)]
    fig.legend(handles, [CLS[k] for k in range(1, 8)] + ["无数据(0)"], ncol=8,
               loc="upper center", bbox_to_anchor=(0.5, 0.035), fontsize=8.5, frameon=False)
    # [图内不设标题/脚注，全部说明移至图题]
    fig.tight_layout(rect=[0, 0.055, 1, 1], h_pad=1.4)
    fig.subplots_adjust(hspace=0.10)
    p = f"{FIG}/fig12_predictions.png"
    fig.savefig(p, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print("  ->", p)


if __name__ == "__main__":
    main()
