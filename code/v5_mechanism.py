"""V5 失败的机制分析：预测分布是否向主类集中、空间是否变"块状"。

V3 的失败给了三条可测机制（log 域爆炸、与 logits 共线、叠加校正）。V5 只有现象
（Val −1.62、dev +0.47、裸地退回去），没有机制。本脚本补三个便宜的指标：

  ① **预测类别分布**：各类被预测的比例。若 V5 把预测推向道路/农田（V2 的两项增益），
     这一列会直接显示出来。
  ② **预测的空间块状化**：3x3 邻域内标签一致的比例（越大越"块"），以及每幅图的
     连通域个数的均值（越少越"块"）。假设：保留空间布局 → 预测更顺从大片同质区，
     于是块状化上升、小目标（裸地斑块）被吞掉。
  ③ **两模型逐像素一致率**：V5 相对 V2 改判了多少像素，以及其中多少流向主类。

用法
----
  python code/v5_mechanism.py --runs V2 V5 --split val
"""
from paths import ROOT  # noqa: E402
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import load_model, predict_tile  # noqa: E402
from loveda_dataset import CLASSES, MEAN, STD, build_index  # noqa: E402
from imgio import imread  # noqa: E402

# repo root from code/paths.py
DATA = os.path.join(ROOT, "data/raw/LoveDA")
EXP = os.path.join(ROOT, "experiments")


def stats_for(run, split, model_name, ckpt, device, limit=0):
    model = load_model(model_name, ckpt, device)
    items = build_index(DATA, split)
    if limit:
        items = items[:limit]
    pred_share = np.zeros(8, np.float64)
    uniform_frac = 0.0
    n_comp = []
    preds = {}
    total = 0
    for i, (ip, mp, _) in enumerate(items, 1):
        img = imread(ip, cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32) / 255.0
        img = torch.from_numpy(((img - MEAN) / STD).transpose(2, 0, 1).copy())
        p = predict_tile(model, img, device).cpu().numpy().astype(np.uint8)
        preds[os.path.basename(ip)] = p
        pred_share += np.bincount(p.ravel(), minlength=8)
        total += p.size
        # 3x3 邻域是否同标签：max/min 池化相等
        mx = cv2.dilate(p, np.ones((3, 3), np.uint8))
        mn = cv2.erode(p, np.ones((3, 3), np.uint8))
        uniform_frac += float((mx == mn).mean())
        # 每幅图的连通域个数（8 邻接）。必须**按类别分别数**再相加：
        # connectedComponents 把所有非零像素当同一个前景，直接喂标签图会永远得到 1。
        per_img = 0
        for c in range(1, 8):
            mask = (p == c).astype(np.uint8)
            if mask.any():
                n_lab, _, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
                per_img += n_lab - 1
        n_comp.append(per_img)
        if i % 100 == 0:
            print(f"    {i}/{len(items)}", flush=True)
    return dict(pred_share=(pred_share / total).tolist(),
                uniform_frac=uniform_frac / len(items),
                mean_components=float(np.mean(n_comp)),
                preds=preds, n=len(items))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["V2", "V5"])
    ap.add_argument("--models", nargs="+", default=["v1", "v5"])
    ap.add_argument("--splits", nargs="+", default=["val"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {}
    for run, mname in zip(args.runs, args.models):
        for split in args.splits:
            print(f"\n[{run}/{split}]", flush=True)
            ckpt = os.path.join(EXP, run, "best.pth")
            s = stats_for(run, split, mname, ckpt, device, args.limit)
            out[f"{run}/{split}"] = {k: v for k, v in s.items() if k != "preds"}
            out[f"{run}/{split}"]["_cache"] = s["preds"]
            print(f"  均匀 3x3 邻域比例 {s['uniform_frac']*100:.2f}%  "
                  f"平均连通域 {s['mean_components']:.1f}/图")

    print("\n" + "=" * 72)
    print("① 预测类别分布（%）")
    keys = list(out.keys())
    print(f"{'class':<12}" + "".join(f"{k:>16}" for k in keys))
    for c in range(1, 8):
        row = f"{CLASSES[c-1]:<12}"
        for k in keys:
            row += f"{out[k]['pred_share'][c]*100:16.2f}"
        print(row)
    print(f"{'背景/主类比':<12}" + "".join(
        f"{out[k]['pred_share'][1]/max(sum(out[k]['pred_share'][2:]),1e-9):16.3f}" for k in keys))

    print("\n② 空间块状化")
    for k in keys:
        print(f"  {k:<12} 3x3 同标签 {out[k]['uniform_frac']*100:5.2f}%  "
              f"连通域 {out[k]['mean_components']:6.1f}/图")

    if len(keys) == 2:
        a, b = keys
        pa, pb = out[a]["_cache"], out[b]["_cache"]
        common = set(pa) & set(pb)
        if common:
            diff = 0
            flow = np.zeros((8, 8), np.int64)
            tot = 0
            for k in common:
                x, y = pa[k], pb[k]
                m = x != y
                np.add.at(flow, (x[m], y[m]), 1)
                diff += int(m.sum()); tot += x.size
            print(f"\n③ 改判像素：{a} → {b} 共 {diff/tot*100:.2f}%（{diff/1e6:.1f} M px）")
            print("   改判流向（行=V2 的类，列=V5 的类），只列占比 >1% 的：")
            f = flow / max(flow.sum(), 1)
            for i in range(1, 8):
                for j in range(1, 8):
                    if i != j and f[i, j] > 0.01:
                        print(f"     {CLASSES[i-1]:>11} → {CLASSES[j-1]:<11} {f[i,j]*100:5.1f}%")

    for k in out:
        out[k].pop("_cache", None)
    path = os.path.join(EXP, "v5_mechanism.json")
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=1)
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
