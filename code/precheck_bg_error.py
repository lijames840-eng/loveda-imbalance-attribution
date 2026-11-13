"""§2.7.1 前置验证：背景误差到底是"同质区内部误判"还是别的东西？

动机
----
H5 的机制叙述是"V1 的 GAP 门控把空间布局压平，导致大片同质区内部误判"。这句话
在此之前只是从"H1 已排除边界（边界只承载 4% 误差）"反推出来的，没有直接证据。
本脚本用 V2 的 dev 预测回答三个问题，并给出 V5 是否可以上马的放行判据：

  ① 真背景像素被判成了哪几类          → 误差流向（先验问题 vs 结构问题）
  ② 误差像素到最近类别边界的距离分布   → 是否真的远离边界
  ③ 误差像素所在同类连通域的面积分布   → 是否落在同质块内部

放行判据（写在 report/stage0_research_plan_C.md §2.7.1）：
  若 >= 60% 的背景误差像素同时满足 "到边界 > 10 px" 且 "所属同质块 > 500 px"，
  则"同质区内部误判"成立，V5（ASPP-Lite）对症；否则不得硬上上下文模块。

用法：
  python code/precheck_bg_error.py --run V2 --model v1 \
      --config code/configs/v2.yaml --split dev
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import load_model, predict_tile  # noqa: E402
from loveda_dataset import CLASSES, MEAN, STD, build_index  # noqa: E402
from imgio import imread  # noqa: E402

BG = 1                      # LoveDA 的 background 标签
DIST_THR = 10               # px，距边界阈值
AREA_THR = 500              # px，同质块面积阈值
N_BINS = 6                  # 距离直方图分箱


def boundary_mask(gt):
    """3x3 形态学梯度：与标签不同的邻域像素即为边界带。"""
    k = np.ones((3, 3), np.uint8)
    mx = cv2.dilate(gt, k)
    mn = cv2.erode(gt, k)
    return (mx != mn)


def dist_to_boundary(bnd):
    """到最近边界像素的欧氏距离（px）。cv2.distanceTransform 求到最近零像素的距离。"""
    src = np.where(bnd, 0, 255).astype(np.uint8)
    return cv2.distanceTransform(src, cv2.DIST_L2, 5)


def component_area(gt, num_classes=8):
    """逐像素所属同类连通域的面积（4 邻接，与统计口径一致用 8 邻接）。"""
    area = np.zeros(gt.shape, np.int32)
    k = np.ones((3, 3), np.uint8)
    for c in range(1, num_classes):
        m = (gt == c).astype(np.uint8)
        if not m.any():
            continue
        n, lab = cv2.connectedComponents(m, connectivity=8)
        if n <= 1:
            continue
        cnt = np.bincount(lab.ravel(), minlength=n).astype(np.int32)
        area[m.astype(bool)] = cnt[lab[m.astype(bool)]]
    return area


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="V2")
    ap.add_argument("--model", default="v1")
    ap.add_argument("--config", default="code/configs/v2.yaml")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp = cfg["paths"]["experiments"]
    ckpt = args.ckpt or os.path.join(exp, args.run, "best.pth")
    model = load_model(args.model, ckpt, device)

    items = build_index(cfg["paths"]["data"], args.split)
    if args.limit:
        items = items[:args.limit]
    print(f"[precheck] {len(items)} tiles, split={args.split}, ckpt={ckpt}", flush=True)

    flow = np.zeros(8, np.int64)          # 背景误差流向（预测标签）
    n_bg = 0
    n_bg_err = 0
    dist_bg, area_bg = [], []             # 背景误差像素的距离/面积（采样存）
    dist_all, area_all = [], []
    n_all_px = 0
    n_all_err = 0
    gate_hits = 0                         # dist>10 且 area>500
    boundary_band_hits = 0                # dist<=3（边界带）
    per_domain = {}

    for i, (ip, mp, domain) in enumerate(items, 1):
        img = imread(ip, cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32) / 255.0
        img = torch.from_numpy(((img - MEAN) / STD).transpose(2, 0, 1).copy())
        pred = predict_tile(model, img, device).cpu().numpy()
        gt = imread(mp, cv2.IMREAD_GRAYSCALE)
        if pred.shape != gt.shape:
            pred = cv2.resize(pred.astype(np.uint8), (gt.shape[1], gt.shape[0]),
                              interpolation=cv2.INTER_NEAREST).astype(np.int64)

        valid = gt != 0
        err = (pred != gt) & valid
        bg = gt == BG
        bg_err = err & bg

        bnd = boundary_mask(gt)
        dist = dist_to_boundary(bnd)
        area = component_area(gt)

        n_bg += int(bg.sum())
        n_bg_err += int(bg_err.sum())
        n_all_px += int(valid.sum())
        n_all_err += int(err.sum())

        if bg_err.any():
            np.add.at(flow, pred[bg_err], 1)
            d = dist[bg_err]
            a = area[bg_err]
            dist_bg.append(d)
            area_bg.append(a)
            gate_hits += int(((d > DIST_THR) & (a > AREA_THR)).sum())
            boundary_band_hits += int((d <= 3).sum())
        if err.any():
            dist_all.append(dist[err])
            area_all.append(area[err])

        d0 = per_domain.setdefault(domain, {"n_bg": 0, "n_bg_err": 0})
        d0["n_bg"] += int(bg.sum())
        d0["n_bg_err"] += int(bg_err.sum())

        if i % 50 == 0:
            print(f"  {i}/{len(items)}", flush=True)

    dist_bg = np.concatenate(dist_bg) if dist_bg else np.array([0.0])
    area_bg = np.concatenate(area_bg) if area_bg else np.array([0])
    dist_all = np.concatenate(dist_all) if dist_all else np.array([0.0])

    flow_share = (flow[1:] / max(n_bg_err, 1))
    gate_share = gate_hits / max(n_bg_err, 1)

    print("\n=== ① 背景误差流向（真背景 → 被判成什么） ===")
    order = np.argsort(-flow[1:])
    for c in order:
        print(f"  {CLASSES[c]:<12} {flow[c+1]:>9d}  ({flow_share[c]*100:5.1f}%)")
    print(f"  合计背景误差 {n_bg_err} px / 背景总数 {n_bg} px  (错误率 {n_bg_err/max(n_bg,1)*100:.1f}%)")

    print("\n=== ② 误差像素到边界的距离（px） ===")
    print(f"  背景误差: p10 {np.percentile(dist_bg,10):6.1f} | 中位 {np.median(dist_bg):6.1f} "
          f"| p90 {np.percentile(dist_bg,90):6.1f} | >10px 占 {(dist_bg>DIST_THR).mean()*100:5.1f}%")
    print(f"  全部误差: 中位 {np.median(dist_all):6.1f} | >10px 占 {(dist_all>DIST_THR).mean()*100:5.1f}%")
    print(f"  背景误差落在边界带(<=3px) 占 {boundary_band_hits/max(n_bg_err,1)*100:.1f}%")

    print("\n=== ③ 误差像素所属同类连通域面积（px） ===")
    print(f"  背景误差: 中位 {np.median(area_bg):7.0f} | >500px 占 {(area_bg>AREA_THR).mean()*100:5.1f}%"
          f" | >5000px 占 {(area_bg>5000).mean()*100:5.1f}%")

    print("\n=== 放行判据 ===")
    print(f"  dist>{DIST_THR} 且 area>{AREA_THR} 的背景误差占比 = {gate_share*100:.1f}%  "
          f"(门槛 60%)  →  {'放行：同质区内部误判成立' if gate_share >= 0.6 else '不放行'}")

    out = dict(run=args.run, split=args.split, n_tiles=len(items),
               n_bg=n_bg, n_bg_err=n_bg_err, bg_err_rate=n_bg_err / max(n_bg, 1),
               n_all_px=n_all_px, n_all_err=n_all_err,
               flow={CLASSES[c]: int(flow[c + 1]) for c in range(7)},
               flow_share={CLASSES[c]: float(flow_share[c]) for c in range(7)},
               dist_bg=dict(p10=float(np.percentile(dist_bg, 10)),
                            p50=float(np.percentile(dist_bg, 50)),
                            p90=float(np.percentile(dist_bg, 90)),
                            share_gt_thr=float((dist_bg > DIST_THR).mean())),
               dist_all=dict(p50=float(np.percentile(dist_all, 50)),
                             share_gt_thr=float((dist_all > DIST_THR).mean())),
               area_bg=dict(p10=float(np.percentile(area_bg, 10)),
                            p50=float(np.median(area_bg)),
                            p90=float(np.percentile(area_bg, 90)),
                            share_gt_500=float((area_bg > AREA_THR).mean())),
               boundary_band_share=float(boundary_band_hits / max(n_bg_err, 1)),
               gate_share=float(gate_share), gate_threshold=0.6,
               gate_pass=bool(gate_share >= 0.6),
               per_domain={k: v for k, v in per_domain.items()})
    path = os.path.join(exp, args.run, "precheck_bg_error.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
