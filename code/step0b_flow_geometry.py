#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Step 0b：量化"农田被让成背景/水体"的空间形态。

对每一幅 Val 乡村图做全分辨率预测，统计：
  A. 错判像素到"农田/非农田"边界的距离分布（对照组：判对的农田像素）
  B. 错判像素所属农田连通域的面积分布（是否来自碎片/窄条）
  C. 错判像素在形态学腐蚀后的"农田核心区"内的占比

若错判高度集中在边界带与碎片上 -> 是细结构分辨率问题（标注按地块整体，模型看见田埂/水渠）
若错判广泛分布在大块内部           -> 是真错，需要改表征
"""
from paths import DATA_RAW, EXPERIMENTS, RESULTS  # noqa: E402
import os, sys, json
import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loveda_dataset import build_index
from evaluate import predict_tile, load_model
from fit_threshold import prep
from imgio import imread  # noqa: E402

ROOT = str(DATA_RAW)
RUN = str(EXPERIMENTS / 'V2')
OUT = str(RESULTS / 'step0_probe')
os.makedirs(OUT, exist_ok=True)

BG, WATER, AGRI = 1, 4, 7
BINS = [0, 3, 6, 10, 20, 40, 80, 10 ** 6]
LBL = ['<=3', '3-6', '6-10', '10-20', '20-40', '40-80', '>80']

items = build_index(ROOT, 'Val')
rural = [it for it in items if it[2] == 'rural']
print('rural tiles', len(rural))

DEV = torch.device('cuda')
model = load_model('v1', os.path.join(RUN, 'best.pth'), DEV)

hist_bad = np.zeros(len(BINS) - 1, np.int64)      # 错判像素，按到边界距离
hist_ok = np.zeros(len(BINS) - 1, np.int64)       # 判对像素，对照
area_bins = [0, 500, 2000, 8000, 30000, 10 ** 9]
area_lbl = ['<=500', '500-2k', '2k-8k', '8k-30k', '>30k']
hist_area_bad = np.zeros(len(area_bins) - 1, np.int64)
hist_area_all = np.zeros(len(area_bins) - 1, np.int64)
erode_stat = dict(bad_in_core=0, bad_total=0, ok_in_core=0, ok_total=0)

K3 = np.ones((3, 3), np.uint8)
K5 = np.ones((5, 5), np.uint8)

for n, (ip, mp, _) in enumerate(rural, 1):
    gt = imread(mp, cv2.IMREAD_GRAYSCALE)
    pred = predict_tile(model, prep(ip), DEV).cpu().numpy() + 1
    agri = (gt == AGRI).astype(np.uint8)
    if agri.sum() == 0:
        continue
    bad = (agri == 1) & ((pred == BG) | (pred == WATER))
    ok = (agri == 1) & (~bad)

    # 到农田边界（农田与非农田的分界）的距离
    edge = agri - cv2.erode(agri, K3)
    dist = cv2.distanceTransform(1 - edge, cv2.DIST_L2, 3)

    for name, mask, hist in (('bad', bad, hist_bad), ('ok', ok, hist_ok)):
        if mask.sum() == 0:
            continue
        d = dist[mask]
        h, _ = np.histogram(d, bins=BINS)
        hist += h.astype(np.int64)

    # 所属农田连通域面积
    nlab, lab, stats, _ = cv2.connectedComponentsWithStats(agri, 8)
    if nlab > 1:
        areas = stats[:, 4]
        pix_area = areas[lab]
        for mask, hist in ((bad, hist_area_bad), (agri.astype(bool), hist_area_all)):
            if mask.sum() == 0:
                continue
            a = pix_area[mask]
            h, _ = np.histogram(a, bins=area_bins)
            hist += h.astype(np.int64)

    # 腐蚀 5×5 后的农田核心区
    core = cv2.erode(agri, K5) > 0
    erode_stat['bad_in_core'] += int((bad & core).sum())
    erode_stat['bad_total'] += int(bad.sum())
    erode_stat['ok_in_core'] += int((ok & core).sum())
    erode_stat['ok_total'] += int(ok.sum())

    if n % 100 == 0:
        print(f'  {n}/{len(rural)}', flush=True)

def pct(h):
    t = h.sum()
    return [round(float(x) / max(t, 1), 4) for x in h]

res = dict(
    n_rural=len(rural),
    bad_px=int(hist_bad.sum()), ok_px=int(hist_ok.sum()),
    dist_bins=LBL,
    bad_dist_pct=pct(hist_bad), ok_dist_pct=pct(hist_ok),
    bad_dist_cnt=hist_bad.tolist(), ok_dist_cnt=hist_ok.tolist(),
    area_bins=area_lbl,
    bad_area_pct=pct(hist_area_bad), all_area_pct=pct(hist_area_all),
    core=erode_stat,
    bad_in_core_pct=erode_stat['bad_in_core'] / max(erode_stat['bad_total'], 1),
    ok_in_core_pct=erode_stat['ok_in_core'] / max(erode_stat['ok_total'], 1),
)
json.dump(res, open(os.path.join(OUT, 'flow_geometry.json'), 'w'), indent=1)

print('\n=== 到农田边界的距离分布（%） ===')
print(f"{'bin(px)':>10} {'错判':>8} {'判对':>8}")
for i, l in enumerate(LBL):
    print(f'{l:>10} {res["bad_dist_pct"][i]*100:7.1f}% {res["ok_dist_pct"][i]*100:7.1f}%')
print('\n=== 所属农田连通域面积分布（%） ===')
print(f"{'bin(px)':>10} {'错判':>8} {'全部农田':>9}")
for i, l in enumerate(area_lbl):
    print(f'{l:>10} {res["bad_area_pct"][i]*100:7.1f}% {res["all_area_pct"][i]*100:8.1f}%')
print(f'\n腐蚀 5x5 核心区内占比： 错判 {res["bad_in_core_pct"]*100:.1f}%   判对 {res["ok_in_core_pct"]*100:.1f}%')
print(f'错判像素总数 {res["bad_px"]:,}  判对 {res["ok_px"]:,}')
