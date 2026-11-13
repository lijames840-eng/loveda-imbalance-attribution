#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Step 0c：错判区域到底长什么样。

(1) 颜色统计：错判的农田像素 vs 判对的农田像素，RGB / 亮度 / 水体指数
(2) 6 幅错判最重的图做高对比度三联（增强原图 / 真值 / 预测）
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
from imgio import imread, imwrite  # noqa: E402

ROOT = str(DATA_RAW)
RUN = str(EXPERIMENTS / 'V2')
OUT = str(RESULTS / 'step0_probe')
os.makedirs(OUT, exist_ok=True)
BG, WATER, AGRI = 1, 4, 7

items = build_index(ROOT, 'Val')
z = np.load(os.path.join(RUN, 'prob_val_ds4.npz'))
prob, gt_ds = z['prob'], z['gt']

# ---------- (1) 颜色统计（ds4 定位 + 原图降采样取色） ----------
acc = {k: np.zeros(3, np.float64) for k in ('bad_bg', 'bad_water', 'ok')}
cnt = {k: 0 for k in acc}
lum = {k: 0.0 for k in acc}
ndwi = {k: 0.0 for k in acc}     # (G-R)/(G+R)，水体偏正

for i, (ip, mp, dom) in enumerate(items):
    if dom != 'rural':
        continue
    g = gt_ds[i]
    a = (g == AGRI)
    if a.sum() < 200:
        continue
    p = prob[i].argmax(0) + 1
    img = imread(ip, cv2.IMREAD_COLOR)          # BGR
    img = cv2.resize(img, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA).astype(np.float32)
    B, G, R = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    groups = {
        'ok': a & (p == AGRI),
        'bad_bg': a & (p == BG),
        'bad_water': a & (p == WATER),
    }
    for k, m in groups.items():
        if m.sum() == 0:
            continue
        acc[k] += np.array([R[m].mean(), G[m].mean(), B[m].mean()]) * m.sum()
        lum[k] += (0.299 * R[m] + 0.587 * G[m] + 0.114 * B[m]).sum()
        ndwi[k] += ((G[m] - R[m]) / (G[m] + R[m] + 1e-6)).sum()
        cnt[k] += int(m.sum())

col = {}
for k in acc:
    if cnt[k] == 0:
        continue
    rgb = acc[k] / cnt[k]
    col[k] = dict(n=cnt[k], R=round(float(rgb[0]), 1), G=round(float(rgb[1]), 1),
                  B=round(float(rgb[2]), 1),
                  lum=round(float(lum[k] / cnt[k]), 1),
                  ndwi=round(float(ndwi[k] / cnt[k]), 4))

print('=== 颜色统计（ds4 降采样后取色）===')
print(f"{'组':>10} {'像素数':>12} {'R':>6} {'G':>6} {'B':>6} {'亮度':>7} {'NDWI':>8}")
for k, v in col.items():
    print(f"{k:>10} {v['n']:>12,} {v['R']:>6} {v['G']:>6} {v['B']:>6} {v['lum']:>7} {v['ndwi']:>8}")

# ---------- (2) 选错判最重的 6 幅 ----------
rank = []
for i, (ip, mp, dom) in enumerate(items):
    if dom != 'rural':
        continue
    g = gt_ds[i]; a = (g == AGRI)
    if a.sum() < 1000:
        continue
    p = prob[i].argmax(0) + 1
    bad = int((a & ((p == BG) | (p == WATER))).sum())
    rank.append((bad, i, os.path.basename(ip)[:-4]))
rank.sort(reverse=True)
pick = rank[:6]
print('\ntop mismatch tiles:', [(t, f'{b:,}') for b, _, t in pick])

DEV = torch.device('cuda')
model = load_model('v1', os.path.join(RUN, 'best.pth'), DEV)
PAL = {1: (170, 170, 170), 2: (200, 60, 60), 3: (60, 60, 200), 4: (40, 140, 230),
       5: (140, 190, 200), 6: (40, 110, 40), 7: (80, 200, 80)}
CELL = 420

def enhance(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

def paint(lbl):
    out = np.zeros((*lbl.shape, 3), np.uint8)
    for k, c in PAL.items():
        out[lbl == k] = c
    return out

rows = []
for bad_n, i, tile in pick:
    ip, mp, _ = items[i]
    bgr = imread(ip, cv2.IMREAD_COLOR)
    gt = imread(mp, cv2.IMREAD_GRAYSCALE)
    pred = predict_tile(model, prep(ip), DEV).cpu().numpy() + 1
    e = enhance(bgr)
    # 错判处描红边（细线，避免遮挡地物）
    m = ((gt == AGRI) & ((pred == BG) | (pred == WATER))).astype(np.uint8)
    cont, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(e, cont, -1, (0, 0, 255), 2)
    trip = np.hstack([cv2.resize(e, (CELL, CELL)),
                      cv2.resize(paint(gt), (CELL, CELL), interpolation=cv2.INTER_NEAREST),
                      cv2.resize(paint(pred), (CELL, CELL), interpolation=cv2.INTER_NEAREST)])
    bar = np.full((34, CELL * 3, 3), 255, np.uint8)
    cv2.putText(bar, f'{tile}   agri->BG/WATER {bad_n:,}px   [orig | GT | pred]',
                (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    rows.append(np.vstack([bar, trip]))
    print(' drawn', tile, flush=True)

for s in range(0, len(rows), 3):
    imwrite(os.path.join(OUT, f'case_{s//3+1}.png'), np.vstack(rows[s:s + 3]))
    print('saved', f'case_{s//3+1}.png')

json.dump(dict(colors=col, top=[dict(tile=t, bad=b) for b, _, t in pick]),
          open(os.path.join(OUT, 'colors.json'), 'w'), indent=1)
