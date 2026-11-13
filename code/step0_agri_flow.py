#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Step 0：核查"农田 -> 背景/水体"的错判是否真的是模型错误。

只读已有产物（V2 的 ds4 概率缓存 + 全分辨率推理），不训练、不改任何 run。
输出：candidates.json（统计）+ 若干张人工核查拼图。
"""
from paths import DATA_RAW, EXPERIMENTS, RESULTS  # noqa: E402
import os, sys, json
import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loveda_dataset import build_index          # noqa: E402
from evaluate import predict_tile, load_model    # noqa: E402
from fit_threshold import prep                   # noqa: E402
from imgio import imread, imwrite  # noqa: E402

ROOT = str(DATA_RAW)
RUN = str(EXPERIMENTS / 'V2')
OUT = str(RESULTS / 'step0_probe')
os.makedirs(OUT, exist_ok=True)

# 类标：1 背景 2 建筑 3 道路 4 水体 5 裸地 6 林地 7 农田
BG, WATER, AGRI = 1, 4, 7
NAME = {1: 'BG', 2: 'BLDG', 3: 'ROAD', 4: 'WATER', 5: 'BARREN', 6: 'FOREST', 7: 'AGRI'}
COLOR = {1: (170, 170, 170), 2: (200, 60, 60), 3: (60, 60, 200), 4: (200, 140, 40),
         5: (120, 170, 190), 6: (40, 110, 40), 7: (80, 200, 80)}

items = build_index(ROOT, 'Val')
z = np.load(os.path.join(RUN, 'prob_val_ds4.npz'))
prob, gt_ds = z['prob'], z['gt']
assert len(items) == prob.shape[0], (len(items), prob.shape)
print(f'Val tiles {len(items)}  prob {prob.shape}')

rural_idx = [i for i, (_, _, d) in enumerate(items) if d == 'rural']
print('rural tiles', len(rural_idx))

# ---------- 1. 统计每一幅乡村图的「农田被让成背景/水体」 ----------
rows = []
for i in rural_idx:
    g, p = gt_ds[i], prob[i].argmax(0) + 1
    agri = (g == AGRI)
    if agri.sum() < 200:
        continue
    bad = agri & ((p == BG) | (p == WATER))
    rows.append(dict(idx=i, tile=os.path.basename(items[i][0])[:-4],
                     agri_ds=int(agri.sum()), bad_ds=int(bad.sum()),
                     bad_ratio=float(bad.sum() / max(agri.sum(), 1))))

tot_agri = sum(r['agri_ds'] for r in rows)
tot_bad = sum(r['bad_ds'] for r in rows)
print(f'[rural] agri px(ds4)={tot_agri}  ->BG/WATER px={tot_bad}  ({tot_bad/max(tot_agri,1):.1%})')

# 误判去向拆分
to_bg = to_wa = 0
for i in rural_idx:
    g, p = gt_ds[i], prob[i].argmax(0) + 1
    m = (g == AGRI)
    to_bg += int((m & (p == BG)).sum())
    to_wa += int((m & (p == WATER)).sum())
print(f'   ->BG {to_bg} ({to_bg/max(tot_bad,1):.1%} of bad)   ->WATER {to_wa} ({to_wa/max(tot_bad,1):.1%})')

# ---------- 2. 选候选：每图一块，取最大连通块 ----------
cands = []
for r in rows:
    i = r['idx']
    g, p = gt_ds[i], prob[i].argmax(0) + 1
    m = ((g == AGRI) & ((p == BG) | (p == WATER))).astype(np.uint8)
    if m.sum() < 200:
        continue
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m, 8)
    if n < 2:
        continue
    k = 1 + int(np.argmax(stats[1:, 4]))
    area = int(stats[k, 4])
    if area < 120:                      # ds4 下 120 px ≈ 原图 1920 px
        continue
    cx, cy = cent[k]
    # 该块内误判去向
    sub = (lab == k)
    nb = int((p[sub] == BG).sum()); nw = int((p[sub] == WATER).sum())
    cands.append(dict(tile=r['tile'], idx=i, ds_area=area,
                      px=cx * 4, py=cy * 4,
                      dom='BG' if nb >= nw else 'WATER',
                      n_bg=nb, n_water=nw, agri_bad=int(r['bad_ds'])))

cands.sort(key=lambda d: -d['ds_area'])
print(f'candidates {len(cands)}')
json.dump(dict(summary=dict(rural_agri_px_ds4=tot_agri, bad_px_ds4=tot_bad,
                            bad_ratio=tot_bad / max(tot_agri, 1),
                            to_bg=to_bg, to_water=to_wa),
               candidates=cands), open(os.path.join(OUT, 'candidates.json'), 'w'),
          ensure_ascii=False, indent=1)

# ---------- 3. 全分辨率预测 ----------
PICK = cands[:12]
DEV = torch.device('cuda')
model = load_model('v1', os.path.join(RUN, 'best.pth'), DEV)
PAD, CELL = 320, 470          # 裁 320×320，放大显示 470
COLS, ROWS = 2, 3

def overlay(img_rgb, gt, pred, box):
    """原图 + 真值农田绿轮廓 + 错判红叠加。"""
    y0, y1, x0, x1 = box
    vis = img_rgb[y0:y1, x0:x1].copy()
    g, p = gt[y0:y1, x0:x1], pred[y0:y1, x0:x1]
    agri = (g == AGRI).astype(np.uint8)
    bad = ((g == AGRI) & ((p == BG) | (p == WATER))).astype(np.uint8)
    red = np.zeros_like(vis); red[:, :] = (235, 30, 30)
    m3 = bad.astype(bool)
    vis[m3] = (0.35 * vis[m3] + 0.65 * red[m3]).astype(np.uint8)
    cont, _ = cv2.findContours(agri, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cont, -1, (0, 255, 0), 1)
    return cv2.resize(vis, (CELL, CELL), interpolation=cv2.INTER_AREA)

sheets = []
idx_by_tile = {os.path.basename(it[0])[:-4]: it for it in items}
for gid in range(0, len(PICK), COLS * ROWS):
    chunk = PICK[gid:gid + COLS * ROWS]
    sheet = np.full((ROWS * (CELL + 34), COLS * CELL, 3), 255, np.uint8)
    for j, c in enumerate(chunk):
        rr, cc = divmod(j, COLS)
        ipath, mpath, _ = idx_by_tile[c['tile']]
        img = imread(ipath, cv2.IMREAD_COLOR)[:, :, ::-1]
        gt = imread(mpath, cv2.IMREAD_GRAYSCALE)
        pred = predict_tile(model, prep(ipath), DEV).cpu().numpy() + 1
        h, w = gt.shape
        cy, cx = int(c['py']), int(c['px'])
        y0 = min(max(cy - PAD // 2, 0), h - PAD); x0 = min(max(cx - PAD // 2, 0), w - PAD)
        vis = overlay(img, gt, pred, (y0, y0 + PAD, x0, x0 + PAD))
        y = rr * (CELL + 34); x = cc * CELL
        sheet[y:y + CELL, x:x + CELL] = vis
        tag = f"#{gid+j+1:02d} {c['tile']} -> {c['dom']}"
        cv2.putText(sheet, tag, (x + 4, y + CELL + 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 1, cv2.LINE_AA)
        print(' drawn', tag, flush=True)
    p = os.path.join(OUT, f'sheet_{len(sheets)+1}.png')
    imwrite(p, sheet[:, :, ::-1]); sheets.append(p)

print('sheets', sheets)
