"""dev→Val 落差的分解（整体 + 分域），以及 H1 判据②的复算。

为什么必须分域
--------------
补跑 dev 后发现：六个模型在 dev 上都是 rural > urban，在 Val 上全部反转成 urban > rural。
把两边的分域指标对起来，规律是

    urban: dev → Val 几乎不动（−3.8 ~ +2.5）
    rural: dev → Val 一致掉 20.3 ~ 23.4 点

也就是说 16.27 的"跨域落差"实际上几乎全部来自 rural 子集。在全量上做先验分解会把
这个结构糊掉，所以下面每一层都算两遍：整体一次，urban / rural 各一次。

两个问题
--------
1. **落差的多少来自"类别构成变了"，多少来自"同类之间更容易混了"？**
   取 Val 的混淆矩阵 C（行=真值，列=预测），把每行按 `s_i = f_dev[i] / f_val[i]` 缩放
   （"假设 Val 的类别构成与 dev 相同，条件误差结构不变"），重算 mIoU = prior-matched Val。
     · prior-matched ≈ dev → 落差主要是先验漂移（操作点可修）
     · prior-matched ≈ Val → 落差主要是条件误差结构变化（特征/传感器漂移）

2. **H1 判据②**：被判为背景的像素占全部误差的比例（现行协议下从未算过；
   旧协议的 54.7% 已作废）。分域也一并给出。

用法
----
  python code/gap_decompose.py --runs V1_ce V2 --splits dev val
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
DOMS = ("urban", "rural")


def confusion(run, model_name, split, ckpt, device):
    """全图滑窗预测 -> (整体 8x8, {urban: 8x8, rural: 8x8})。gt=0 的像素忽略。"""
    model = load_model(model_name, ckpt, device)
    items = build_index(DATA, split)
    cm = np.zeros((8, 8), np.int64)
    cm_by = {d: np.zeros((8, 8), np.int64) for d in DOMS}
    for i, (ip, mp, dom) in enumerate(items, 1):
        img = imread(ip, cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32) / 255.0
        img = torch.from_numpy(((img - MEAN) / STD).transpose(2, 0, 1).copy())
        pred = predict_tile(model, img, device).cpu().numpy()
        gt = imread(mp, cv2.IMREAD_GRAYSCALE)
        if pred.shape != gt.shape:
            pred = cv2.resize(pred.astype(np.uint8), (gt.shape[1], gt.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        m = gt != 0
        np.add.at(cm, (gt[m].ravel(), pred[m].ravel()), 1)
        if dom in cm_by:
            np.add.at(cm_by[dom], (gt[m].ravel(), pred[m].ravel()), 1)
        if i % 100 == 0:
            print(f"    {i}/{len(items)}", flush=True)
    return cm, cm_by


def miou_from_cm(cm):
    out = []
    for c in range(1, 8):
        tp = cm[c, c]
        fp = cm[1:, c].sum() - tp
        fn = cm[c, 1:].sum() - tp
        den = tp + fp + fn
        out.append(tp / den if den else np.nan)
    return float(np.nanmean(out)), out


def oa_from_cm(cm):
    total = cm[1:, 1:].sum()
    return float(np.trace(cm[1:, 1:]) / total) if total else 0.0


def class_freq(cm):
    """真值类别构成（1..7），按像素计。"""
    tot = cm[1:, :].sum()
    return cm[1:, :].sum(1) / tot if tot else np.zeros(7)


def prior_matched(cm_src, cm_dst):
    """把 cm_dst 的类别构成换成 cm_src 的，条件误差结构保持不变。"""
    f_src, f_dst = class_freq(cm_src), class_freq(cm_dst)
    scale = np.where(f_dst > 0, f_src / np.maximum(f_dst, 1e-12), 0.0)
    cm_pm = cm_dst.astype(np.float64).copy()
    cm_pm[1:, :] = cm_pm[1:, :] * scale[:, None]
    return miou_from_cm(cm_pm)[0], f_src, f_dst


def bg_absorb(cm):
    """被判为背景的像素占全部误差的比例。"""
    off = cm[1:, 1:].sum() - np.trace(cm[1:, 1:])
    # cm[1,1] is "true background predicted background" -- a correct prediction, not an
    # error. Counting it pushed the ratio above 100% in the first run.
    to_bg = cm[1:, 1].sum() - cm[1, 1]
    return float(to_bg / off) if off else 0.0


def error_flows(cm, k=6):
    """误差流向 top-k：(真值类 -> 预测类, 占全部误差的比例)。

    比 prior-matched 更有解释力。原因：mIoU 是宏平均，IoU_c 只是 TP/(TP+FP+FN)，
    与类别像素数无关，所以"类别构成变了"本身不会改变 mIoU（自检里构造过：构成完全
    不同而每类条件准确率相同时，mIoU 差 0.00）。这意味着 20+ 点的落差必然来自
    "同一类更难分了"，而流向表能直接指出难在哪一对上。
    """
    off = cm[1:, 1:].astype(np.float64).copy()
    np.fill_diagonal(off, 0)
    tot = off.sum()
    if not tot:
        return []
    order = np.argsort(off.ravel())[::-1][:k]
    out = []
    for flat in order:
        i, j = divmod(int(flat), off.shape[1])
        if off[i, j] <= 0:
            continue
        out.append((CLASSES[i], CLASSES[j], float(off[i, j] / tot)))
    return out


def composition_l1(cm_a, cm_b):
    """两个混淆矩阵的真值类别构成 L1 距离（0=相同，2=完全不重叠）。"""
    return float(np.abs(class_freq(cm_a) - class_freq(cm_b)).sum())


def jsonable(o):
    """递归转成可序列化对象。np.float64 藏在 list 里时手写转换很容易漏。"""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (list, tuple)):
        return [jsonable(x) for x in o]
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    return o


def load_cached(run, split):
    """从 cm_{split}.json 读回，跳过重复预测（预测一轮要 12 min，分析只要 1 s）。"""
    def rd(p):
        return np.array(json.load(open(p)), dtype=np.int64)
    base = os.path.join(EXP, run)
    cm = rd(os.path.join(base, f"cm_{split}.json"))
    cm_by = {d: rd(os.path.join(base, f"cm_{split}_{d}.json")) for d in DOMS}
    return cm, cm_by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["V1_ce", "V2"])
    ap.add_argument("--models", nargs="+", default=["v1", "v1"])
    ap.add_argument("--configs", nargs="+", default=["e0_ce", "v2"])
    ap.add_argument("--splits", nargs="+", default=["dev", "val"])
    ap.add_argument("--from-cache", action="store_true",
                    help="reuse cm_*.json instead of re-running inference (12 min -> 1 s)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    for run, mname, cfgname in zip(args.runs, args.models, args.configs):
        ckpt = os.path.join(EXP, run, "best.pth")
        results[run] = {}
        for split in args.splits:
            if args.from_cache:
                print(f"\n[{run}/{split}] loading cached confusion matrix ...", flush=True)
                cm, cm_by = load_cached(run, split)
            else:
                print(f"\n[{run}/{split}] confusion matrix ...", flush=True)
                cm, cm_by = confusion(run, mname, split, ckpt, device)
                json.dump(cm.tolist(), open(os.path.join(EXP, run, f"cm_{split}.json"), "w"))
                for d in DOMS:
                    json.dump(cm_by[d].tolist(),
                              open(os.path.join(EXP, run, f"cm_{split}_{d}.json"), "w"))
            m, per = miou_from_cm(cm)
            results[run][split] = dict(
                cm=cm, mIoU=m, OA=oa_from_cm(cm), per=per,
                cm_by=cm_by,
                mIoU_by={d: miou_from_cm(cm_by[d])[0] for d in DOMS},
            )
            print("  mIoU=%.2f  OA=%.2f  | urban %.2f  rural %.2f" % (
                m * 100, oa_from_cm(cm) * 100,
                results[run][split]["mIoU_by"]["urban"] * 100,
                results[run][split]["mIoU_by"]["rural"] * 100))

    # ---------- H1 判据② ----------
    print("\n" + "=" * 80)
    print("H1 判据②：被判为背景的像素占全部误差的比例（整体 / 分域）")
    for run, r in results.items():
        for split, d in r.items():
            line = f"  {run:<8} {split:<4} 全量 {bg_absorb(d['cm'])*100:5.1f}%"
            line += "".join(f"   {dm} {bg_absorb(d['cm_by'][dm])*100:5.1f}%" for dm in DOMS)
            print(line)

    # ---------- 落差分解 ----------
    print("\n" + "=" * 80)
    print("落差分解：先验漂移能解释多少（整体 + 分域）")
    for run, r in results.items():
        if "dev" not in r or "val" not in r:
            continue
        m_pm, f_dev, f_val = prior_matched(r["dev"]["cm"], r["val"]["cm"])
        gap = (r["dev"]["mIoU"] - r["val"]["mIoU"]) * 100
        closed = (m_pm - r["val"]["mIoU"]) * 100
        r["prior_matched_val"] = m_pm
        r["gap"] = gap
        r["gap_from_prior"] = closed
        print(f"\n  {run}  全量: dev {r['dev']['mIoU']*100:.2f} | Val {r['val']['mIoU']*100:.2f} "
              f"| 落差 {gap:.2f}")
        print(f"    prior-matched {m_pm*100:.2f} → 先验漂移解释 {closed:.2f} 点 "
              f"（{closed/gap*100 if gap else 0:.0f}%），残差 {gap-closed:.2f} 点")

        r["by_domain"] = {}
        for dm in DOMS:
            cdev, cval = r["dev"]["cm_by"][dm], r["val"]["cm_by"][dm]
            m_pm_d, fd, fv = prior_matched(cdev, cval)
            g = (miou_from_cm(cdev)[0] - miou_from_cm(cval)[0]) * 100
            cl = (m_pm_d - miou_from_cm(cval)[0]) * 100
            r["by_domain"][dm] = dict(gap=g, from_prior=cl, prior_matched=m_pm_d)
            share = f"{cl/g*100:4.0f}%" if g >= 1.0 else " n/a"  # 负落差时百分比没有意义
            print(f"    {dm:<6} dev {miou_from_cm(cdev)[0]*100:6.2f} | "
                  f"Val {miou_from_cm(cval)[0]*100:6.2f} | 落差 {g:6.2f} | "
                  f"先验解释 {cl:5.2f}（{share}）| 残差 {g-cl:5.2f}")
            print("           构成 " + " ".join(
                f"{CLASSES[i][:4]} {fd[i]:.3f}→{fv[i]:.3f}" for i in range(7)))

    # ---------- 落差的去向：rural 上的逐类变化与误差流向 ----------
    print("\n" + "=" * 80)
    print("rural 的逐类 IoU：dev → Val")
    for run, r in results.items():
        if "dev" not in r or "val" not in r:
            continue
        _, di = miou_from_cm(r["dev"]["cm_by"]["rural"])
        _, vi = miou_from_cm(r["val"]["cm_by"]["rural"])
        r["rural_per_class_delta"] = {CLASSES[i]: float((vi[i] - di[i]) * 100)
                                      for i in range(7)}
        print(f"  {run:<8} " + "  ".join(
            f"{CLASSES[i][:4]} {di[i]*100:5.1f}→{vi[i]*100:5.1f}({(vi[i]-di[i])*100:+5.1f})"
            for i in range(7)))

    print("\n" + "=" * 80)
    print("rural 的误差流向（占该域全部误差的比例）：dev 的前 3 大 vs Val 的前 3 大")
    for run, r in results.items():
        if "dev" not in r or "val" not in r:
            continue
        fd_ = error_flows(r["dev"]["cm_by"]["rural"], k=3)
        fv_ = error_flows(r["val"]["cm_by"]["rural"], k=3)
        r["rural_flows"] = {"dev": fd_, "val": fv_}
        print(f"\n  {run}")
        print("    dev: " + " | ".join(f"{a}→{b} {p*100:.1f}%" for a, b, p in fd_))
        print("    Val: " + " | ".join(f"{a}→{b} {p*100:.1f}%" for a, b, p in fv_))

    print("\n" + "=" * 80)
    print("类别构成的 L1 距离（说明构成确实变了，但 mIoU 对它免疫，不是落差的来源）")
    for run, r in results.items():
        if "dev" not in r or "val" not in r:
            continue
        l1_all = composition_l1(r["dev"]["cm"], r["val"]["cm"])
        l1_u = composition_l1(r["dev"]["cm_by"]["urban"], r["val"]["cm_by"]["urban"])
        l1_r = composition_l1(r["dev"]["cm_by"]["rural"], r["val"]["cm_by"]["rural"])
        print(f"  {run:<8} 全量 {l1_all:.3f} | urban {l1_u:.3f} | rural {l1_r:.3f}")

    # ---------- save ----------
    out = {}
    for run, r in results.items():
        d = {}
        for k, v in r.items():
            if k == "cm_by":
                d[k] = {kk: vv.tolist() for kk, vv in v.items()}
            elif isinstance(v, np.ndarray):
                d[k] = v.tolist()
            elif isinstance(v, dict):
                d[k] = {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                        for kk, vv in v.items()}
            else:
                d[k] = v
        out[run] = d
    path = os.path.join(EXP, "gap_decompose.json")
    json.dump(jsonable(out), open(path, "w"), ensure_ascii=False, indent=1)
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
