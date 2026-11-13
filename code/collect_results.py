"""Collect every finished run into one snapshot, and audit what is missing.

Why this exists: metrics used to live in three different shapes -- ``eval_val.json``
from evaluate.py, ``pah_eval_{dev,val}.json`` from eval_pah.py (V3 only), and plain
stdout for dev before eval_split.py learned to dump JSON. Writing the report meant
grepping logs and hand-copying numbers, which is exactly how transcription errors
happen. This script reads all shapes and emits one table.

Usage
-----
    python code/collect_results.py            # write report/tables/results_summary.{md,json}
    python code/collect_results.py --check    # only audit missing artifacts, no files written

The JSON sidecar is meant to be imported by figure scripts, so the numbers have a
single source of truth even as runs are added.
"""
from paths import ROOT, TABLES  # noqa: E402
import argparse
import csv
import json
import os
import sys

# repo root from code/paths.py
EXP = os.path.join(ROOT, "experiments")
OUTDIR = str(TABLES)

# Display order: the ablation ladder first, then baselines, then the V2 ablations.
ORDER = ["V1_ce", "V2", "V3", "V5", "E10_unet", "E11_deeplab", "E12_segformer",
         "AB_w", "AB_dice", "AB_bnd"]
LABEL = {"V1_ce": "V1", "V2": "V2", "V3": "V3", "V5": "V5",
         "E10_unet": "E10 UNet-R34", "E11_deeplab": "E11 DeepLabV3+",
         "E12_segformer": "E12 SegFormer-B0",
         "AB_w": "AB 只加权", "AB_dice": "AB 只 Dice", "AB_bnd": "AB 只边界"}

# Artifacts that must survive for the report to be writable without re-running anything.
CORE = {"best.pth": "权重（推理/复现必需）",
        "last.pth": "末轮权重（含优化器，续训/协议噪声用）",
        "train_log.csv": "逐 epoch 曲线（best ep、收敛判断）",
        "config.yaml": "训练配置快照（可复现的关键）"}
EVAL = {"eval_val.json": "Val 全图指标",
        "eval_dev.json": "dev 全图指标"}


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def pick_eval(run_dir, split):
    """Return (metrics_dict, source_tag) for a split, tolerating the V3 schema."""
    std = load_json(os.path.join(run_dir, f"eval_{split}.json"))
    if std:
        return std, f"eval_{split}.json"
    pah = load_json(os.path.join(run_dir, f"pah_eval_{split}.json"))
    if pah and "mode_none" in pah:
        m = dict(pah["mode_none"])
        # V3's per_domain is nested by mode ({'none': {...}, 'pred': {...}}); we report
        # the uncorrected branch because that is what V3's headline number uses.
        pd = pah.get("per_domain") or {}
        if isinstance(pd, dict) and "none" in pd:
            pd = pd["none"]
        m["per_domain"] = pd
        return m, f"pah_eval_{split}.json:mode_none"
    return None, ""


def parse_dev_log(path, run, model):
    """Rebuild eval_dev.json from a stdout log.

    Needed because eval_split.py used to print without dumping, so V2/V5's dev numbers
    exist only as text. V1_ce has no dev log at all (it was run interactively), so that
    one must be re-evaluated rather than reconstructed.
    """
    import re
    if not os.path.exists(path):
        return None
    txt = open(path, encoding="utf-8", errors="ignore").read()
    m = re.search(r"split=dev\s+mIoU=([\d.]+)\s+OA=([\d.]+)", txt)
    if not m:
        return None
    rows = re.findall(r"^\s{2}(\w+)\s+IoU=\s*([\d.]+)\s+R=\s*([\d.]+)\s+P=\s*([\d.]+)",
                      txt, re.M)
    if not rows:
        return None
    pc, f1s = {}, []
    for name, iou, rec, pre in rows:
        iou, rec, pre = float(iou) / 100, float(rec) / 100, float(pre) / 100
        f1 = 0.0 if pre + rec == 0 else 2 * pre * rec / (pre + rec)
        f1s.append(f1)
        pc[name] = dict(IoU=iou, Recall=rec, Precision=pre, F1=f1)
    return dict(run=run, model=model, split="dev", n_tiles=None,
                mIoU=float(m.group(1)) / 100, OA=float(m.group(2)) / 100,
                mF1=sum(f1s) / len(f1s), per_class=pc, per_domain={},
                source=f"parsed from {os.path.basename(path)}（per_domain 缺失，需重跑补齐）")


def backfill_dev():
    """Write eval_dev.json for runs whose dev numbers only exist in logs."""
    logmap = {"V2": "logs/V2_dev.log", "V5": "logs/V5_dev.log"}
    done = []
    for run, rel in logmap.items():
        d = os.path.join(EXP, run)
        out = os.path.join(d, "eval_dev.json")
        if not os.path.isdir(d) or os.path.exists(out):
            continue
        r = parse_dev_log(os.path.join(ROOT, rel), run, "v1" if run == "V2" else "v5")
        if r:
            with open(out, "w", encoding="utf-8") as f:
                json.dump(r, f, ensure_ascii=False, indent=1)
            done.append(run)
    return done


def param_count(model_name):
    """Actual parameter count, so the table never carries a hand-typed number."""
    try:
        sys.path.insert(0, os.path.join(ROOT, "code"))
        from models import build_model
        m = build_model(model_name, num_classes=8)
        return sum(p.numel() for p in m.parameters()) / 1e6
    except Exception:
        return None


def collect():
    if not os.path.isdir(EXP):
        sys.exit(f"no experiments dir: {EXP}")
    runs = [d for d in sorted(os.listdir(EXP))
            if os.path.isdir(os.path.join(EXP, d)) and not d.startswith("_")]
    runs.sort(key=lambda r: ORDER.index(r) if r in ORDER else 99)

    rows, audit = [], {}
    pcount = {}
    for run in runs:
        d = os.path.join(EXP, run)
        cfg = {}
        cfg_path = os.path.join(d, "config.yaml")
        if os.path.exists(cfg_path):
            try:
                import yaml
                cfg = yaml.safe_load(open(cfg_path, encoding="utf-8")) or {}
            except Exception:
                cfg = {}
        model = (cfg.get("model") or {}).get("name") if isinstance(cfg.get("model"), dict) else None
        if model is None:
            # config.yaml snapshots store the model under different keys depending on
            # which train.py version wrote them; fall back to the run-name convention.
            model = {"V1_ce": "v1", "V2": "v1", "V3": "v3", "V5": "v5",
                     "E10_unet": "unet_r34", "E11_deeplab": "deeplabv3p_r50",
                     "E12_segformer": "segformer_b0"}.get(run)
        if model and model not in pcount:
            pcount[model] = param_count(model)

        log = read_csv(os.path.join(d, "train_log.csv"))
        best_ep = best_dev = last_dev = mins = None
        if log:
            b = max(log, key=lambda r: float(r["dev_mIoU"]))
            best_ep = int(b["epoch"])
            best_dev = float(b["dev_mIoU"]) * 100
            last_dev = float(log[-1]["dev_mIoU"]) * 100
            mins = sum(float(r["epoch_time_s"]) for r in log) / 60

        val, val_src = pick_eval(d, "val")
        dev, dev_src = pick_eval(d, "dev")
        target = None
        try:
            target = int((cfg.get("train") or {}).get("epochs") or 0) or None
        except Exception:
            pass

        rows.append(dict(
            run=run, label=LABEL.get(run, run), model=model,
            params=pcount.get(model), best_ep=best_ep, best_dev=best_dev,
            last_dev=last_dev, epochs=len(log) if log else 0, train_min=mins,
            target_ep=target,
            val=val, val_src=val_src, dev=dev, dev_src=dev_src,
        ))

        missing = [f for f in CORE if not os.path.exists(os.path.join(d, f))]
        # Val/dev are judged by whether a usable file was actually found, not by filename:
        # V3 legitimately keeps its numbers in pah_eval_*.json instead of eval_*.json.
        for split, m in (("val", val), ("dev", dev)):
            if m is None:
                missing.append(f"eval_{split}.json（或 pah_eval_{split}.json）")
        audit[run] = missing

    return rows, audit


def fmt(v, nd=2, dash="—"):
    return dash if v is None else f"{v:.{nd}f}"


def write_outputs(rows):
    os.makedirs(OUTDIR, exist_ok=True)

    # ---- JSON sidecar: single source of truth for figure scripts ----
    js = []
    for r in rows:
        v = r["val"] or {}
        dv = r["dev"] or {}
        js.append(dict(
            run=r["run"], label=r["label"], model=r["model"], params_M=r["params"],
            epochs=r["epochs"], best_epoch=r["best_ep"],
            train_min=None if r["train_min"] is None else round(r["train_min"], 1),
            val_mIoU=None if not v else v.get("mIoU"),
            val_OA=None if not v else v.get("OA"),
            val_mF1=None if not v else v.get("mF1"),
            val_per_class=None if not v else v.get("per_class"),
            val_per_domain=None if not v else v.get("per_domain"),
            dev_mIoU=None if not dv else dv.get("mIoU"),
            dev_OA=None if not dv else dv.get("OA"),
            dev_per_class=None if not dv else dv.get("per_class"),
            dev_monitor_best=None if r["best_dev"] is None else r["best_dev"] / 100,
            sources=dict(val=r["val_src"], dev=r["dev_src"]),
        ))
    with open(os.path.join(OUTDIR, "results_summary.json"), "w", encoding="utf-8") as f:
        json.dump(js, f, ensure_ascii=False, indent=1)

    # ---- Markdown ----
    L = ["# 结果总表（自动生成，勿手改）", "",
         f"由 `code/collect_results.py` 从 `experiments/*/` 抽取，生成于运行时刻。"
         f"Val/dev 均为 1024×1024 全图滑窗（window 512, stride 256）。", "",
         "## 主表", "",
         "| run | 模型 | 参数(M) | ep | Val mIoU | Val OA | mF1 | Urban | Rural | dev(全量) | 落差 | 训练(min) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        v, dv = r["val"] or {}, r["dev"] or {}
        vm, dm = v.get("mIoU"), dv.get("mIoU")
        gap = None if (vm is None or dm is None) else (dm - vm) * 100
        pdom = v.get("per_domain") or {}
        ep_txt = "—" if not r["best_ep"] else str(r["best_ep"])
        if r["target_ep"] and r["epochs"] < r["target_ep"]:
            ep_txt = f"{ep_txt} ⏳{r['epochs']}/{r['target_ep']}"
        L.append("| {lbl} | {mdl} | {p} | {ep} | {vm} | {vo} | {vf} | {u} | {rr} | {dm} | {gap} | {tm} |".format(
            lbl=r["label"], mdl=r["model"] or "—",
            p=fmt(r["params"], 2), ep=ep_txt,
            vm=fmt(None if vm is None else vm * 100), vo=fmt(None if not v else (v.get("OA") or 0) * 100),
            vf=fmt(None if not v else (v.get("mF1") or 0) * 100),
            u=fmt(None if not pdom.get("urban") else pdom["urban"] * 100),
            rr=fmt(None if not pdom.get("rural") else pdom["rural"] * 100),
            dm=fmt(None if dm is None else dm * 100),
            gap=fmt(gap), tm=fmt(r["train_min"], 1)))

    # ---- per-class block ----
    L += ["", "## 逐类 IoU（Val）", ""]
    cls = ["background", "building", "road", "water", "barren", "forest", "agriculture"]
    L.append("| run | " + " | ".join(cls) + " |")
    L.append("|---" * (len(cls) + 1) + "|")
    for r in rows:
        v = r["val"] or {}
        pc = v.get("per_class") or {}
        cells = [fmt(None if c not in pc else pc[c]["IoU"] * 100) for c in cls]
        L.append(f"| {r['label']} | " + " | ".join(cells) + " |")

    # ---- audit ----
    L += ["", "## 产物完整性检查", "",
          "| run | 缺失产物 |", "|---|---|"]
    for r in rows:
        miss = audit_cache.get(r["run"], [])
        L.append(f"| {r['label']} | {'、'.join(miss) if miss else '✅ 齐全'} |")
    L += ["",
          "> `best.pth` / `last.pth` / `train_log.csv` / `config.yaml` 不可再生，必须保留；"
          "`prob_*_ds4.npz`（概率缓存）可随时由 `fit_threshold.py` 重算，不在检查范围内。",
          "",
          "## 口径提醒", "",
          "- **dev(全量)** 来自 `eval_split.py`（1024 全图），与 `train_log.csv` 里的 `dev_mIoU`"
          "（训练期 center-crop 512 监控）**不是同一个数**，两者不可混用。",
          "- V3 的指标取自 `pah_eval_{dev,val}.json` 的 `mode_none`（即不校正模式），"
          "因为 V3 走的是 `eval_pah.py` 而非 `evaluate.py`。",
          "- 跨 run 比较的可解读下限：Val **0.65**、dev **1.5**（协议噪声实测）。"]

    with open(os.path.join(OUTDIR, "results_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    return os.path.join(OUTDIR, "results_summary.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only audit, write nothing")
    ap.add_argument("--backfill-dev", action="store_true",
                    help="rebuild eval_dev.json for runs whose dev numbers live only in logs")
    args = ap.parse_args()

    if args.backfill_dev:
        done = backfill_dev()
        print(f"回填 eval_dev.json: {done if done else '无需回填（已存在或无日志）'}")

    global audit_cache
    rows, audit_cache = collect()

    if args.check:
        bad = 0
        for run, miss in audit_cache.items():
            flag = "OK  " if not miss else "MISS"
            print(f"[{flag}] {run:<16} {'、'.join(miss) if miss else '齐全'}")
            bad += bool(miss)
        print(f"\n{len(rows)} 个 run，{bad} 个有缺失")
        return

    out = write_outputs(rows)
    print(f"{len(rows)} 个 run -> {out}")
    for run, miss in audit_cache.items():
        if miss:
            print(f"  ! {run}: 缺 {'、'.join(miss)}")


audit_cache = {}
if __name__ == "__main__":
    main()
