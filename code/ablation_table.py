"""V2 三件套的单因素消融：汇总表 + 边际增益图。

口径（写进论文时照这个说）
--------------------------
三条消融都是"**在 V1 上只加一项**"，不是"从 V2 上拿掉一项"：

    V1        朴素 CE                                      ← 基准
    AB_w      + 逐图自适应加权（含份额对齐）                    ← 单因素
    AB_dice   + Dice (α=0.5)                               ← 单因素
    AB_bnd    + 边界加权（边界像素 ×2）                        ← 单因素
    V2        三项全开                                       ← 组合

因此每个单因素行报的是 `AB_x − V1`。**三项之和 ≠ V2 − V1**，差额是交互项；
本脚本会把这个差额显式算出来并打印，避免有人把三项相加当成 V2 的分解。

数据来源：`experiments/<run>/eval_val.json`（evaluate.py 的全图滑窗结果）与
`train_log.csv`（训练期 512 center-crop 监控值，仅作参照，不与全图混用）。

用法
----
    python code/ablation_table.py            # 默认读新的消融组，出表 + 图
    python code/ablation_table.py --no-fig   # 只出表
"""
from paths import EXPERIMENTS, FIGURES, TABLES  # noqa: E402
import argparse
import json
import os

EXP = str(EXPERIMENTS)
OUT = str(TABLES)
FIG = str(FIGURES)
CLASSES = ["background", "building", "road", "water", "barren", "forest", "agriculture"]
CLS_CN = dict(zip(CLASSES, ["背景", "建筑", "道路", "水体", "裸地", "林地", "农田"]))

DEFAULT = [
    ("V1_ce", "V1（朴素 CE）"),
    ("AB_w", "+ 逐图加权"),
    ("AB_dice", "+ Dice"),
    ("AB_bnd", "+ 边界加权"),
    ("V2", "V2（三项全开）"),
]


def load_eval(run):
    p = os.path.join(EXP, run, "eval_val.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p, encoding="utf-8"))
    return dict(mIoU=d["mIoU"], OA=d["OA"], mF1=d.get("mF1"),
                urban=d.get("per_domain", {}).get("urban"),
                rural=d.get("per_domain", {}).get("rural"),
                per_class={k: v["IoU"] for k, v in d["per_class"].items()})


def monitor_dev(run):
    p = os.path.join(EXP, run, "train_log.csv")
    if not os.path.exists(p):
        return None, None
    rows = [r.strip().split(",") for r in open(p).read().strip().split("\n")[1:] if r.strip()]
    if not rows:
        return None, None
    best = max(rows, key=lambda r: float(r[3]))
    return float(best[3]) * 100, int(best[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    rows, base = [], None
    for run, label in DEFAULT:
        m = load_eval(run)
        if m is None:
            print(f"[skip] {run}: 无 eval_val.json（可能还没跑完）")
            rows.append(dict(run=run, label=label, mIoU=None))
            continue
        dev, ep = monitor_dev(run)
        if base is None:
            base = m["mIoU"]
        rows.append(dict(run=run, label=label, mIoU=m["mIoU"], OA=m["OA"], mF1=m["mF1"],
                         urban=m["urban"], rural=m["rural"], per_class=m["per_class"],
                         dev_monitor=dev, best_ep=ep))

    ok = [r for r in rows if r["mIoU"] is not None]
    if not ok:
        print("没有任何可用的结果，退出。")
        return

    print(f"\n{'实验':<16}{'Val mIoU':>9}{'Δ vs V1':>9}{'OA':>7}{'Urban':>7}{'Rural':>7}"
          f"{'dev(监控)':>11}{'best ep':>8}")
    for r in rows:
        if r["mIoU"] is None:
            print(f"{r['label']:<16}{'—':>9}")
            continue
        d = (r["mIoU"] - base) * 100
        print(f"{r['label']:<16}{r['mIoU']*100:9.2f}{d:+9.2f}{r['OA']*100:7.2f}"
              f"{r['urban']*100:7.2f}{r['rural']*100:7.2f}"
              f"{(r['dev_monitor'] or 0):11.2f}{(r['best_ep'] or 0):8d}")

    single = {r["run"]: (r["mIoU"] - base) * 100 for r in ok if r["run"].startswith("AB_")}
    v2 = next((r for r in ok if r["run"] == "V2"), None)
    s = combo = None
    if v2 is not None and len(single) == 3:
        combo = (v2["mIoU"] - base) * 100
        s = sum(single.values())
        print(f"\n单因素之和 {s:+.2f} | 组合增益 {combo:+.2f} | **交互项 {combo - s:+.2f}**")
        print("  → 论文只能写\"主导项是 X\"，不能把三项相加当成 V2 的分解。")
        lead = max(single, key=single.get)
        print(f"  主导项：{lead}（{single[lead]:+.2f}）")

    print("\n逐类 IoU")
    print(f"{'类别':<8}" + "".join(f"{r['label'][:10]:>12}" for r in ok))
    for c in CLASSES:
        print(f"{CLS_CN[c]:<8}" + "".join(f"{r['per_class'][c]*100:12.2f}" for r in ok))

    md = ["# V2 三件套单因素消融（全图滑窗，官方 Val 1669 幅）", "",
          "> 每条消融都是「在 V1 上只加一项」。三项单因素增益之和 ≠ 组合增益，差额为交互项。", "",
          "| 实验 | Val mIoU | Δ vs V1 | OA | Urban | Rural | dev（监控） | best ep |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["mIoU"] is None:
            md.append(f"| {r['label']} | — | — | — | — | — | — | — |")
            continue
        md.append(f"| {r['label']} | {r['mIoU']*100:.2f} | {(r['mIoU']-base)*100:+.2f} | "
                  f"{r['OA']*100:.2f} | {r['urban']*100:.2f} | {r['rural']*100:.2f} | "
                  f"{(r['dev_monitor'] or 0):.2f} | {r['best_ep'] or 0} |")
    if s is not None:
        md += ["", f"单因素之和 **{s:+.2f}**，组合增益 **{combo:+.2f}**，交互项 **{combo - s:+.2f}**。"]
    p = os.path.join(OUT, "ablation_v2_factors.md")
    open(p, "w", encoding="utf-8").write("\n".join(md) + "\n")
    print(f"\nsaved -> {p}")

    if args.no_fig or v2 is None:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception as e:
        print(f"[fig] matplotlib 不可用（{e}），跳过作图")
        return
    labels = [r["label"] for r in ok]
    vals = [(r["mIoU"] - base) * 100 for r in ok]
    colors = ["#8c8c8c"] + ["#2e7d5b"] * 3 + ["#c0392b"]
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=200)
    bars = ax.bar(range(len(vals)), vals, color=colors[:len(vals)])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=12, ha="right", fontsize=9)
    ax.set_ylabel("Δ mIoU vs V1（点）")
    # [图内不设标题/脚注，全部说明移至图题]
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.12 if v >= 0 else -0.35),
                f"{v:+.2f}", ha="center", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    lo, hi = min(vals), max(vals)
    ax.set_ylim(lo - 0.55, hi + 0.55)
    # 交互项必须写在图上：三项之和 -0.28 而组合是 +3.00，差额才是主体
    inter = (v2["mIoU"] - base) * 100 - sum(vals[1:-1])
    ax.text(0.025, 0.97,
            f"三项之和 {sum(vals[1:-1]):+.2f}　→　组合 {vals[-1]:+.2f}\n"
            f"交互项 {inter:+.2f}（主导）",
            transform=ax.transAxes, ha="left", va="top", fontsize=8,
            color="#7b341e",
            bbox=dict(fc="#fff7ed", ec="#c05621", lw=0.7, alpha=0.95, pad=2.5))
    os.makedirs(FIG, exist_ok=True)
    fp = os.path.join(FIG, "fig8b_ablation_v2.png")
    fig.tight_layout()
    fig.savefig(fp)
    print(f"saved -> {fp}")


if __name__ == "__main__":
    main()
