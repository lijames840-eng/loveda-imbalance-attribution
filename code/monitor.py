"""Print training progress and an ETA from ``train_log.csv``.

Usage:
    python code/monitor.py --run V1_ce --epochs 30

The ETA uses the mean epoch time of the *completed* epochs, so it self-corrects as
soon as one or two epochs have finished.  Nothing is written; safe to call at any
time while training runs.
"""
from paths import EXPERIMENTS  # noqa: E402
import argparse
import csv
import os
from datetime import datetime, timedelta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="V1_ce")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--exp", default=str(EXPERIMENTS))
    ap.add_argument("--tail", type=int, default=3, help="show last N epochs")
    args = ap.parse_args()

    path = os.path.join(args.exp, args.run, "train_log.csv")
    if not os.path.exists(path):
        print(f"[{args.run}] 日志尚未生成（模型正在初始化或已结束）")
        return
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if not rows:
        print(f"[{args.run}] 日志为空（第一个 epoch 还没跑完）")
        return

    times = [float(r["epoch_time_s"]) for r in rows]
    done = len(rows)
    avg = sum(times) / len(times)
    elapsed = sum(times)
    remain = max(args.epochs - done, 0) * avg
    eta = datetime.now() + timedelta(seconds=remain)

    print(f"[{args.run}]  {done}/{args.epochs} epoch   已用 {elapsed / 60:.1f} min")
    for r in rows[-args.tail:]:
        print(f"    ep{r['epoch']:>2s}  mIoU={float(r['dev_mIoU']) * 100:5.2f}  "
              f"OA={float(r['dev_OA']) * 100:5.2f}  loss={r['train_loss']}  "
              f"{float(r['epoch_time_s']):.0f}s")
    print(f"    平均 {avg:.0f}s/epoch（最近一次 {times[-1]:.0f}s）")
    if done >= 2:
        print(f"    剩余 {remain / 60:.1f} min  预计完成 {eta.strftime('%H:%M')}")
    else:
        print(f"    剩余约 {remain / 60:.1f} min（仅 1 轮样本，误差较大）  预计完成 {eta.strftime('%H:%M')}")


if __name__ == "__main__":
    main()
