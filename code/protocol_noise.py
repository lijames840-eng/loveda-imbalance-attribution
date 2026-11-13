"""协议噪声估计：同一训练run、不同 checkpoint 选点，在 Val 上的差有多大？

动机
----
stage0 §6 有一条规则"单 seed，凡 |Δ| < 1.0 一律写无差异"，但 **1.0 这个数字从未被实测过**，
它只来自"dev 253 幅噪声约 ±1.5"的推断。而 H5 的判据 1 是 "Val ≥ V2 + 0.5"——
用 0.5 作判决阈值，却用 1.0 作"不可解读"的门槛，二者自相矛盾。

本脚本给一个**量级估计**（不是严格的种子方差，因为不重训）：
  同一 run 内，best.pth（按 dev 选的）与 last.pth（最后一轮）在 Val 上的 mIoU 差。
两点评测之间的差异来自"选点抖动"，与"换种子"是同一量级的协议噪声，
可以用来说明：Val 上多少点以内的差异不应当被解读。

注意：这是**下界式**的近似——它只含选点抖动，不含数据顺序/初始化带来的种子方差，
所以实测值若已经 >= 0.5，就足以证明判据 1 的 0.5 阈值偏紧。

用法：
  python code/protocol_noise.py                     # V1_ce 与 V2 的 last 都测
  python code/protocol_noise.py --runs V1_ce
"""
from paths import ROOT  # noqa: E402
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import evaluate  # noqa: E402

# repo root from code/paths.py
EXP = os.path.join(ROOT, "experiments")
DATA = os.path.join(ROOT, "data/raw/LoveDA")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["V1_ce", "V2"])
    ap.add_argument("--model", default="v1")
    ap.add_argument("--split", default="val")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {}
    for run in args.runs:
        rec = {}
        for tag, fname in (("best", "best.pth"), ("last", "last.pth")):
            ck = os.path.join(EXP, run, fname)
            if not os.path.exists(ck):
                print(f"[skip] {run}/{fname} 不存在")
                continue
            print(f"\n[eval] {run}/{fname} ...", flush=True)
            res, _ = evaluate(args.model, ck, args.split, DATA, device)
            rec[tag] = dict(mIoU=float(res["mIoU"]), OA=float(res["OA"]),
                            urban=float(res["per_domain"]["urban"]),
                            rural=float(res["per_domain"]["rural"]),
                            ckpt=ck)
            print(f"  {run}/{tag}: mIoU={res['mIoU']*100:.2f} OA={res['OA']*100:.2f}",
                  flush=True)
        if "best" in rec and "last" in rec:
            rec["delta_last_minus_best"] = rec["last"]["mIoU"] - rec["best"]["mIoU"]
            print(f"  → {run}: last − best = {rec['delta_last_minus_best']*100:+.2f}", flush=True)
        out[run] = rec

    deltas = [abs(v["delta_last_minus_best"]) for v in out.values() if "delta_last_minus_best" in v]
    if deltas:
        print("\n=== 协议噪声（选点抖动）===")
        for run, v in out.items():
            if "delta_last_minus_best" in v:
                print(f"  {run}: {v['delta_last_minus_best']*100:+.2f}")
        print(f"  平均绝对抖动 {sum(deltas)/len(deltas)*100:.2f} 点，最大 {max(deltas)*100:.2f} 点")
        out["summary"] = dict(mean_abs=sum(deltas) / len(deltas), max_abs=max(deltas))

    path = os.path.join(EXP, "protocol_noise.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
