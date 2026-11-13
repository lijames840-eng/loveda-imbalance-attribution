"""统一效率基准：所有模型交替测量，多轮取中位数。

为什么不能直接读 eval_val.json 里的 bench：那是各次评测结束时的热态读数，
V5 出现过 113.8 与复测 157 的差异。交替测量能把 GPU 频率状态的影响摊平。
"""
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import TABLES  # noqa: E402
from evaluate import flops  # noqa: E402
from models import build_model  # noqa: E402

MODELS = [("V1/V2", "v1"), ("V3", "v3"), ("V5", "v5"),
          ("E10 UNet-R34", "unet_r34"), ("E11 DeepLabV3+", "deeplabv3p_r50"),
          ("E12 SegFormer-B0", "segformer_b0")]
ROUNDS, WARM, ITER = 3, 30, 50


def bench_one(name, model_name, bs=1):
    m = build_model(model_name, num_classes=8).cuda().eval()
    x = torch.randn(bs, 3, 512, 512).cuda()
    # FP16 与 evaluate.benchmark 保持一致；不用 autocast 会慢近 3 倍（17.2 vs 6.0 ms）
    with torch.no_grad():
        for _ in range(WARM):
            with torch.amp.autocast("cuda"):
                m(x)
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(ITER):
            with torch.amp.autocast("cuda"):
                m(x)
        torch.cuda.synchronize()
    dt = (time.time() - t) / ITER
    del m
    torch.cuda.empty_cache()
    return dt


def gpu_clock():
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader"],
            text=True).strip()
        return out
    except Exception:
        return "?"


def warm_gpu(seconds=25):
    """笔记本 GPU 在短时突发下停在 1425 MHz（最大 2100），直接测会得到 3 倍偏慢的数。
    先持续负载把它拉到高频，再开始计时。"""
    m = build_model("deeplabv3p_r50", num_classes=8).cuda().eval()
    x = torch.randn(1, 3, 512, 512).cuda()
    t0 = time.time()
    with torch.no_grad():
        while time.time() - t0 < seconds:
            with torch.amp.autocast("cuda"):
                m(x)
    torch.cuda.synchronize()
    del m
    print(f"warm-up {seconds}s done, clock={gpu_clock()}")


def main():
    warm_gpu()
    samples = {n: [] for n, _ in MODELS}
    for r in range(ROUNDS):
        for label, mname in MODELS:
            samples[label].append(bench_one(label, mname, bs=1))
    # batch-1 单张延迟被 CPU launch overhead 主导（3.5M 模型的 GPU 时间只有几 ms），
    # 再加一个 batch-4 吞吐，才能看出 GPU 侧的真实能力差异。
    thr = {}
    for label, mname in MODELS:
        s = []
        for _ in range(2):
            s.append(bench_one(label, mname, bs=4))
        thr[label] = statistics.median(s)
    rows = []
    print(f"{'model':<18}{'params(M)':>10}{'GFLOPs':>9}{'ms@b1':>8}{'FPS':>8}{'img/s@b4':>10}")
    print(f"测量时频率: {gpu_clock()} (最大 2100 MHz)")
    for label, mname in MODELS:
        m = build_model(mname, num_classes=8)
        p = sum(q.numel() for q in m.parameters()) / 1e6
        del m
        med = statistics.median(samples[label])
        f = flops(mname)
        print(f"{label:<18}{p:10.3f}{f:9.2f}{med*1000:8.2f}{1/med:8.1f}{4/thr[label]:10.1f}")
        rows.append(dict(model=label, key=mname, params_M=p, flops_G=f,
                         ms_b1=round(med * 1000, 2), fps_b1=round(1 / med, 1),
                         imgs_per_s_b4=round(4 / thr[label], 1),
                         all_ms=[round(s * 1000, 2) for s in samples[label]]))
    json.dump(rows, open(str(TABLES / "efficiency.json"), "w"),
              ensure_ascii=False, indent=1)
    print("\nsaved -> report/tables/efficiency.json")


if __name__ == "__main__":
    main()
