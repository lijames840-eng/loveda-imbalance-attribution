"""Params / FLOPs / FPS for any registered model (fills the efficiency table).

FLOPs are counted with forward hooks over Conv2d / Linear only (the dominant cost
for these networks); this is the same convention as thop for CNN-style models.
"""
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import build_model  # noqa: E402


def count_flops(model, size=512):
    total = {"conv": 0, "linear": 0}
    hooks = []

    def conv_hook(m, inp, out):
        o = out[0] if isinstance(out, (tuple, list)) else out
        k = m.kernel_size[0] * m.kernel_size[1]
        total["conv"] += int(o.shape[1] * o.shape[2] * o.shape[3] * (m.in_channels // m.groups) * k)

    def lin_hook(m, inp, out):
        total["linear"] += int(m.in_features * m.out_features)

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(lin_hook))
    model.eval()
    with torch.no_grad():
        model(torch.randn(1, 3, size, size))
    for h in hooks:
        h.remove()
    return (total["conv"] + total["linear"]) / 1e9      # GFLOPs (1 MAC = 1 FLOP here)


@torch.no_grad()
def bench(model, size=512, warmup=20, iters=50, device="cuda"):
    model = model.to(device).eval()
    x = torch.randn(1, 3, size, size, device=device)
    for _ in range(warmup):
        with torch.amp.autocast("cuda", enabled=device == "cuda"):
            model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    t = time.time()
    for _ in range(iters):
        with torch.amp.autocast("cuda", enabled=device == "cuda"):
            model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    ms = (time.time() - t) / iters * 1000
    return ms, 1000 / ms


def main():
    names = sys.argv[1:] or ["v1"]
    print(f"{'model':16s} {'params(M)':>10s} {'GFLOPs@512':>11s} {'ms/img':>8s} {'FPS':>7s}")
    for n in names:
        net = build_model(n, num_classes=8)
        p = sum(q.numel() for q in net.parameters()) / 1e6
        g = count_flops(net, 512)
        ms, fps = bench(net)
        print(f"{n:16s} {p:10.2f} {g:11.2f} {ms:8.1f} {fps:7.1f}")


if __name__ == "__main__":
    main()
