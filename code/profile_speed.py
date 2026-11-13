"""One-off profiling: where does the per-iteration time go (data vs model)?"""
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_RAW  # noqa: E402
from loveda_dataset import LoveDA  # noqa: E402
from models.v1 import V1Net  # noqa: E402
from train import CombinedLoss  # noqa: E402


def main():
    ds = LoveDA(str(DATA_RAW), "train", crop=512, train_mode=True)
    for nw in (4, 8):
        ld = DataLoader(ds, batch_size=4, shuffle=True, num_workers=nw, drop_last=True)
        it = iter(ld)
        for _ in range(3):
            next(it)
        t = time.time()
        for _ in range(15):
            next(it)
        print(f"data-only workers={nw}: {(time.time()-t)/15*1000:.0f} ms/batch", flush=True)

    m = V1Net().cuda()
    opt = torch.optim.AdamW(m.parameters(), 1e-4)
    crit = CombinedLoss().cuda()
    x = torch.randn(4, 3, 512, 512).cuda()
    y = torch.randint(0, 8, (4, 512, 512)).cuda()
    for _ in range(3):
        with torch.amp.autocast("cuda"):
            loss = crit(m(x), y)
        loss.backward()
        opt.step()
        opt.zero_grad()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(8):
        with torch.amp.autocast("cuda"):
            loss = crit(m(x), y)
        loss.backward()
        opt.step()
        opt.zero_grad()
    torch.cuda.synchronize()
    print(f"model fwd+bwd: {(time.time()-t)/8*1000:.0f} ms/iter")
    print("peak mem %.2f GB" % (torch.cuda.max_memory_allocated() / 1e9))


if __name__ == "__main__":
    main()
