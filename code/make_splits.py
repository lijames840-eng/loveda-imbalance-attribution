"""Create a deterministic train/dev split from the official LoveDA Train set.

Why: thresholds and model selection must not touch the official Val set, otherwise
the reported number is a test-set-tuned number. We hold out 10% of Train as `dev`.

Output: data/splits.json  {"train": [...], "dev": [...]}  (relative paths, seed=42)
"""
from paths import DATA_RAW, SPLITS  # noqa: E402
import json
import os
import random

ROOT = str(DATA_RAW)
OUT = str(SPLITS)
DEV_RATIO = 0.10
SEED = 42


def main():
    train = []
    for domain in ("Urban", "Rural"):
        d = os.path.join(ROOT, "Train", domain, "images_png")
        for f in sorted(os.listdir(d)):
            if f.endswith(".png"):
                train.append(f"Train/{domain}/{f[:-4]}")

    # stratified by domain so both urban and rural tiles appear in dev
    rng = random.Random(SEED)
    dev, tr = [], []
    for domain in ("Urban", "Rural"):
        ids = [t for t in train if t.split("/")[1] == domain]
        rng.shuffle(ids)
        k = int(round(len(ids) * DEV_RATIO))
        dev += ids[:k]
        tr += ids[k:]
    dev.sort()
    tr.sort()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"train": tr, "dev": dev, "seed": SEED, "dev_ratio": DEV_RATIO}, f, indent=1)
    u = sum(1 for t in dev if t.split("/")[1] == "Urban")
    print(f"train={len(tr)}  dev={len(dev)} (urban={u}, rural={len(dev)-u}) -> {OUT}")


if __name__ == "__main__":
    main()
