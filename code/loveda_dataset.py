"""LoveDA dataset / dataloader for the course project.

Official label mapping (rsidea.whu.edu.cn/LoveDA.htm):
    0 no-data (ignore) | 1 background | 2 building | 3 road
    4 water            | 5 barren     | 6 forest   | 7 agriculture

Tiles are 1024x1024; training crops 512x512 (a 6 GB laptop GPU cannot hold more).
"""
from paths import DATA_RAW, SPLITS  # noqa: E402
import json
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from imgio import imread  # noqa: E402

CLASSES = ["background", "building", "road", "water", "barren", "forest", "agriculture"]
CLASS_IDS = list(range(1, 8))          # 1..7 are the real classes
NUM_CLASSES = 8                        # 0..7, class 0 is ignored in the loss
IGNORE = 0
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_index(root, split, splits_json=str(SPLITS)):
    """split in {'train','dev','val'}; returns [(img_path, mask_path, domain)].

    'train'/'dev' come from the official Train set, partitioned by splits.json so that
    model selection never touches the official Val set.
    """
    if split in ("train", "dev"):
        with open(splits_json) as f:
            ids = set(json.load(f)[split])
        items = []
        for rel in sorted(ids):
            parts = rel.split("/")            # Train/<Domain>/<id>
            domain, tid = parts[1], parts[2]
            ip = os.path.join(root, "Train", domain, "images_png", tid + ".png")
            mp = os.path.join(root, "Train", domain, "masks_png", tid + ".png")
            if os.path.exists(ip) and os.path.exists(mp):
                items.append((ip, mp, domain.lower()))
        return items

    sub = "Val"
    items = []
    for domain in ("Urban", "Rural"):
        img_dir = os.path.join(root, sub, domain, "images_png")
        msk_dir = os.path.join(root, sub, domain, "masks_png")
        if not os.path.isdir(img_dir):
            continue
        for f in sorted(os.listdir(img_dir)):
            if not f.endswith(".png"):
                continue
            m = os.path.join(msk_dir, f)
            if os.path.exists(m):
                items.append((os.path.join(img_dir, f), m, domain.lower()))
    return items


class LoveDA(Dataset):
    def __init__(self, root, split="train", crop=512, train_mode=True, scale_range=(0.75, 1.5)):
        self.items = build_index(root, split)
        self.crop = crop
        self.train_mode = train_mode
        self.scale_range = scale_range
        if not self.items:
            raise FileNotFoundError(f"no tiles found under {root}/{split}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ipath, mpath, domain = self.items[i]
        img = imread(ipath, cv2.IMREAD_COLOR)[:, :, ::-1]          # BGR -> RGB
        msk = imread(mpath, cv2.IMREAD_GRAYSCALE)

        if self.train_mode:
            img, msk = self._augment(img, msk)
        else:
            # deterministic centre area for validation speed; full-tile eval is done
            # separately by sliding-window inference when we report final numbers
            h, w = msk.shape
            y, x = (h - self.crop) // 2, (w - self.crop) // 2
            if h > self.crop:
                img, msk = img[y:y + self.crop], msk[y:y + self.crop]
            if w > self.crop:
                img, msk = img[:, x:x + self.crop], msk[:, x:x + self.crop]

        img = np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32) / 255.0
        img = (img - MEAN[:, None, None]) / STD[:, None, None]
        return torch.from_numpy(img), torch.from_numpy(msk.astype(np.int64)), domain

    def _augment(self, img, msk):
        c = self.crop
        # random scale then crop: cheap multi-scale augmentation without extra memory
        s = random.uniform(*self.scale_range)
        if abs(s - 1.0) > 1e-3:
            h, w = msk.shape
            nh, nw = max(int(h * s), c), max(int(w * s), c)
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
            msk = cv2.resize(msk, (nw, nh), interpolation=cv2.INTER_NEAREST)
        h, w = msk.shape
        y = random.randint(0, h - c)
        x = random.randint(0, w - c)
        img, msk = img[y:y + c, x:x + c], msk[y:y + c, x:x + c]

        if random.random() < 0.5:
            img, msk = img[:, ::-1], msk[:, ::-1]
        if random.random() < 0.5:
            img, msk = img[::-1], msk[::-1]
        if random.random() < 0.5:
            k = random.randint(1, 3)
            img, msk = np.rot90(img, k), np.rot90(msk, k)

        if random.random() < 0.3:      # photometric jitter, applied after geometry
            img = img.astype(np.float32)
            img *= random.uniform(0.85, 1.15)
            img += random.uniform(-12, 12)
            img = np.clip(img, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(img), np.ascontiguousarray(msk)


if __name__ == "__main__":
    root = str(DATA_RAW)
    for split in ("train", "val"):
        ds = LoveDA(root, split, crop=512, train_mode=(split == "train"))
        x, y, d = ds[0]
        print(split, len(ds), tuple(x.shape), tuple(y.shape), d,
              "mask-values:", np.unique(y.numpy()).tolist())
