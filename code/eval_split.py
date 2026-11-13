"""Evaluate a checkpoint on an arbitrary split with the same full-tile protocol.

Control experiment: if a model scores much higher on `dev` (same distribution as
training) than on `val` (different region), the gap is domain shift, not capacity.
"""
import argparse
import json
import os
import sys

import cv2
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loveda_dataset import CLASSES, MEAN, STD, build_index  # noqa: E402
from metrics import ConfusionMatrix  # noqa: E402
from evaluate import load_model, predict_tile  # noqa: E402
from imgio import imread  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="V1_ce")
    ap.add_argument("--model", default="v1")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--config", default="code/configs/e0_ce.yaml")
    ap.add_argument("--ckpt", default="")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = args.ckpt or os.path.join(cfg["paths"]["experiments"], args.run, "best.pth")
    model = load_model(args.model, ckpt, device)
    items = build_index(cfg["paths"]["data"], args.split)

    cm = ConfusionMatrix()
    cm_dom = {"urban": ConfusionMatrix(), "rural": ConfusionMatrix()}
    for i, (ipath, mpath, dom) in enumerate(items, 1):
        img = imread(ipath, cv2.IMREAD_COLOR)[:, :, ::-1].astype("float32") / 255.0
        gt = imread(mpath, cv2.IMREAD_GRAYSCALE)
        x = torch.from_numpy(((img - MEAN) / STD).transpose(2, 0, 1).copy())
        pred = predict_tile(model, x, device).cpu().numpy()
        if pred.shape != gt.shape:
            pred = cv2.resize(pred.astype("uint8"), (gt.shape[1], gt.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        cm.update(pred, gt)
        if dom in cm_dom:
            cm_dom[dom].update(pred, gt)
        if i % 50 == 0:
            print(f"  {i}/{len(items)}", flush=True)

    r = cm.compute()
    print(f"\n[{args.run}] split={args.split}  mIoU={r['mIoU']*100:.2f}  OA={r['OA']*100:.2f}")
    for i, c in enumerate(CLASSES):
        print(f"  {c:12s} IoU={r['IoU'][i]*100:5.1f}  R={r['Recall'][i]*100:5.1f}  "
              f"P={r['Precision'][i]*100:5.1f}")

    # Persist the numbers: before this, dev metrics existed only in stdout, so clearing
    # logs would destroy them (Val already goes through evaluate.py, which does dump JSON).
    # Structure mirrors eval_{split}.json from evaluate.py so downstream tools read one schema.
    def _num(x):
        x = float(x)
        return None if x != x else x          # NaN -> null, keeps the file valid JSON

    res = dict(
        run=args.run, model=args.model, split=args.split, ckpt=ckpt, n_tiles=len(items),
        mIoU=_num(r["mIoU"]), OA=_num(r["OA"]), mF1=_num(r["mF1"]),
        per_class={c: {k: _num(v) for k, v in d.items()} for c, d in r["per_class"].items()},
        per_domain={k: _num(v.compute()["mIoU"]) for k, v in cm_dom.items()},
    )
    out = os.path.join(cfg["paths"]["experiments"], args.run, f"eval_{args.split}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
