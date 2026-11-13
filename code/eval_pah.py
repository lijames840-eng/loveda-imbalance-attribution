"""Evaluate V3 under the four prior modes, reusing ONE checkpoint.

None of these runs re-trains anything.  They differ only in which prior ``pi`` is
plugged into ``argmax_c [ log p_c - tau * log(pi_c) ]`` at inference:

+--------+-------------------------------------------+---------------------------------+
| mode   | pi                                        | what it isolates                |
+========+===========================================+=================================+
| none   | -- (no correction)                        | multi-task regularisation only  |
| pred   | PAH's own prediction for this tile        | the proposed model (V3)         |
| global | training-set class frequency (fixed)      | does per-tile matter? (V3-b)    |
| oracle | the tile's true composition               | upper bound of this family (V3-c)|
+--------+-------------------------------------------+---------------------------------+

Because the checkpoint, data order and seed are identical across the four columns,
any difference is attributable to the prior alone -- a stricter control than
re-training each variant, where seed noise can be the same size as the effect.

`none` is also the V3-d control: the model was trained with the auxiliary
composition loss but the correction is switched off, so its gap to `pred` measures
whether the gain comes from the correction or merely from the extra supervision.

Outputs ``experiments/<run>/pah_eval.json`` with a tau sweep per mode plus the
accuracy of pi_hat (mean absolute error against the true composition, gate: < 0.08).

Tau selection protocol (see report/stage0_research_plan_C.md §3.3)
------------------------------------------------------------------
Run this script twice: once with ``--split dev`` to pick tau* (the only legitimate
place a tau may be chosen), then with ``--split val`` to report that tau as the
headline, next to the pre-registered tau=1 column.  When ``--split val`` is used,
the ``best_tau`` field of the JSON is fitted **on Val** and is therefore a
cheat bound of the same nature as the E-delta ceiling -- report it as a diagnostic,
never as a result.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import load_model, predict_tile  # noqa: E402
from loveda_dataset import CLASSES, MEAN, STD, build_index  # noqa: E402
from metrics import ConfusionMatrix  # noqa: E402
from imgio import imread  # noqa: E402

MODES = ["none", "pred", "global", "oracle"]
# Fine grid near zero: V2's loss already removed most of the prior bias, so the
# optimum can sit below 0.25 -- and "the correction never helps" is only a defensible
# claim if the neighbourhood of tau=0 was actually searched.
TAUS = [0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]


def prep(ipath):
    img = imread(ipath, cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32) / 255.0
    return torch.from_numpy(((img - MEAN) / STD).transpose(2, 0, 1).copy())


def composition(gt):
    cnt = np.bincount(gt.reshape(-1), minlength=8)[1:].astype(np.float64)
    s = cnt.sum()
    return cnt / s if s > 0 else np.ones(7) / 7.0


def summarise(r):
    return dict(mIoU=float(r["mIoU"]), OA=float(r["OA"]), mF1=float(r["mF1"]),
                per_class={c: dict(IoU=float(r["IoU"][i]), Recall=float(r["Recall"][i]),
                                   Precision=float(r["Precision"][i]))
                           for i, c in enumerate(CLASSES)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="V3")
    ap.add_argument("--model", default="v3")
    ap.add_argument("--config", default="code/configs/v3.yaml")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--freq", default="data/stats/train_class_freq.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp = cfg["paths"]["experiments"]
    ckpt = args.ckpt or os.path.join(exp, args.run, "best.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")

    model = load_model(args.model, ckpt, device)
    items = build_index(cfg["paths"]["data"], args.split)
    if args.limit:
        items = items[:args.limit]
    print(f"[pah-eval] {len(items)} tiles from split={args.split}")

    fj = json.load(open(args.freq, encoding="utf-8"))
    pi_global = np.array([fj["freq"][c] for c in CLASSES], np.float64)
    pi_global = pi_global / pi_global.sum()
    print(f"[pah-eval] pi_global = {np.round(pi_global, 3).tolist()}")

    # one confusion matrix per (mode, tau); 'none' ignores tau
    cms = {(m, t): ConfusionMatrix() for m in MODES for t in TAUS}
    cm_none = ConfusionMatrix()
    per_domain = {m: {"urban": ConfusionMatrix(), "rural": ConfusionMatrix()} for m in MODES}

    mae_sum, mae_n, pi_rows = 0.0, 0, []

    for i, (ip, mp, domain) in enumerate(items, 1):
        prob, pi_pred = predict_tile(model, prep(ip), device,
                                     return_prob=True, return_prior=True)
        gt = imread(mp, cv2.IMREAD_GRAYSCALE)
        pi_gt = composition(gt)

        if pi_pred is not None:
            p_np = pi_pred.cpu().numpy().astype(np.float64)
            mae_sum += np.abs(p_np - pi_gt).sum()
            mae_n += 7
            if len(pi_rows) < 400:
                pi_rows.append(dict(tile=os.path.basename(ip)[:-4], domain=domain,
                                    pred=p_np.tolist(), gt=pi_gt.tolist()))

        logp = prob.clamp_min(1e-8).log()          # (7,H,W)

        pred_none = (logp.argmax(0) + 1).cpu().numpy()
        cm_none.update(pred_none, gt)
        per_domain["none"][domain].update(pred_none, gt)

        sources = {"pred": (pi_pred.cpu().numpy() if pi_pred is not None else None),
                   "global": pi_global,
                   "oracle": pi_gt}

        for mode in ("pred", "global", "oracle"):
            pi = sources[mode]
            if pi is None:
                continue
            logpi = torch.from_numpy(np.log(np.maximum(pi, 1e-8))).float().to(device)
            logpi = logpi.view(7, 1, 1)
            for tau in TAUS:
                pred = (logp - tau * logpi).argmax(0)
                pred = (pred + 1).cpu().numpy()
                cms[(mode, tau)].update(pred, gt)
                if abs(tau - 1.0) < 1e-9:
                    per_domain[mode][domain].update(pred, gt)

        if i % 100 == 0:
            print(f"  {i}/{len(items)}", flush=True)

    out = dict(run=args.run, split=args.split, n_tiles=len(items),
               tau_list=TAUS,
               prior_mae=(mae_sum / mae_n if mae_n else None),
               mode_none=summarise(cm_none.compute()),
               per_domain={m: {k: float(v.compute()["mIoU"]) for k, v in d.items()}
                           for m, d in per_domain.items()},
               sweep={})
    for mode in ("pred", "global", "oracle"):
        out["sweep"][mode] = {}
        for tau in TAUS:
            out["sweep"][mode][f"{tau:g}"] = summarise(cms[(mode, tau)].compute())

    # best tau per mode (report as a diagnostic; the headline number uses tau = 1)
    out["best_tau"] = {}
    for mode in ("pred", "global", "oracle"):
        rows = out["sweep"][mode]
        bt = max(rows, key=lambda k: rows[k]["mIoU"])
        out["best_tau"][mode] = dict(tau=float(bt), mIoU=rows[bt]["mIoU"])

    out["pi_samples"] = pi_rows
    # the split is part of the name on purpose: dev is where tau is chosen, val is
    # where it is reported, and the two JSONs must not overwrite each other
    path = os.path.join(exp, args.run, f"pah_eval_{args.split}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print("\n=== mIoU by mode (tau = 1) ===")
    print(f"{'none':>10s} {out['mode_none']['mIoU'] * 100:7.2f}")
    for mode in ("pred", "global", "oracle"):
        r = out["sweep"][mode]["1"]
        print(f"{mode:>10s} {r['mIoU'] * 100:7.2f}   (best tau {out['best_tau'][mode]['tau']:g}"
              f" -> {out['best_tau'][mode]['mIoU'] * 100:.2f})")
    if out["prior_mae"] is not None:
        print(f"\npi_hat MAE = {out['prior_mae']:.4f}   (gate < 0.08)")
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
