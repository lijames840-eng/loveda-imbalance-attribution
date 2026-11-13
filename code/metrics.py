"""Segmentation metrics computed from a global confusion matrix.

All numbers reported in the paper come from here so that mIoU / OA / per-class
recall are consistent across models.
"""
import numpy as np

CLASSES = ["background", "building", "road", "water", "barren", "forest", "agriculture"]
NUM_CLASSES = 8  # includes the ignored no-data channel


class ConfusionMatrix:
    def __init__(self, num_classes=NUM_CLASSES, ignore=0):
        self.n = num_classes
        self.ignore = ignore
        self.mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred, target):
        """pred/target: (N,) int arrays of the same length."""
        pred = pred.reshape(-1)
        target = target.reshape(-1)
        keep = target != self.ignore
        pred, target = pred[keep], target[keep]
        idx = target * self.n + pred
        self.mat += np.bincount(idx, minlength=self.n * self.n).reshape(self.n, self.n)

    def compute(self):
        m = self.mat[1:, 1:]                     # drop the ignore row/col
        tp = np.diag(m).astype(np.float64)
        fp = m.sum(0) - tp
        fn = m.sum(1) - tp
        denom_iou = tp + fp + fn
        iou = np.where(denom_iou > 0, tp / np.maximum(denom_iou, 1), np.nan)
        rec = np.where((tp + fn) > 0, tp / np.maximum(tp + fn, 1), np.nan)
        pre = np.where((tp + fp) > 0, tp / np.maximum(tp + fp, 1), np.nan)
        f1 = np.where((2 * tp + fp + fn) > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), np.nan)
        oa = tp.sum() / max(m.sum(), 1)
        return dict(
            mIoU=float(np.nanmean(iou)), OA=float(oa), mF1=float(np.nanmean(f1)),
            IoU=iou, Recall=rec, Precision=pre, F1=f1,
            per_class={c: dict(IoU=float(iou[i]), Recall=float(rec[i]),
                               Precision=float(pre[i]), F1=float(f1[i]))
                       for i, c in enumerate(CLASSES)},
        )

    def table(self):
        r = self.compute()
        lines = [f"{'class':12s} {'IoU':>7s} {'Recall':>7s} {'Prec':>7s} {'F1':>7s}"]
        for i, c in enumerate(CLASSES):
            lines.append(f"{c:12s} {r['IoU'][i]*100:7.2f} {r['Recall'][i]*100:7.2f} "
                         f"{r['Precision'][i]*100:7.2f} {r['F1'][i]*100:7.2f}")
        lines.append(f"{'mIoU/OA/mF1':12s} {r['mIoU']*100:7.2f} {r['OA']*100:7.2f} {r['mF1']*100:7.2f}")
        return "\n".join(lines)


def sliding_window_predict(model, img, window=512, stride=256, num_classes=NUM_CLASSES, device="cuda"):
    """Full-tile inference with overlap averaging. img: (3,H,W) float tensor (normalised)."""
    import torch

    model.eval()
    _, h, w = img.shape
    prob = torch.zeros((num_classes, h, w), dtype=torch.float32, device=device)
    count = torch.zeros((1, h, w), dtype=torch.float32, device=device)
    ys = list(range(0, max(h - window, 0) + 1, stride)) or [0]
    xs = list(range(0, max(w - window, 0) + 1, stride)) or [0]
    if ys[-1] + window < h:
        ys.append(h - window)
    if xs[-1] + window < w:
        xs.append(w - window)
    with torch.no_grad():
        for y in ys:
            for x in xs:
                patch = img[:, y:y + window, x:x + window].unsqueeze(0).to(device)
                logit = model(patch)
                if isinstance(logit, (tuple, list)):
                    logit = logit[0]
                p = torch.softmax(logit.float(), dim=1)[0]
                prob[:, y:y + window, x:x + window] += p
                count[:, y:y + window, x:x + window] += 1.0
    return (prob / count.clamp(min=1)).argmax(0)
