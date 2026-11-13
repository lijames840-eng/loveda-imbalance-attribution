"""Loss functions for the ablation chain.

V1 / E0 : plain cross-entropy (dice_weight = 0, per_image = False)
V1 / E1 : CE + Dice
V2      : per-image adaptive class weighting + Dice   <- the "loss layer" intervention

Why per-image weights: LoveDA's imbalance is *local* (a tile averages only 5 of the
7 classes, and the dominant class flips between urban and rural). A single global
weight vector is therefore a compromise; the weights should follow each tile.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE = 0
NUM_CLASSES = 8


class CEDiceLoss(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dice_weight=0.5, per_image=False,
                 weight_clip=(0.5, 4.0), class_weights=None, boundary_weight=0.0,
                 share_scale=None):
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.per_image = per_image
        self.clip = weight_clip
        self.boundary_weight = boundary_weight
        if class_weights is not None:
            self.register_buffer("cw", torch.as_tensor(class_weights, dtype=torch.float32))
        else:
            self.cw = None
        # Per-class factors that restore dataset-level class shares; see
        # code/calibrate_share.py.  Without them the per-tile normalisation below
        # answers "how rare is this class in this scene" instead of "in the dataset",
        # which starved barren to 0.60x its global share (report/weighting_diagnosis.md).
        if share_scale is not None:
            self.register_buffer("share_scale", torch.as_tensor(share_scale, dtype=torch.float32))
        else:
            self.register_buffer("share_scale", torch.empty(0))

    # ------------------------------------------------------------------ #
    def _per_image_weights(self, target):
        """w[b, c] = sqrt(1 / freq_b(c)), mean-normalised over present classes.

        When ``share_scale`` is provided, each class's weight is additionally
        multiplied by a fixed factor so that its dataset-level share matches the
        global scheme.  The clip is applied *before* the scale on purpose: clipping
        afterwards would undo the alignment this factor exists to enforce.
        """
        b, h, w = target.shape
        out = torch.zeros_like(target, dtype=torch.float32)
        use_scale = self.share_scale.numel() == self.num_classes
        for i in range(b):
            t = target[i]
            cnt = torch.bincount(t.reshape(-1), minlength=self.num_classes).float()
            freq = cnt / cnt.sum().clamp(min=1)
            wgt = torch.zeros(self.num_classes, device=t.device)
            present = freq > 0
            wgt[present] = (1.0 / freq[present]).sqrt()
            wgt[present] = wgt[present] / wgt[present].mean()
            wgt[0] = 0.0                                    # ignore class 0
            if self.clip:
                wgt = wgt.clamp(self.clip[0], self.clip[1])
            if use_scale:
                wgt = wgt * self.share_scale.to(wgt.device)
            out[i] = wgt[t]
        return out

    def _boundary_map(self, target):
        """Per-pixel weight >1 on class boundaries; computed with max-pooling, no cv2."""
        t = target.unsqueeze(1).float()          # (B,1,H,W)
        mx = F.max_pool2d(t, 3, 1, 1)
        mn = -F.max_pool2d(-t, 3, 1, 1)
        edge = (mx != mn).float().squeeze(1)     # (B,H,W) -- must match the weight map
        return 1.0 + self.boundary_weight * edge

    # ------------------------------------------------------------------ #
    def forward(self, logit, target):
        logp = F.log_softmax(logit, dim=1)
        nll = -logp.gather(1, target.unsqueeze(1)).squeeze(1)     # (B,H,W)
        mask = target != IGNORE

        w = torch.ones_like(nll)
        if self.per_image:
            w = self._per_image_weights(target)
        elif self.cw is not None:
            w = self.cw.to(nll.device)[target]
        if self.boundary_weight > 0:
            w = w * self._boundary_map(target)

        ce = (nll * w)[mask].sum() / w[mask].sum().clamp(min=1e-6)

        if self.dice_weight <= 0:
            return ce

        prob = torch.softmax(logit, dim=1)
        dice, n = 0.0, 0
        for c in range(1, self.num_classes):
            gt = ((target == c) & mask).float()
            if gt.sum() == 0:
                continue
            pr = prob[:, c] * mask.float()
            inter = (pr * gt).sum()
            dice = dice + (2 * inter + 1.0) / (pr.sum() + gt.sum() + 1.0)
            n += 1
        dice = dice / max(n, 1)
        return (1 - self.dice_weight) * ce + self.dice_weight * (1 - dice)
