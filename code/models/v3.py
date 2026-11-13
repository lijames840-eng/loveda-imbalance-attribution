"""V3 = V1 + PAH (Prior-Aware Head).

Design (see report/stage0_research_plan_C.md §3.1, §5.4)
-------------------------------------------------------
Standard segmentation heads assume every class is equally likely a priori, then
`argmax` the logits.  LoveDA violates that assumption strongly and *per tile*: a
rural crop is mostly agriculture, an urban one mostly building/road, and a tile
averages only 5 of the 7 classes.  PAH makes the class prior explicit:

    pi_hat = softmax(MLP(GAP(bottleneck)))            # (B, 7), predicts this tile's composition
    z_c   <- z_c - tau * log(pi_hat_c)                # applied at inference, c = 1..7

This is an **inverse-prior (logit-adjustment) correction**, NOT the Bayes posterior.
The sign matters: writing the posterior would be ``z_c + log p(y=c)``, which
*reinforces* the dataset prior; subtracting it *removes* the prior, i.e. moves the
decision rule towards balanced error.  That is what the diagnosis calls for -- the
model already reproduces the training prior and therefore under-predicts rare
classes at test time, so the prior has to be taken out, not added back.
Because softmax is invariant to a per-sample constant, the correction only matters
through the *differences* between classes.

Why the bottleneck (/32) and not the decoder: a tile's composition is a global
property, so the features with the largest receptive field are the right place to
read it from.  GAP there also removes the spatial layout, which is what we want --
the head must predict *how much of each class*, not *where*.

Train/inference asymmetry, by design
------------------------------------
The main segmentation loss uses the **uncorrected** logits; the correction is only
applied at inference.  If the main loss also saw the corrected logits, the network
could learn to cancel pi_hat against the logits (a degenerate solution that looks
fine in training and destroys the prior's meaning).  The cost is a train/test
mismatch; `V3-d` (same model, no correction at inference) measures whether that
mismatch hurts.

Parameter cost: GAP(448) -> 128 -> 7 is ~58 k, i.e. +1.7 % over V1's 3.48 M.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .v1 import V1Net


class PAH(nn.Module):
    """Predicts the tile's class composition from globally pooled bottleneck features.

    Outputs 7 probabilities that sum to 1 (class indices 1..7 of the mask); the
    no-data channel is never part of a composition.
    """

    def __init__(self, cin, num_classes=8, hidden=128):
        super().__init__()
        self.num_classes = num_classes
        self.body = nn.Sequential(
            nn.Linear(cin, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_classes - 1),
        )

    def forward(self, feat):
        return torch.softmax(self.body(feat.mean((2, 3))), dim=1)


class V3Net(V1Net):
    def __init__(self, num_classes=8, widths=(64, 128, 256, 320, 448), depth=(2, 3, 3, 4),
                 prior_hidden=128):
        super().__init__(num_classes=num_classes, widths=widths, depth=depth)
        self.num_classes = num_classes
        self.pah = PAH(widths[-1], num_classes, prior_hidden)

    def forward(self, x, return_prior=True):
        h, w = x.shape[-2:]
        x1, x2, x3, x4 = self._features(x)
        logit = self._decode(x1, x2, x3, x4)
        logit = F.interpolate(logit, size=(h, w), mode="bilinear", align_corners=False)
        if not return_prior:
            return logit
        return logit, self.pah(x4)


def apply_prior_correction(logit, pi, tau=1.0):
    """logit_c <- logit_c - tau * log(pi_c) for c = 1..7; channel 0 (no-data) untouched.

    logit: (B, 8, H, W) or (8, H, W);  pi: (B, 7) or (7,).
    """
    if tau == 0:
        return logit
    squeeze = logit.dim() == 3
    if squeeze:
        logit = logit.unsqueeze(0)
        pi = pi.unsqueeze(0)
    corr = -tau * torch.log(pi.clamp_min(1e-6))
    out = logit.clone()
    out[:, 1:] = out[:, 1:] + corr[:, :, None, None]
    return out.squeeze(0) if squeeze else out


def class_composition(target, num_classes=8):
    """Ground-truth composition of a mask batch: (B, 7), no-data pixels excluded."""
    b = target.shape[0]
    out = torch.zeros(b, num_classes - 1, device=target.device, dtype=torch.float32)
    for i in range(b):
        cnt = torch.bincount(target[i].reshape(-1), minlength=num_classes)[1:].float()
        out[i] = cnt / cnt.sum().clamp(min=1)
    return out


def prior_loss(pi_hat, target):
    """Cross-entropy between the predicted composition and the mask's composition.

    Soft-label CE rather than KL: it differs from KL by the entropy of the target,
    which is a constant w.r.t. the parameters.
    """
    pi_gt = class_composition(target, pi_hat.shape[1] + 1)
    return -(pi_gt * torch.log(pi_hat.clamp_min(1e-8))).sum(1).mean()


if __name__ == "__main__":
    m = V3Net()
    n_all = sum(p.numel() for p in m.parameters())
    n_pah = sum(p.numel() for p in m.pah.parameters())
    x = torch.randn(2, 3, 512, 512)
    logit, pi = m(x)
    print(f"params={n_all / 1e6:.3f}M (PAH {n_pah / 1e3:.1f}k, +{n_pah / n_all * 100:.1f}%)")
    print(f"logit={tuple(logit.shape)}  pi={tuple(pi.shape)}  pi.sum={pi.sum(1).tolist()}")
    corr = apply_prior_correction(logit, pi, tau=1.0)
    print("correction changes logits:", not torch.allclose(logit, corr))
    print(f"prior_loss={prior_loss(pi, torch.randint(1, 8, (2, 64, 64))).item():.4f}")
