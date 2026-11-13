"""Self-test for fit_threshold.coordinate_ascent -- no model, no GPU, no data.

Builds synthetic probability maps whose true labels are known, then deliberately
suppresses one class's probability (a synthetic "decision threshold offset") and
checks that coordinate ascent recovers the injected bias.  This validates the
search logic independently of any trained checkpoint.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_threshold import confusion, coordinate_ascent, miou  # noqa: E402


def make_case(seed=0, n=60, h=64, w=64, suppressed=3, bias=-2.5, noise=0.9):
    rng = np.random.default_rng(seed)
    gt = rng.integers(1, 8, (n, h, w))                 # 1..7
    prob = rng.random((n, 7, h, w)).astype(np.float32) * noise
    for c in range(7):
        prob[:, c][gt == c + 1] += 1.0
    prob /= prob.sum(1, keepdims=True)

    delta = np.zeros(7, np.float32)
    delta[suppressed] = bias
    biased = prob * np.exp(delta)[None, :, None, None]
    biased /= biased.sum(1, keepdims=True)
    return biased, gt, delta


def flatten_local(prob, gt):
    p = prob.transpose(0, 2, 3, 1).reshape(-1, 7)
    p = np.log(np.maximum(p, 1e-8))
    t = gt.reshape(-1)
    keep = t > 0
    return np.ascontiguousarray(p[keep]), (t[keep] - 1).astype(np.int64)


def main():
    prob, gt, injected = make_case()
    flat, tc = flatten_local(prob, gt)
    buf = np.empty_like(flat)

    m0, _ = miou(confusion(flat, tc, np.zeros(7, np.float32), buf))
    delta, _, m1, _ = coordinate_ascent(flat, tc, buf, rounds=3)

    # delta is identifiable only up to a common shift, so compare after centring
    rec = delta - delta.mean()
    want = (-injected) - (-injected).mean()          # the bias that would undo it
    err = np.abs(rec - want).max()

    print(f"\nmIoU before {m0 * 100:.2f} -> after {m1 * 100:.2f}")
    print(f"recovered (centred): {np.round(rec, 2).tolist()}")
    print(f"expected  (centred): {np.round(want, 2).tolist()}")
    print(f"max abs error: {err:.2f}")

    assert m1 > m0, "search failed to improve mIoU"
    assert err <= 0.5, f"recovered bias deviates too much ({err:.2f})"
    print("\nSELFTEST PASSED")


if __name__ == "__main__":
    main()
