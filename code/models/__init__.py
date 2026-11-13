"""Model registry. Add new architectures here so train/evaluate stay generic."""
from .v1 import V1Net


def build_model(name, num_classes=8, **kw):
    name = name.lower()
    if name in ("v1", "v1net", "ours_v1"):
        return V1Net(num_classes=num_classes)
    if name in ("v3", "v3net", "ours_v3"):
        from .v3 import V3Net
        return V3Net(num_classes=num_classes, **kw)
    if name in ("v5", "v5net", "ours_v5"):
        from .v5 import V5Net
        return V5Net(num_classes=num_classes, **kw)
    if name in ("unet_r34", "deeplabv3p_r50", "segformer_b0"):
        from .baselines import build_baseline
        return build_baseline(name, num_classes=num_classes)
    raise ValueError(f"unknown model: {name}")
