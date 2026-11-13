"""Baseline models for the comparison table.

unet_r34          : UNet + ImageNet ResNet-34 encoder      (~24 M params, CNN, classic)
deeplabv3p_r50    : DeepLabV3+ + ImageNet ResNet-50        (~40 M params, CNN, heavy)
segformer_b0      : MiT-B0 encoder + light head            (~3.5 M params, Transformer)

Initialisation is NOT uniform, and the paper must say so:

  E10 / E11  ImageNet-pretrained backbones (torchvision weights, industry convention)
  E12        trained from scratch -- this is a local re-implementation of MiT-B0 and
             loads no official weights. Treat it as a scratch model, same as V1/V5.
  V1, V5     from scratch

So the informative same-budget comparison is **E12 vs V1** (3.45 M vs 3.48 M, both
scratch, transformer vs depthwise-CNN). E10/E11 get a pretrained head start; if they
still lose to V1, that only strengthens the claim.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def _resnet(arch, pretrained=True):
    if arch == "resnet34":
        w = torchvision.models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        m = torchvision.models.resnet34(weights=w)
        ch = (64, 64, 128, 256, 512)
    else:
        w = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        m = torchvision.models.resnet50(weights=w)
        ch = (64, 256, 512, 1024, 2048)
    return m, ch


class UNet(nn.Module):
    """UNet with a ResNet encoder: skip connections taken at /2, /4, /8, /16, /32."""

    def __init__(self, num_classes=8, arch="resnet34", pretrained=True):
        super().__init__()
        enc, ch = _resnet(arch, pretrained)
        self.stem = nn.Sequential(enc.conv1, enc.bn1, enc.relu)      # /2, ch[0]
        self.pool = enc.maxpool
        self.l1, self.l2, self.l3, self.l4 = enc.layer1, enc.layer2, enc.layer3, enc.layer4

        def blk(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

        self.d4 = blk(ch[4] + ch[3], 256)
        self.d3 = blk(256 + ch[2], 128)
        self.d2 = blk(128 + ch[1], 64)
        self.d1 = blk(64 + ch[0], 64)
        self.out = nn.Sequential(blk(64, 64), nn.Conv2d(64, num_classes, 1))

    def forward(self, x):
        h, w = x.shape[-2:]
        s = self.stem(x)
        p = self.pool(s)
        c1 = self.l1(p)
        c2 = self.l2(c1)
        c3 = self.l3(c2)
        c4 = self.l4(c3)
        u = self.d4(torch.cat([c4, F.interpolate(c3, size=c4.shape[-2:], mode="bilinear",
                                                 align_corners=False)], 1))
        u = self.d3(torch.cat([u, c2], 1)) if u.shape[-2:] == c2.shape[-2:] else \
            self.d3(torch.cat([F.interpolate(u, size=c2.shape[-2:], mode="bilinear", align_corners=False), c2], 1))
        u = self.d2(torch.cat([F.interpolate(u, size=c1.shape[-2:], mode="bilinear", align_corners=False), c1], 1))
        u = self.d1(torch.cat([F.interpolate(u, size=s.shape[-2:], mode="bilinear", align_corners=False), s], 1))
        u = F.interpolate(self.out(u), size=(h, w), mode="bilinear", align_corners=False)
        return u


class DeepLabV3P(nn.Module):
    """torchvision DeepLabV3+ (ResNet-50 backbone), wrapped to return a single tensor."""

    def __init__(self, num_classes=8, pretrained=True):
        super().__init__()
        wb = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.net = torchvision.models.segmentation.deeplabv3_resnet50(
            weights=None, weights_backbone=wb, num_classes=num_classes, aux_loss=False)

    def forward(self, x):
        return self.net(x)["out"]


# --------------------------------------------------------------------------- #
# SegFormer-B0 (MiT-B0 encoder, Xie et al. 2021).  Compact re-implementation so
# that the comparison does not depend on an extra heavy dependency.
# --------------------------------------------------------------------------- #
class OverlapPatchEmbed(nn.Module):
    def __init__(self, cin, dim, k=7, s=4):
        super().__init__()
        self.proj = nn.Conv2d(cin, dim, k, s, k // 2)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.proj(x)
        h, w = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x), h, w


class EfficientAttention(nn.Module):
    def __init__(self, dim, heads, sr_ratio):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.sr = nn.Conv2d(dim, dim, sr_ratio, sr_ratio) if sr_ratio > 1 else None
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, h, w):
        b, n, c = x.shape
        if self.sr is not None:
            xr = x.transpose(1, 2).reshape(b, c, h, w)
            xr = self.norm(self.sr(xr).flatten(2).transpose(1, 2))
        else:
            xr = x
        q = self.q(x).reshape(b, n, self.heads, -1).permute(0, 2, 1, 3)
        kv = self.kv(xr).reshape(b, -1, 2, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(out)


class MixFFN(nn.Module):
    def __init__(self, dim, ratio=4):
        super().__init__()
        hid = int(dim * ratio)
        self.fc1 = nn.Linear(dim, hid)
        self.dw = nn.Conv2d(hid, hid, 3, 1, 1, groups=hid)
        self.fc2 = nn.Linear(hid, dim)

    def forward(self, x, h, w):
        b, n, _ = x.shape
        x = self.fc1(x)
        x = x.transpose(1, 2).reshape(b, -1, h, w)
        x = self.dw(x).flatten(2).transpose(1, 2)
        return self.fc2(F.gelu(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, sr_ratio):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = EfficientAttention(dim, heads, sr_ratio)
        self.n2 = nn.LayerNorm(dim)
        self.ffn = MixFFN(dim)

    def forward(self, x, h, w):
        x = x + self.attn(self.n1(x), h, w)
        return x + self.ffn(self.n2(x), h, w)


class SegFormerB0(nn.Module):
    """embed dims [32,64,160,256], depths [2,2,2,2], heads [1,2,5,8], sr [8,4,2,1]."""

    def __init__(self, num_classes=8, dims=(32, 64, 160, 256), depths=(2, 2, 2, 2),
                 heads=(1, 2, 5, 8), sr=(8, 4, 2, 1)):
        super().__init__()
        self.stages = nn.ModuleList()
        self.embeds = nn.ModuleList()
        cin = 3
        for i, d in enumerate(dims):
            k, s = (7, 4) if i == 0 else (3, 2)
            self.embeds.append(OverlapPatchEmbed(cin, d, k, s))
            self.stages.append(nn.ModuleList([TransformerBlock(d, heads[i], sr[i])
                                              for _ in range(depths[i])]))
            cin = d
        self.norms = nn.ModuleList([nn.LayerNorm(d) for d in dims])
        self.fuse = nn.Sequential(nn.Conv2d(sum(dims), 256, 1, bias=False),
                                  nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.head = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        h0, w0 = x.shape[-2:]
        outs, feat = [], x
        for emb, blocks, norm in zip(self.embeds, self.stages, self.norms):
            x, h, w = emb(feat)
            for blk in blocks:
                x = blk(x, h, w)
            x = norm(x)
            feat = x.transpose(1, 2).reshape(-1, x.shape[-1], h, w)
            outs.append(feat)
        target = outs[0].shape[-2:]
        feats = [F.interpolate(o, size=target, mode="bilinear", align_corners=False) for o in outs]
        y = self.fuse(torch.cat(feats, 1))
        return F.interpolate(self.head(y), size=(h0, w0), mode="bilinear", align_corners=False)


def build_baseline(name, num_classes=8):
    name = name.lower()
    if name == "unet_r34":
        return UNet(num_classes)
    if name == "deeplabv3p_r50":
        return DeepLabV3P(num_classes)
    if name == "segformer_b0":
        return SegFormerB0(num_classes)
    raise ValueError(name)


if __name__ == "__main__":
    for n in ("unet_r34", "deeplabv3p_r50", "segformer_b0"):
        m = build_baseline(n)
        p = sum(q.numel() for q in m.parameters()) / 1e6
        with torch.no_grad():
            y = m(torch.randn(1, 3, 512, 512))
        print(f"{n:16s} params={p:6.2f}M out={tuple(y.shape)}")
