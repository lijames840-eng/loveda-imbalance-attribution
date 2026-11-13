"""V1: self-designed lightweight encoder-decoder for LoveDA (from scratch).

Design intent (see report/stage0_research_plan_C.md):
  * parameter budget 3-5 M so that it sits next to SegFormer-B0 (3.7 M);
  * depthwise-separable residual blocks keep FLOPs low on a 6 GB laptop GPU;
  * output stride 4 (not 8/16) because 37.8% of road pixels are boundary pixels;
  * a light global context block at the bottleneck, since agriculture/background
    regions have a median connected-component area 8x larger than buildings.

V1 is deliberately "plain": no prior head, no boundary supervision. Those are the
interventions that V2-V4 add, and V1 must be a clean baseline for the ablation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DSConv(nn.Module):
    """Depthwise separable conv + BN + ReLU (cheap on both params and FLOPs)."""

    def __init__(self, cin, cout, stride=1, k=3):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, k, stride, k // 2, groups=cin, bias=False)
        self.pw = nn.Conv2d(cin, cout, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class ResBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = DSConv(cin, cout, stride)
        self.c2 = nn.Sequential(
            nn.Conv2d(cout, cout, 3, 1, 1, groups=cout, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 1, 1, 0, bias=False),
            nn.BatchNorm2d(cout))
        self.short = (nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                    nn.BatchNorm2d(cout))
                      if (stride != 1 or cin != cout) else nn.Identity())
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.c2(self.c1(x)) + self.short(x))


class GlobalContext(nn.Module):
    """Squeeze-excite style channel gate fed by global average pooling.
    Cheap substitute for self-attention when the whole tile is one land-cover type."""

    def __init__(self, c, r=4):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(c, c // r), nn.ReLU(inplace=True),
                                nn.Linear(c // r, c), nn.Sigmoid())

    def forward(self, x):
        w = self.fc(x.mean((2, 3))).unsqueeze(-1).unsqueeze(-1)
        return x * w


class ConvBNR(nn.Module):
    def __init__(self, cin, cout, k=3):
        super().__init__()
        self.b = nn.Sequential(nn.Conv2d(cin, cout, k, 1, k // 2, bias=False),
                               nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.b(x)


class V1Net(nn.Module):
    """Encoder strides: 2,4,8,16,32. Decoder fuses to stride 4 then upsamples x4."""

    def __init__(self, num_classes=8, widths=(64, 128, 256, 320, 448), depth=(2, 3, 3, 4)):
        super().__init__()
        w = widths
        self.stem = nn.Sequential(
            nn.Conv2d(3, w[0], 3, 2, 1, bias=False), nn.BatchNorm2d(w[0]), nn.ReLU(inplace=True),
            ResBlock(w[0], w[0]))                                   # /2
        self.s1 = nn.Sequential(*[ResBlock(w[0] if i == 0 else w[1], w[1], 2 if i == 0 else 1)
                                  for i in range(depth[0])])         # /4
        self.s2 = nn.Sequential(*[ResBlock(w[1] if i == 0 else w[2], w[2], 2 if i == 0 else 1)
                                  for i in range(depth[1])])         # /8
        self.s3 = nn.Sequential(*[ResBlock(w[2] if i == 0 else w[3], w[3], 2 if i == 0 else 1)
                                  for i in range(depth[2])])         # /16
        self.s4 = nn.Sequential(*[ResBlock(w[3] if i == 0 else w[4], w[4], 2 if i == 0 else 1)
                                  for i in range(depth[3])])         # /32
        self.gc = GlobalContext(w[4])

        self.lat4 = ConvBNR(w[4], 128, 1)
        self.lat3 = ConvBNR(w[3], 128, 1)
        self.lat2 = ConvBNR(w[2], 96, 1)
        self.lat1 = ConvBNR(w[1], 64, 1)

        self.fuse3 = ConvBNR(128, 128)
        self.proj3 = ConvBNR(128, 96, 1)
        self.fuse2 = ConvBNR(96, 96)
        self.proj2 = ConvBNR(96, 64, 1)
        self.fuse1 = ConvBNR(64, 64)
        self.head = nn.Sequential(ConvBNR(64, 64), nn.Conv2d(64, num_classes, 1))

    def _features(self, x):
        """Encoder: returns the four skip levels (strides 4, 8, 16, 32)."""
        x0 = self.stem(x)
        x1 = self.s1(x0)
        x2 = self.s2(x1)
        x3 = self.s3(x2)
        x4 = self.gc(self.s4(x3))
        return x1, x2, x3, x4

    def _decode(self, x1, x2, x3, x4):
        """Top-down FPN fusion down to stride 4 (no final upsample)."""
        p4 = self.lat4(x4)
        p3 = self.fuse3(self.lat3(x3) + F.interpolate(p4, size=x3.shape[-2:], mode="nearest"))
        p3 = self.proj3(p3)
        p2 = self.fuse2(self.lat2(x2) + F.interpolate(p3, size=x2.shape[-2:], mode="nearest"))
        p2 = self.proj2(p2)
        p1 = self.fuse1(self.lat1(x1) + F.interpolate(p2, size=x1.shape[-2:], mode="nearest"))
        return self.head(p1)

    def forward(self, x):
        h, w = x.shape[-2:]
        out = self._decode(*self._features(x))
        return F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)


if __name__ == "__main__":
    m = V1Net()
    n = sum(p.numel() for p in m.parameters()) / 1e6
    x = torch.randn(2, 3, 512, 512)
    y = m(x)
    print(f"params={n:.2f}M  out={tuple(y.shape)}")
