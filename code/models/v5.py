"""V5 = V2 (相同损失) + 换掉瓶颈的上下文聚合模块。

设计依据（见 report/stage0_research_plan_C.md §2.7 / §3 V5）
------------------------------------------------------------
V1/V2 的瓶颈是一个 squeeze-excite 式门控：

    x4 = x * Sigmoid(FC(GAP(x)))          GAP -> FC(448->112) -> FC(112->448) -> Sigmoid

它把整幅图压成一个**通道级**权重，空间布局在 GAP 那一步就被抹掉了。而前置验证
（code/precheck_bg_error.py）显示：V1/V2 的背景误差有 68.5% / 69.6% 落在"距类别
边界 > 10 px 且所属同质块 > 500 px"的区域——即大片同质区**内部**的区域级误判，
不是边界毛刺。

V5 因此把"上下文怎么聚合"换掉：保留一支全局支路（GAP 不独占），并联三支空洞
depthwise 卷积（dilation 1/2/4），在**保留空间布局**的前提下给多尺度上下文，
再残差相加。

等参数设计
----------
去掉 SE（100.9 k）换成 ASPP-Lite（64.1 k），总参数 3.4842 M -> 3.4474 M（-1.06%）。
这是刻意的：若 V5 涨分，**不可能是"参数变多"**，容量混杂被设计直接排除。

单变量说明：严格说变的是"上下文聚合模块"整体（去 SE + 加多尺度 + 残差），不是
"加上下文"。所以结论只能写"换聚合方式有效/无效"，拆到具体成分需要 V5-ctl。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .v1 import V1Net


class ASPPLite(nn.Module):
    """三分支空洞 depthwise + 全局支路 -> 1x1 分组融合 -> 残差。

    在 /32 分辨率上操作（512 输入即 16x16），算力增量可忽略。
    """

    def __init__(self, c, dil=(1, 2, 4), groups=4):
        super().__init__()
        self.br = nn.ModuleList([
            nn.Conv2d(c, c, 3, padding=d, dilation=d, groups=c, bias=False) for d in dil
        ])
        self.bn1 = nn.BatchNorm2d(c)
        self.fuse = nn.Conv2d(c, c, 1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(c)

    def forward(self, x):
        n = len(self.br) + 1                      # +1 for the global branch
        h = sum(b(x) for b in self.br) / n
        h = h + x.mean((2, 3), keepdim=True) / n  # 全局支路：保留全局信息但不独占
        h = F.relu(self.bn1(h), inplace=True)
        return x + self.bn2(self.fuse(h))         # 残差


class V5Net(V1Net):
    """与 V1 同构，仅把 self.gc 换成 ASPPLite（参数量略降）。"""

    def __init__(self, num_classes=8, widths=(64, 128, 256, 320, 448), depth=(2, 3, 3, 4),
                 dil=(1, 2, 4), groups=4):
        super().__init__(num_classes=num_classes, widths=widths, depth=depth)
        self.gc = ASPPLite(widths[-1], dil=dil, groups=groups)


if __name__ == "__main__":
    m1, m5 = V1Net(), V5Net()
    n1 = sum(p.numel() for p in m1.parameters())
    n5 = sum(p.numel() for p in m5.parameters())
    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        y1, y5 = m1(x), m5(x)
    print(f"V1 {n1/1e6:.4f} M | V5 {n5/1e6:.4f} M ({100*(n5-n1)/n1:+.2f}%)")
    print(f"shape equal: {tuple(y1.shape) == tuple(y5.shape)}  {tuple(y5.shape)}")
