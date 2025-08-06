import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple

class ASST_GCN(nn.Module):
    """自适应选择时空图卷积网络"""
    def __init__(self, inchan : int, outchan : int, kersize : int = 3):
        super().__init__()
        self.inchan = inchan # 输入通道数
        self.outchan = outchan # 输出通道数
        self.kersize = kersize # 卷积核大小

        self.gcn = nn.Conv2d(inchan, outchan, kernel_size = (1, 1))

        self.tcn1 = nn.Conv2d(
            outchan,
            outchan,
            kernel_size = (kersize, 1),
            padding = (kersize // 2, 0))
        self.tcn2 = nn.Conv2d(
            outchan,
            outchan,
            kernel_size = (kersize + 2, 1),
            padding = ((kersize + 2) // 2, 0))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(outchan, 2)
        self.bn = nn.BatchNorm2d(outchan)
        self.relu = nn.ReLU()

    def forward(self, x : Tensor) -> Tensor:
        x_gcn = self.gcn(x)
        x_gcn = self.bn(x_gcn)
        x_gcn = self.relu(x_gcn)

        x_tcn1 = self.tcn1(x_gcn)
        x_tcn2 = self.tcn2(x_gcn)

        b, c, _, _ = x_gcn.size()
        gap = self.gap(x_gcn).view(b, c)
        weights = F.softmax(self.fc(gap), dim = 1)

        x_tcn = weights[:, 0].view(b, 1, 1, 1) * x_tcn1 + weights[:, 1].view(b, 1, 1, 1) * x_tcn2

        return x_gcn + x_tcn

class CSMI(nn.Module):
    """跨尺度映射交互机制"""
    def __init__(self, chan : int):
        super().__init__()
        self.chan = chan

        self.atten1 = nn.Sequential(
            nn.Conv2d(chan, chan // 4, 1),
            nn.BatchNorm2d(chan // 4),
            nn.ReLU(),
            nn.Conv2d(chan // 4, chan, 1),
            nn.Sigmoid()
        )

        self.atten2 = nn.Sequential(
            nn.Conv2d(chan, chan // 4, 1),
            nn.BatchNorm2d(chan // 4),
            nn.ReLU(),
            nn.Conv2d(chan // 4, chan, 1),
            nn.Sigmoid()
        )

        self.fc = nn.Linear(chan * 2, 1)

    def forward(self, x : Tensor, y : Tensor) -> Tuple[Tensor, Tensor]:
        atten_x = self.atten1(x)
        x_atten = x * atten_x

        atten_y = self.atten2(y)
        y_atten = y * atten_y

        B, C, T, V = x.size()
        x_adj = x_atten.permute(0, 2, 3, 1).reshape(B * T * V, C)
        y_adj = y_atten.permute(0, 2, 3, 1).reshape(B * T * V, C)

        xy = torch.cat([x_adj, y_adj], dim = 1)
        adj = torch.sigmoid(self.fc(xy)).view(B, T, V, V)

        x_out = torch.einsum('btvu,bctu->bctv', adj, x_adj)
        y_out = torch.einsum('btuv,bctu->bctv', adj, y_adj)

        return x_out, y_out

class MSA_GCN(nn.Module):
    """多尺度自适应图卷积网络"""
    def __init__(self, class_num : int = 4, inchan : int = 3):
        super().__init__()
        self.class_num = class_num

        self.ini = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(inchan, 64, kernel_size = 1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            ) for i in range(3)
        ])

        self.asst_gcn = nn.ModuleList([
            ASST_GCN(64, 64),
            ASST_GCN(64, 64),
            ASST_GCN(64, 64)
        ])

        self.csmi = CSMI(64)

        self.fc = nn.Linear(64, class_num)

    def forward(self, x : Tensor) -> Tensor:
        mulscale = [conv(x.unsqueeze(-1)) for conv in self.ini]

        features = []
        for i, (scale, gcn) in enumerate(zip(mulscale, self.asst_gcn)):
            fea = gcn(scale)
            features.append(fea)

        x_sc1, x_sc2 = self.csmi(features[0], features[1])
        x_sc2, x_sc3 = self.csmi(x_sc2, features[2])

        fused = (x_sc1 + x_sc2 + x_sc3) / 3

        pool = F.adaptive_avg_pool2d(fused, (1, 1)).squeeze(-1).squeeze(-1)

        out = self.fc(pool)

        return out

