# app/models/cnn1d.py
"""
Two-Branch 1D-CNN for transit classification.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch:int, out_ch:int, k:int, pool:int=2):
        super().__init__()
        pad = k // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=pad)
        self.bn   = nn.BatchNorm1d(out_ch)
        self.pool = nn.MaxPool1d(pool)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        x = self.pool(x)
        return x

class Branch(nn.Module):
    def __init__(self, in_ch:int, width:int, ks:list[int]):
        super().__init__()
        c1, c2 = width, width*2
        self.b1 = ConvBlock(in_ch, c1, ks[0])
        self.b2 = ConvBlock(c1,  c2, ks[1])
        self.b3 = ConvBlock(c2,  c2, ks[2])
        self.gap = nn.AdaptiveAvgPool1d(1)
    def forward(self, x):
        x = self.b1(x); x = self.b2(x); x = self.b3(x)
        x = self.gap(x)
        return torch.flatten(x, 1)

class TwoBranchCNN1D(nn.Module):
    def __init__(self, in_ch:int=1, width:int=32):
        super().__init__()
        self.g = Branch(in_ch, width, ks=[7,5,5])
        self.l = Branch(in_ch, width, ks=[5,3,3])
        self.fc1 = nn.Linear(width*4, 128)
        self.drop= nn.Dropout(0.30)
        self.fc2 = nn.Linear(128, 1)
    def forward(self, xg, xl):
        zg = self.g(xg)
        zl = self.l(xl)
        z  = torch.cat([zg, zl], dim=1)
        z  = F.relu(self.fc1(z), inplace=True)
        z  = self.drop(z)
        logits = self.fc2(z)
        return logits

def make_model() -> TwoBranchCNN1D:
    return TwoBranchCNN1D(in_ch=1, width=32)
