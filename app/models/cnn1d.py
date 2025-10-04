import torch, torch.nn as nn, torch.nn.functional as F
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, pool=2):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k//2)
        self.bn   = nn.BatchNorm1d(out_ch)
        self.pool = nn.MaxPool1d(pool)
    def forward(self, x):
        return self.pool(F.relu(self.bn(self.conv(x)), inplace=True))
class Branch(nn.Module):
    def __init__(self, in_ch, width, ks):
        super().__init__()
        c1,c2 = width, width*2
        self.b1=ConvBlock(in_ch,c1,ks[0]); self.b2=ConvBlock(c1,c2,ks[1]); self.b3=ConvBlock(c2,c2,ks[2]); self.gap=nn.AdaptiveAvgPool1d(1)
    def forward(self,x): x=self.b1(x); x=self.b2(x); x=self.b3(x); x=self.gap(x); return torch.flatten(x,1)
class TwoBranchCNN1D(nn.Module):
    def __init__(self,in_ch=1,width=32):
        super().__init__()
        self.g=Branch(in_ch,width,[7,5,5]); self.l=Branch(in_ch,width,[5,3,3])
        self.fc1=nn.Linear(width*4,128); self.drop=nn.Dropout(0.3); self.fc2=nn.Linear(128,1)
    def forward(self,xg,xl):
        zg=self.g(xg); zl=self.l(xl); z=torch.cat([zg,zl],1); z=F.relu(self.fc1(z),inplace=True); z=self.drop(z); return self.fc2(z)
def make_model(): return TwoBranchCNN1D(1,32)
