""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.conv=nn.Conv2d(in_channels=1,out_channels=1,kernel_size=1,padding=0)
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        b,c,w,h=x.shape
        x_reshape=x.reshape(-1,1,h,w)
        xn=self.avg_pool(x_reshape)
        xn_=self.conv(xn)
        out=x_reshape*self.sigmoid(xn_)
        return out.reshape(b,-1,w,h)

class Mlp(nn.Module):
    def __init__(self,in_featuers, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        self.fc1=nn.Conv2d(in_channels=in_featuers, out_channels=hidden_features,kernel_size=1)
        self.act=act_layer()
        self.fc2 = nn.Conv2d(in_channels=hidden_features, out_channels=out_features, kernel_size=1)
        self.drop=nn.Dropout(drop)
        self.apply(self._init_weight)

    def _init_weight(self,m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        x=self.fc1(x)
        x=self.act(x)
        x=self.drop(x)
        x=self.fc2(x)
        x=self.drop(x)
        return x

class CABlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,padding=1,drop_path=0.,layer_scale_init_value=1e-5):
        super(CABlock,self).__init__()
        self.norm1=nn.BatchNorm2d(in_channels)
        self.norm2 = nn.BatchNorm2d(in_channels)
        self.lRelu=nn.LeakyReLU(negative_slope=0.2)
        self.drop_path = DropPath(0.) if drop_path > 0. \
            else nn.Identity()
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value*torch.ones((out_channels)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((out_channels)), requires_grad=True)
        self.token_mixer = Attention()
        self.mlp= Mlp(in_channels,in_channels//2,out_channels)

    def forward(self,x):
        x = x+self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)*self.token_mixer(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True)
        )
        self.CA=CABlock(in_channels=out_channels,out_channels=out_channels)

    def forward(self, x):
        x_conv=self.double_conv(x)
        x_out=self.CA(x_conv)
        return x_out


class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=2,stride=2),
            DoubleConv(out_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
