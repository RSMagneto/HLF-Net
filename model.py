import torch.nn as nn
import torch
import numpy as np
from torch.nn import functional as F
import pdb
import scipy.io as scio
from torch.nn.parameter import Parameter
from unet_parts import *
from mmcv.cnn import ConvModule
from torch import Tensor
from torch.autograd import Function
from torch.nn.modules.utils import _pair
from string import Template
from collections import namedtuple
from timm.models.layers import DropPath, trunc_normal_

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class involution(nn.Module):
    def __init__(self,channels,kernel_size,stride):
        super(involution, self).__init__()
        self.kernel_size=kernel_size
        self.stride=stride
        self.channels=channels
        reduction_ratio=4
        self.group_channels=4
        self.groups=self.channels//self.group_channels
        self.conv1=ConvModule(
            in_channels=channels,
            out_channels=channels//reduction_ratio,
            kernel_size=1,
            conv_cfg=None,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU')
        )
        self.conv2 = ConvModule(
            in_channels=channels // reduction_ratio,
            out_channels=kernel_size**2 * self.groups,
            kernel_size=1,
            stride=1,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None
        )
        if stride > 1:
            self.avgpool=nn.AvgPool2d(stride,stride)
        self.unflod=nn.Unfold(kernel_size,1,(kernel_size-1)//2,stride)

    def forward(self, x):
        weight=self.conv2(self.conv1(x if self.stride==1 else self.avgpool(x)))
        b,c,h,w=weight.shape
        weight=weight.view(b,self.groups,self.kernel_size**2,h,w).unsqueeze(2)
        out=self.unflod(x).view(b,self.groups,self.group_channels,self.kernel_size**2,h,w)
        out=(weight*out).sum(dim=3).view(b,self.channels,h,w)
        return out

class poolingNet(nn.Module):
    def __init__(self,n_channels):
        super(poolingNet, self).__init__()
        # self.inc = DoubleConv(n_channels, 32)
        self.inc = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(n_channels, 32, kernel_size=3, padding=0),
            nn.ReLU(inplace=True)
        )
        self.involution=involution(channels=32,kernel_size=3,stride=1)
        # self.involution=nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1)
        self.outc = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, n_channels-1, kernel_size=3, padding=0),
            nn.ReLU(inplace=True)
        )
    def forward(self, pan,ms) :
        x=torch.cat([pan,ms],1)
        x1=self.inc(x)
        x4=self.involution(x1)
        x5=self.outc(x4+x1)
        return x5

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

class SkipConnection(nn.Module):
    def __init__(self,in_channels,num_convblocks,d_model):
        super(SkipConnection,self).__init__()
        self.skip_blocks = [DeformableTransformerDecoderLayer(d_model) for k in range(num_convblocks)]
        self.skip_path=nn.Sequential(*self.skip_blocks)

    def forward(self,x):
        return self.skip_path(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.skip1 = SkipConnection(in_channels=64,num_convblocks=2,d_model=128)
        self.down2 = Down(64, 128// factor)
        self.skip2 = SkipConnection(in_channels=32, num_convblocks=2,d_model=256)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = DoubleConv(32, n_classes)

    def forward(self, pan,ms):
        x = torch.cat([pan, ms], 1)
        x1 = self.inc(x) # torch.Size([16, 64, 256, 256])
        x2 = self.down1(x1) # torch.Size([16, 128, 128, 128])
        x3 = self.down2(x2) # torch.Size([16, 256, 64, 64])
        x = self.up3(x3, self.skip1(x2)) # torch.Size([16, 64, 128, 128])
        x = self.up4(x, self.skip2(x1)) # torch.Size([16, 64, 256, 256])
        logits = self.outc(x) # torch.Size([16, 4, 256, 256])
        return logits

class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=32, d_ffn=1024,
                 dropout=0.1,
                 n_heads=4):
        super().__init__()

        # cross attention
        self.cross_attn = Attention()
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.GLU()
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward_ffn(self, tgt):
        tgt2 = self.dropout2(self.linear2(self.dropout3(self.activation(self.linear1(tgt)))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, src_padding_mask=None):

        tgt2 = self.cross_attn(tgt)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt = self.forward_ffn(tgt)

        return tgt
