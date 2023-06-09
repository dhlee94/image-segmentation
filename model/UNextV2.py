import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import trunc_normal_
from random  import random
import os
from torchsummary import summary as summary_
class EfficientChannelAttention(nn.Module):
    def __init__(self, in_channels, k_size=3):
        super(EfficientChannelAttention, self).__init__()
        self.in_channel = in_channels
        # self.t = int(abs((log(self.in_channel, 2) + b) / gamma))
        # self.k = self.t if self.t % 2 else self.t + 1
        self.GAP = nn.AdaptiveAvgPool3d(1)
        self.Conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=((k_size - 1) // 2), bias=False)
        self.S = nn.Sigmoid()
    def forward(self, x):
        y = self.GAP(x)
        y = self.Conv(y.squeeze(-1).squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1).unsqueeze(-1)
        y = self.S(y)
        return x.mul(y.expand_as(x))

class Attentionblock(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels, depth_size=1):
        super(Attentionblock, self).__init__()
        self.block1 = nn.Sequential(
            LayerNorm(in_channels1, eps=1e-6, data_format="channels_first"),
            nn.GELU(),
            nn.Conv3d(in_channels1, out_channels, kernel_size=3, padding=1),           
            nn.MaxPool3d((depth_size, 2, 2))
        )
        self.block2 = nn.Sequential(
            LayerNorm(in_channels2, eps=1e-6, data_format="channels_first"),
            nn.GELU(),
            nn.Conv3d(in_channels2, out_channels, kernel_size=3, padding=1)
        )
        self.block3 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            LayerNorm(out_channels, eps=1e-6, data_format="channels_first"),
            nn.GELU(),            
        )

    def forward(self, x1, x2):
        x1 = self.block1(x1)
        x2 = self.block2(x2)
        x = torch.add(x1, x2)
        x = self.block3(x)
        x = x.mul(x2)
        return x

class PSPPooling(nn.Module):
    def __init__(self, in_channel, out_channel, up_mode="bilinear", filter_size=[1, 2, 4, 8]):
        super(PSPPooling, self).__init__()
        self.PSPblock = nn.ModuleList()
        self.filter_size = filter_size
        self.filter_depth = len(filter_size)
        for i in self.filter_size:
            self.PSPblock.append(
                nn.Sequential(
                    nn.MaxPool3d(kernel_size=(1, i, i), stride=(1, i, i)),
                    nn.Upsample(scale_factor=(1, i, i), mode=up_mode, align_corners=True),
                    nn.Conv3d(in_channel, in_channel // self.filter_depth, kernel_size=1, stride=1, padding=0),
                    LayerNorm(in_channel // self.filter_depth, eps=1e-6, data_format="channels_first")))
        self.out = nn.Sequential(
                    nn.Conv3d(2*in_channel, out_channel, kernel_size=1, stride=1, padding=0),
                    LayerNorm(out_channel, eps=1e-6, data_format="channels_first"))

    def forward(self, x):
        total_x = [x]
        for i in range(self.filter_depth):
            sub_x = self.PSPblock[i](x)
            total_x.append(sub_x)
        total_x = torch.cat(total_x, dim=1)
        x = self.out(total_x)
        return x

class Stemblock(nn.Module):
    def __init__(self, in_channels, out_channels, strides=1):
        super(Stemblock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, stride=strides),
            LayerNorm(out_channels, eps=1e-6, data_format="channels_first"),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0, stride=strides)
        self.ECA = EfficientChannelAttention(in_channels=out_channels)
    def forward(self, x):
        s = self.shortcut(x)
        x = torch.add(self.block(x), s)
        x = self.ECA(x)
        return x

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, out_dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        #self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=(1, 7, 7), stride=1, padding=(0, 3, 3), groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ECA = EfficientChannelAttention(in_channels=out_dim)
    def forward(self, x, h=None):
        input = x
        x = self.dwconv(x)
        #x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = x.permute(0, 2, 3, 4, 1) # (N, C, D, H, W) -> (N, D, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        #x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = x.permute(0, 4, 1, 2, 3) # (N, D, H, W, C) -> (N, C, D, H, W)
        x = input + self.drop_path(x)
        x = self.ECA(x)
        return x
    
class SegmentationTask(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SegmentationTask, self).__init__()
        self.layer = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        return self.layer(x)

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x

class UNext(nn.Module):
    def __init__(self, img_size=128, in_channels=1, out_channels=1, drop_path=0., layer_scale_init_value=1e-6, filtersize=[64, 128, 256, 512], up_mode="trilinear", check_sigmoid=False):
        super(UNext, self).__init__()
        self.channels = in_channels
        self.classes = out_channels
        self.check_sigmoid = check_sigmoid
        self.filtersize = filtersize
        self.depth = len(self.filtersize)
        self.Stem = Stemblock(self.channels, self.filtersize[0])
        #####################################################################################################################
        #Make Encoder Layer
        #by Resblock and Maxpooling
        #####################################################################################################################
        self.encoder = nn.ModuleList()
        for idx in range(self.depth - 1):
            kernel = 2 if idx==0 else 3
            stride = 2 if idx==0 else 1
            self.down_layer = nn.Sequential(
                LayerNorm(self.filtersize[idx], eps=1e-6, data_format="channels_first"),
                nn.Conv3d(self.filtersize[idx], self.filtersize[idx+1], kernel_size=(kernel, 2, 2), stride=(stride, 2, 2), padding=0)
            )
            layer = [Block(dim=filtersize[idx], out_dim=filtersize[idx+1],
                            drop_path=drop_path, layer_scale_init_value=layer_scale_init_value), 
                     self.down_layer]
            self.encoder.append(nn.Sequential(*layer))
        #####################################################################################################################
        #Make PSPPooling Block
        #After last encoder layer and Before last Decoder layer
        #####################################################################################################################
        self.Pspp1 = PSPPooling(self.filtersize[-1], self.filtersize[-1], up_mode=up_mode)
        self.Pspp2 = PSPPooling(self.filtersize[0], self.filtersize[0], up_mode=up_mode)
        #####################################################################################################################
        #Make Decoder Layer
        #by Attentionblock, Resblock and Upsampling
        #####################################################################################################################
        self.decoder = nn.ModuleList()
        for idx in range(self.depth-1, 0, -1):
            kernel = 2 if idx==1 else 1
            self.Up = nn.Sequential(
                LayerNorm(self.filtersize[idx], eps=1e-6,  data_format="channels_first"),
                nn.ConvTranspose3d(self.filtersize[idx], self.filtersize[idx-1], kernel_size=(kernel, 2, 2), stride=(kernel, 2, 2), padding=0)
                #nn.ConvTranspose2d(depths[idx], depths[idx-1], kernel_size=2, stride=2, padding=0)
            )
            layer = [Attentionblock(self.filtersize[idx-1], self.filtersize[idx], self.filtersize[idx], depth_size=1 if idx!=1 else 2), 
                     self.Up, 
                     Block((self.filtersize[idx]), self.filtersize[idx-1], drop_path=drop_path, layer_scale_init_value=layer_scale_init_value)]
            self.decoder.append(nn.Sequential(*layer))
        #####################################################################################################################
        #Make Out Layer
        #by SegmentationTask Layer and Sigmoid (if you didn't want to use sigmoid check_sigmoid=False)
        #####################################################################################################################
        self.segmentation_out = SegmentationTask(self.filtersize[0], classes)
        self.Sigmoid = nn.Sigmoid()
        self.apply(self._init_weights)
        
    def forward(self, x):
        ##Encoder
        x = self.Stem(x)
        x_downsample = []
        x_downsample.append(x)
        for idx, layer_down in enumerate(self.encoder):
            x = layer_down(x)
            if idx != len(self.encoder)-1:
                x_downsample.append(x)# x1,x2 ...
        x = self.Pspp1(x)
        ##Decoder
        for idx, layer_up in enumerate(self.decoder):
            if len(self.decoder)-1 > idx:
                pad = (0, 0, 0, 0, 1, 1)
                x = F.pad(x, pad, "constant", 0)
            x = layer_up[0](x_downsample[-1-idx], x)
            x = layer_up[1](x)
            x = torch.cat((x, x_downsample[-1-idx]), dim=1)
            x = layer_up[2](x)
            
        x = self.Pspp2(x)
        ##model output
        x = self.segmentation_out(x)
        if self.check_sigmoid:
            x = self.Sigmoid(x)        
        return x
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "8"
    model = ResUnetA()
    model.cuda()
    summary_(model, (1, 34, 128, 128), batch_size=1)