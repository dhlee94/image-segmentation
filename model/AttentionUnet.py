import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RRCNN_Block(nn.Module):
    def __init__(self, input_channel, output_channel, act_layer=nn.ReLU, depth=2, t=2):
        super(RRCNN_Block, self).__init__()
        self.act_layer = act_layer(inplace=True)
        self.t = t
        self.channel_conv = nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=1, padding=0)
        self.module_list = nn.ModuleList()
        self.depth = depth
        for _ in range(self.depth):
            conv = nn.Sequential(
                    nn.Conv2d(output_channel, output_channel, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.BatchNorm2d(output_channel),
                    self.act_layer)
            self.module_list.append(conv)
    def forward(self, x):
        x = self.channel_conv(x)
        for idx in range(self.depth):
            for idx2 in range(self.t):
                if idx2 == 0:
                    new_x = self.module_list[idx](x)
                new_x = self.module_list[idx](x+new_x)
            x = new_x
        return x

class AttentionBlock(nn.Module):
    def __init__(self, channel_g, channel_x, hidden_channel):
        super(AttentionBlock, self).__init__()
        self.w_g = nn.Sequential(
            nn.Conv2d(channel_g, hidden_channel, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(hidden_channel)
            )
        self.w_x = nn.Sequential(
            nn.Conv2d(channel_x, hidden_channel, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(hidden_channel)
            )
        self.psi = nn.Sequential(
            nn.Conv2d(hidden_channel, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
            )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        h_g = self.w_g(g)
        h_x = self.w_x(x)
        psi = self.relu(h_g + h_x)
        psi = self.psi(psi)        
        return x*psi

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, act_layer=nn.ReLU):
        super(UpConv, self).__init__()
        self.act_layer = act_layer(inplace=True)
        self.up_conv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            self.act_layer
        )
    def forward(self, x):
        x = self.up_conv(x)
        return x

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        return self.conv(x)

class Decoder(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(Decoder, self).__init__()
        self.upconv = UpConv(in_channels=input_channel, out_channels=output_channel)
        self.attention = AttentionBlock(channel_g=output_channel, channel_x=output_channel, hidden_channel=output_channel//2)
        self.rrcnn = RRCNN_Block(input_channel=input_channel, output_channel=output_channel, t=2)
    
    def forward(self, x, previous):
        x = self.upconv(x)
        previous = self.attention(x, previous)
        x = torch.cat((previous, x), dim=1)
        x = self.rrcnn(x)
        return x
    
class AttentionUnet(nn.Module):
    def __init__(self, input_channel=3, classes=1, t=2, check_sigmoid=False):
        super(AttentionUnet, self).__init__()
        self.downsampling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.classes = classes
        self.filter_size = [64, 128, 256, 512, 1024]
        self.down_list = nn.ModuleList()
        self.check_sigmoid = check_sigmoid
        for idx in range(len(self.filter_size)):
            if idx == 0:
                tmp = RRCNN_Block(input_channel, self.filter_size[idx], t=2)
            else:
                tmp = RRCNN_Block(self.filter_size[idx-1], self.filter_size[idx], t=2)
            self.down_list.append(tmp)
        self.attention_list = nn.ModuleList()
        for idx in range(len(self.filter_size)-1):
            self.attention_list.append(Decoder(input_channel=self.filter_size[-(1+idx)], output_channel=self.filter_size[-(2+idx)]))
        self.Output = OutConv(self.filter_size[0], self.classes)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        down_x = []
        for idx, layer in enumerate(self.down_list):
            x = layer(x)
            if idx != 0:
                x = self.downsampling(x)
            if idx != len(self.filter_size)-1:
                down_x.append(x)
        for idx, layer in enumerate(self.attention_list):
            x = layer(x, down_x[-(idx+1)])
        x = self.Output(x)
        x = self.softmax(x)
        return x