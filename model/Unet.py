from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F

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
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, act_layer=nn.GELU, change_size=True):
        super(DownBlock, self).__init__()
        self.change_size = change_size
        if hidden_channels==None:
            hideen_channels = out_channels
        if change_size:
            self.down = nn.Sequential(
                            LayerNorm(in_channels, eps=1e-6, data_format="channels_first"),
                            nn.Conv2d(in_channels, out_channels, kernel_size=(2, 2), stride=(2, 2), padding=0)
                        )
        self.layer = nn.Sequential(
                            LayerNorm(in_channels, eps=1e-6, data_format="channels_first"),                       
                            nn.Conv2d(in_channels=in_channels, out_channels=in_channels*2, kernel_size=1, stride=1, padding=0),
                            act_layer(),
                            LayerNorm(in_channels*2, eps=1e-6, data_format="channels_first"),
                            nn.Conv2d(in_channels=in_channels*2, out_channels=in_channels*2, kernel_size=3, stride=1, padding=1),
                            act_layer(),
                            LayerNorm(in_channels*2, eps=1e-6, data_format="channels_first"),                       
                            nn.Conv2d(in_channels=in_channels*2, out_channels=in_channels, kernel_size=1, stride=1, padding=0),
                            act_layer(),
                            )
        
    def forward(self, x):
        pre_x = self.layer(x)
        if self.change_size:
            x = self.down(pre_x)
            return pre_x, x
        return pre_x

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None,act_layer=nn.GELU, up_method='transpose'):
        super(UpBlock, self).__init__()
        self.in_channels = in_channels
        if hidden_channels==None:
            hidden_channels=out_channels
        self.layer = nn.Sequential(
                            LayerNorm(in_channels, eps=1e-6, data_format="channels_first"),
                            nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1),
                            act_layer(),
                            LayerNorm(hidden_channels, eps=1e-6, data_format="channels_first"),                       
                            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1),
                            act_layer(),
                            LayerNorm(hidden_channels, eps=1e-6, data_format="channels_first"),                       
                            nn.Conv2d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                            act_layer()
                            )
        if up_method=='bilinear':
            self.up = nn.Sequential(
                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                            nn.Conv2d(in_channels, in_channels//2, kernel_size=1, stride=1, padding=0)
                            )
        elif up_method=='transpose':
            self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=(2, 2), stride=(2, 2))
    def forward(self, pre_x, x):
        pre_x = self.up(pre_x)
        x = torch.cat([x, pre_x], dim=1)
        x = self.layer(x)  
        return x
     
class Outlayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Outlayer, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        return self.conv(x)
        
class Unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, img_size=128, depth=[32, 64, 128, 256, 512]):
        super(Unet, self).__init__()
        channels = in_channels
        self.classes = out_channels
        self.down_layer = nn.ModuleList()     
        for idx in range(len(depth)):
            if idx == len(depth)-1 :
                self.down_layer.append(DownBlock(in_channels=depth[idx], out_channels=depth[idx], change_size=False))
            else:
                self.down_layer.append(DownBlock(in_channels=depth[idx], out_channels=depth[idx+1]))
        self.up_layer = nn.ModuleList()
        for idx in range(len(depth)-1, 0, -1):
            self.up_layer.append(UpBlock(in_channels=depth[idx], out_channels=depth[idx-1]))
        self.outlayer = Outlayer(in_channels=depth[0], out_channels=self.classes)
        self.sigmoid = nn.Sigmoid()    
    def forward(self, x):
        x1, x = self.down_layer[0](x)
        x2, x  = self.down_layer[1](x)
        x3, x = self.down_layer[2](x)
        x4, x = self.down_layer[3](x)
        x = self.down_layer[4](x)
        x = self.up_layer[0](x, x4)
        x = self.up_layer[1](x, x3)
        x = self.up_layer[2](x, x2)
        x = self.up_layer[3](x, x1)
        output = self.outlayer(x)
        return output
    
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
        