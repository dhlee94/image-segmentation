import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from torchsummary import summary as summary_
import os

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
    def __init__(self, dim, out_dim, drop_path=0., layer_scale_init_value=1e-6, down=False):
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
        self.down = down
        if down:
            self.down_layer = nn.Sequential(
                LayerNorm(dim, eps=1e-6, data_format="channels_first"),
                #nn.Conv2d(dim, out_dim, kernel_size=2, stride=2, padding=0)
                nn.Conv3d(dim, out_dim, kernel_size=(3, 2, 2), stride=(1, 2, 2), padding=0)
            )
        else:
            self.down_layer = nn.Sequential(
                LayerNorm(dim, eps=1e-6, data_format="channels_first"),
                #nn.Conv2d(dim, out_dim, kernel_size=1, stride=1, padding=0)
                nn.Conv3d(dim, out_dim, kernel_size=1, stride=1, padding=0)
            )
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
        if self.down:
            return self.down_layer(x)
        else:
            return self.down_layer(x)

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
            #x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Outlayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Outlayer, self).__init__()
        #self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        return self.conv(x)
            
class UNext(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, img_size=128, drop_path_rate=0.2, layer_scale_init_value=1e-6, layer_sizes=[3, 3, 9, 3], depths=[32, 64, 128, 256, 512]):
        super(UNext, self).__init__()
        self.classes = out_channels
        self.depth = len(depths)
        #Init Stem Block
        self.stem = nn.Sequential(
                                nn.Conv3d(in_channels, depths[0], kernel_size=1, stride=1),
                                LayerNorm(depths[0], eps=1e-6, data_format="channels_first")
                                )
        #Init Down Block
        self.down_layer = nn.ModuleList()
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(layer_sizes))]
        decoder_dp_rates=[x.item() for x in torch.linspace(drop_path_rate, 0, sum(layer_sizes))]
        cur = 0
        for idx in range(0, len(depths)-1):
            layers = [Block(dim=depths[idx], out_dim=depths[idx] if i != layer_sizes[idx]-1 else depths[idx+1], 
                            drop_path=dp_rates[cur + i], layer_scale_init_value=layer_scale_init_value,
                            down=False if i != layer_sizes[idx]-1 else True) for i in range(layer_sizes[idx])]
            cur += layer_sizes[idx]
            self.down_layer.append(nn.Sequential(*layers))
        #Init Up Block
        self.up_layer = nn.ModuleList()
        cur = 0
        for idx in range(len(depths)-1, 1, -1):
            layers = [Block(dim=depths[idx], out_dim=depths[idx] if i != layer_sizes[idx-1]-1 else depths[idx-1],
                            drop_path=decoder_dp_rates[cur + i], layer_scale_init_value=layer_scale_init_value, 
                            down=False) for i in range(layer_sizes[idx-1])]
            self.up_layer.append(nn.Sequential(*layers))
            cur += layer_sizes[idx-1]
        
        self.Up = nn.ModuleList()
        for idx in range(len(depths)-1, 0, -1):
            kernel = 2 if idx==1 else 1
            self.Up.append(nn.Sequential(
                LayerNorm(depths[idx], eps=1e-6,  data_format="channels_first"),
                nn.ConvTranspose3d(depths[idx], depths[idx-1], kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0)
                #nn.ConvTranspose2d(depths[idx], depths[idx-1], kernel_size=2, stride=2, padding=0)
            ))
        self.final_layer = Block(dim=depths[1], out_dim=depths[0],
                                 drop_path=0., layer_scale_init_value=layer_scale_init_value, down=False)
        self.outlayer = Outlayer(in_channels=depths[0], out_channels=self.classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.stem(x)
        shortcut_lists = []
        for layer in self.down_layer:
            shortcut_lists.append(x)
            x = layer(x)
        
        for idx, (up, layer) in enumerate(zip(self.Up[:-1], self.up_layer)): 
            x = up(x)
            pad = (0, 0, 0, 0, 1, 1)
            x = F.pad(x, pad, "constant", 0)
            x = torch.cat((x, shortcut_lists[-1-idx]), dim=1)
            x = layer(x)
        x = self.Up[-1](x)
        pad = (0, 0, 0, 0, 1, 1)
        x = F.pad(x, pad, "constant", 0)
        x = torch.cat((x, shortcut_lists[0]), dim=1)
        x = self.final_layer(x)
        
        x = self.outlayer(x)
        x = self.softmax(x)
        return x
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)
            
if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    model = UNext(in_channels=1)
    model.cuda()
    summary_(model, (1, 10, 128, 128), batch_size=1)