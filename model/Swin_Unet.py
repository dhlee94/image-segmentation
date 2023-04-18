import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#from .Swin_Parts import *
# import sys
# sys.path.append('./')
from models.Swin_Parts import *
# from .Swin_Parts import *
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.utils.checkpoint as checkpoint
import math

class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 attn_type='cos', drop_path=0., norm_layer=nn.LayerNorm, 
                 downsample=None, upsample=None, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                   num_heads=num_heads, window_size=window_size,
                                   shift_size=0 if (i % 2 == 0) else window_size // 2,
                                   mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                                   qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, 
                                   attn_type=attn_type, 
                                   drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                   norm_layer=norm_layer)
            for i in range(depth)
        ])

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
            self.upsample = None
        elif upsample is not None:
            self.upsample = upsample(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
            self.downsample = None
        else:
            self.downsample = None
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        elif self.downsample is not None:
            x = self.downsample(x)
        return x

class Upsample(nn.Sequential):
    """Upsample module.
    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.
    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.num_feat * 3 * 9
        return flops

class Swin_Unet(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 2, 2], depths_decoder=[2, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., attn_type='cos', drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", final_tanh=True, **kwargs):
        super().__init__()

        print("SwinTransformer_v2 expand initial----depths:{};depths_decoder:{};drop_path_rate:{};num_classes:{}".format(depths,
        depths_decoder,drop_path_rate,num_classes))

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample

        # split image into non-overlapping patches
        self.patch_embedding = PatchEmbedding(img_size=img_size, patch_size=patch_size,
                                               in_channels=in_chans, embed_dim=embed_dim,
                                               norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embedding.num_patches
        patches_resolution = self.patch_embedding.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build encoder and bottleneck layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(self.embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               attn_type=attn_type,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()

        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                      int(embed_dim * 2 ** (
                                                  self.num_layers - 1 - i_layer))) if i_layer > 0 else nn.Identity()
            if i_layer == 0:
                layer_up = PatchExpanding(
                    input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                      patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    dim_scale=2, norm_layer=norm_layer)
            else:
                layer_up = BasicLayer(dim=int(self.embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                      input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                                        patches_resolution[1] // ( 2 ** (self.num_layers - 1 - i_layer))),
                                      depth=depths_decoder[(self.num_layers - 1 - i_layer)],
                                      num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                                      window_size=window_size,
                                      mlp_ratio=self.mlp_ratio,
                                      qkv_bias=qkv_bias, qk_scale=qk_scale,
                                      drop=drop_rate, attn_drop=attn_drop_rate,
                                      drop_path=dpr[sum(depths_decoder[:(self.num_layers - 1 - i_layer)]):sum(
                                          depths_decoder[:(self.num_layers - 1 - i_layer) + 1])],
                                      norm_layer=norm_layer,
                                      upsample=PatchExpanding if (i_layer < self.num_layers - 1) else None,
                                      use_checkpoint=use_checkpoint)
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)
        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(embed_dim)

        
        if self.final_upsample == "expand_first":
            upscale = 4
            self.up = FinalPatchExpand_X4(
                input_resolution=(img_size // patch_size, img_size // patch_size), dim_scale=upscale,
                dim=embed_dim)
            self.out = nn.Conv2d(in_channels=embed_dim, out_channels=num_classes, kernel_size=1, bias=False)
        elif self.final_upsample == "pixel_shuffle":
            self.up_pixelshuffle = []
            upscale = 2
            self.up_patchexpanding = PatchExpanding_X2(input_resolution=(img_size // patch_size, img_size // patch_size),
                                                    dim=embed_dim, dim_scale=2, norm_layer=norm_layer)
            self.up_pixelshuffle = nn.Sequential(nn.Conv2d(embed_dim, (upscale ** 2) * embed_dim, 3, 1, 1),
                                                 nn.PixelShuffle(upscale))
            self.out = nn.Conv2d(in_channels=embed_dim, out_channels=num_classes, kernel_size=1, bias=False)
        
        self.final_tanh = final_tanh
        self.tanh = nn.Tanh()
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"cpb", "logit_scale", 'relative_position_bias_table'}


    def forward(self, x):
        if x.size()[1] == 1:
          x = x.repeat(1, 3, 1, 1)
          
        #Dencoder and Skip connection
        x = self.patch_embedding(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x_downsample = []
        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x)

        x = self.norm(x)  # B L C

        #Dencoder and Skip connection
        for idx, layer_up in enumerate(self.layers_up):
            if idx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x, x_downsample[3 - idx]], -1)
                x = self.concat_back_dim[idx](x)
                x = layer_up(x)
        x = self.norm_up(x)  # B L C


        H, W = self.patches_resolution
        B, L, C = x.shape

        assert L == H * W, "input features has wrong size"

        if self.final_upsample == "expand_first":
            x = self.up(x)
            x = x.view(B, 4 * H, 4 * W, -1)
            x = x.permute(0, 3, 1, 2)
            x = self.out(x)
        elif self.final_upsample == "pixel_shuffle":
            x = self.up_patchexpanding(x)
            x = x.view(B, 2 * H, 2 * W, -1)
            x = x.permute(0, 3, 1, 2)
            x = self.up_pixelshuffle(x)
            x = self.out(x)

        if self.final_tanh:
            x = self.tanh(x)

        return x

class SwinEncoder(nn.Module):
    def __init__(self, model):
        super(SwinEncoder, self).__init__()
        self.pos_embed = model.patch_embedding
        self.layers = model.layers

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        x = self.pos_embed(x)
        for layer in self.layers:
            x = layer(x)

        return x