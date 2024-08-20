import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from natten import NeighborhoodAttention2D as NeighborhoodAttention
torch.backends.cudnn.enabled = False
class RLN(nn.Module):
    r"""Revised LayerNorm"""

    def __init__(self, dim, eps=1e-5, detach_grad=False):
        super(RLN, self).__init__()
        self.eps = eps
        self.detach_grad = detach_grad
        # 对应文章的γ和β
        self.weight = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.bias = nn.Parameter(torch.zeros((1, dim, 1, 1)))

        self.meta1 = nn.Conv2d(1, dim, 1)
        self.meta2 = nn.Conv2d(1, dim, 1)

        # 初始化
        trunc_normal_(self.meta1.weight, std=.02)
        nn.init.constant_(self.meta1.bias, 1)

        trunc_normal_(self.meta2.weight, std=.02)
        nn.init.constant_(self.meta2.bias, 0)

    def forward(self, input):
        mean = torch.mean(input, dim=(1, 2, 3), keepdim=True)
        std = torch.sqrt((input - mean).pow(2).mean(dim=(1, 2, 3), keepdim=True) + self.eps)

        normalized_input = (input - mean) / std

        if self.detach_grad:
            rescale, rebias = self.meta1(std.detach()), self.meta2(mean.detach())
        else:
            rescale, rebias = self.meta1(std), self.meta2(mean)

        out = normalized_input * self.weight + self.bias
        return out, rescale, rebias


class Mlp(nn.Module):
    def __init__(self, network_depth, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.network_depth = network_depth

        self.mlp = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden_features, out_features, 1)
        )

        self.apply(self._init_weights)

    # 初始化权重和偏置项 Xavier/Glorot 初始化
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            gain = (8 * self.network_depth) ** (-1 / 4)
            fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.mlp(x)




def get_relative_positions(window_size):
    coords_h = torch.arange(window_size)
    coords_w = torch.arange(window_size)

    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_positions = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww

    relative_positions = relative_positions.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    # 计算相对位置的绝对值的对数，同时保留其符号。
    relative_positions_log = torch.sign(relative_positions) * torch.log(1. + relative_positions.abs())

    return relative_positions_log

class ConvTokenizer(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(
                in_chans,
                embed_dim // 4,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            ),
            # nn.Conv2d(
            #     embed_dim // 2,
            #     embed_dim,
            #     kernel_size=(3, 3),
            #     stride=(2, 2),
            #     padding=(1, 1),
            # ),
        )
        # if norm_layer is not None:
        #     self.norm = norm_layer(embed_dim)
        # else:
        #     self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        # if self.norm is not None:
        #     x = self.norm(x)
        return x
import torch.nn as nn

class ReverseConvTokenizer(nn.Module):
    def __init__(self, in_chans=24, out_chans=4, embed_dim=96, norm_layer=None, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        self.proj = nn.Sequential(
            nn.Conv2d(
                in_chans,
                embed_dim,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            ),
            # nn.PixelShuffle(scale_factor)
        )
    def forward(self, x):
        x = self.proj(x)
        return x.permute(0, 2, 3, 1)

class ConvDownsampler(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.reduction = nn.Conv2d(
            dim, 2 * dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )
        # self.norm = norm_layer(2 * dim)

    def forward(self, x):
        x = self.reduction(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        # x = self.norm(x)
        return x # BHWC


class ConvUpsampler(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.expansion = nn.ConvTranspose2d(
            dim, dim // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1), bias=False
        )
        self.norm = norm_layer(dim // 2)

    def forward(self, x):
        x = self.expansion(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = self.norm(x)
        return x

class NATLayer(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        network_depth,
        kernel_size=7,
        dilation=None,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        # drop=0.0,
        # attn_drop=0.0,
        # drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        layer_scale=None,
        conv_type=None
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = NeighborhoodAttention(
            dim,
            kernel_size=kernel_size,
            dilation=dilation,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            # attn_drop=attn_drop,
            # proj_drop=drop,
        )

        # if self.conv_type == 'Conv':
        #     self.conv = nn.Sequential(
        #         nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect'),
        #         nn.ReLU(True),
        #         nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect')
        #     )
        #
        # if self.conv_type == 'DWConv':
        #     self.conv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, padding_mode='reflect')
        #     self.proj = nn.Conv2d(dim, dim, 1)
        # self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            network_depth,
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),

            # act_layer=act_layer,
            # drop=drop,
        )
        self.layer_scale = False
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(
                layer_scale * torch.ones(dim), requires_grad=True
            )
            self.gamma2 = nn.Parameter(
                layer_scale * torch.ones(dim), requires_grad=True
            )

    def forward(self, x):
        if not self.layer_scale:
            shortcut = x
            # x = self.norm1(x)
            x, rescale, rebias = self.norm1(x)
            x = self.attn(x)
            x = shortcut + self.drop_path(x)+ x
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        x = x.permute(0, 3, 1, 2) # B C H W
        shortcut = x
        x, rescale, rebias = self.norm1(x)
        # x = self.norm1(x)
        x = x.permute(0, 2, 3, 1) # B H W C
        x = self.attn(x)
        x = x.permute(0, 3, 1, 2) #B C H W
        x = x * rescale + rebias
        # self.gamma1 = self.gamma1.permute(1,0,2,3)
        # assert shortcut.shape[3] == self.drop_path(self.gamma1 * x).shape[3] == x.shape[3], "Dimensions don't match"
        # x = shortcut + self.drop_path(self.gamma1 * x)+ x
        # x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        x = shortcut + x
        x, rescale, rebias = self.norm2(x)
        x = self.mlp(x)
        x = x * rescale + rebias
        x = shortcut + x
        x= x.permute(0, 2, 3, 1)
        return x
class NATBlock(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        network_depth,
        num_heads,
        kernel_size,
        dilations=None,
        downsample=True,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        # drop=0.0,
        # attn_drop=0.0,
        # drop_path=0.0,
        norm_layer=nn.LayerNorm,
        layer_scale=None,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth

        self.blocks = nn.ModuleList(
            [
                NATLayer(
                    dim=dim,
                    num_heads=num_heads,
                    kernel_size=kernel_size,
                    dilation=None if dilations is None else dilations[i],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    # drop=drop,
                    # attn_drop=attn_drop,
                    # drop_path=drop_path[i]
                    # if isinstance(drop_path, list)
                    # else drop_path,
                    norm_layer=norm_layer,
                    layer_scale=layer_scale,
                    network_depth=network_depth
                )
                for i in range(depth)
            ]
        )

        self.downsample = (
            None if not downsample else ConvDownsampler(dim=dim, norm_layer=norm_layer)
        )

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
            # x B H W C
        if self.downsample is None:
            return x
        return self.downsample(x), x
        # return x



# class PatchUnEmbed(nn.Module):
#     def __init__(self, out_chans=3, embed_dim=96, kernel_size=None):
#         super().__init__()
#         self.out_chans = out_chans
#         self.embed_dim = embed_dim
#
#         # if kernel_size is None:
#         #     kernel_size = 1
#
#         self.proj = nn.Sequential(
#             nn.Conv2d(
#                 embed_dim * 4,
#                 out_chans,
#                 kernel_size=(3, 3),
#                 stride=(1, 1),
#                 padding=(1, 1),
#             )
#
#         )
#
#     def forward(self, x):
#         x = self.proj(x)
#         return x

class PatchUnEmbed(nn.Module):
    def __init__(self, scale_factor=2, out_chans=3, embed_dim=96, kernel_size=3):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans, kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode='reflect'),
            # nn.PixelShuffle(scale_factor)
        )

    def forward(self, x):
        x = self.proj(x)
        return x
class SKFusion(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(SKFusion, self).__init__()

        self.height = height
        d = max(int(dim / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim * height, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape

        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)

        # attn根据跨通道维度的空间信息计算注意力权重 ( )，并且原始输入 ( in_feats) 通过这些注意力权重进行加权以产生最终输出 ( out)
        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        out = torch.sum(in_feats * attn, dim=1)
        return out


class DehazeFormer(nn.Module):
    def __init__(self, in_chans=3, out_chans=4, kernel_size=8,
                 embed_dims=[24, 48, 96, 48, 24],
                 mlp_ratios=[2., 4., 4., 2., 2.],
                 depths=[16, 16, 16, 8, 8],
                 num_heads=[2, 4, 6, 1, 1],
                 # attn_ratio=[1 / 4, 1 / 2, 3 / 4, 0, 0],
                 # conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'],
                 norm_layer=[RLN, RLN, RLN, RLN, RLN],
                 layer_scale=None,
                 dilations=None,
                 # num_classes=1000,
                 qkv_bias=True,
                 qk_scale=None,
                 # attn_drop_rate=0.0,
                 # drop_rate=0.0,
                 # drop_path_rate=0.2
                 ):
        super(DehazeFormer, self).__init__()

        # setting
        # self.img_size = 256

        self.kernel_size = kernel_size
        self.mlp_ratios = mlp_ratios

        # split image into overlapping patches
        self.patch_embed = ConvTokenizer(
            in_chans=in_chans, norm_layer=nn.LayerNorm
        )

        # self.pos_drop = nn.Dropout(p=drop_rate)
        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]


        # backbone
        self.layer1 = NATBlock( dim=embed_dims[0],depth=depths[0],num_heads=num_heads[0],kernel_size=kernel_size,network_depth=sum(depths),
                                dilations=None if dilations is None else dilations[0],mlp_ratio=self.mlp_ratios[0],
                                qkv_bias=qkv_bias,qk_scale=qk_scale,
                                # drop_path=dpr[sum(depths[:0]) : sum(depths[:1])],drop=drop_rate,attn_drop=attn_drop_rate,
                                norm_layer=norm_layer[0],downsample=True,layer_scale=layer_scale,
                                )
                                 # dim=embed_dims[0], depth=depths[0],
                                 # num_heads=num_heads[0], mlp_ratio=mlp_ratios[0],
                                 # norm_layer=norm_layer[0], window_size=window_size,
                                 # attn_ratio=attn_ratio[0], attn_loc='last', conv_type=conv_type[0])

        # self.patch_merge1 = ConvDownsampler(dim=embed_dims[0], norm_layer=norm_layer[0])

        self.skip1 = nn.Conv2d(embed_dims[0], embed_dims[0], 1)

        # self.layer2 = BasicLayer(
        #                          network_depth=sum(depths), dim=embed_dims[1], depth=depths[1],
        #                          num_heads=num_heads[1], mlp_ratio=mlp_ratios[1],
        #                          norm_layer=norm_layer[1], window_size=window_size,
        #                          attn_ratio=attn_ratio[1], attn_loc='last', conv_type=conv_type[1])
        self.layer2 = NATBlock(dim=embed_dims[1], depth=depths[1], num_heads=num_heads[1], kernel_size=kernel_size,network_depth=sum(depths),
                               dilations=None if dilations is None else dilations[1], mlp_ratio=self.mlp_ratios[1],
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               # drop_path=dpr[sum(depths[:1]): sum(depths[:2])],drop=drop_rate, attn_drop=attn_drop_rate,
                               norm_layer=norm_layer[1], downsample=True, layer_scale=layer_scale,
                               )
        # self.patch_merge1 = ConvDownsampler(dim=embed_dims[1], norm_layer=norm_layer[1])
        # self.patch_merge2 = PatchEmbed(
        #     patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])

        self.skip2 = nn.Conv2d(embed_dims[1], embed_dims[1], 1)
        # i_layer = i_layer - 1
        self.layer3 = NATBlock(dim=embed_dims[2], depth=depths[2], num_heads=num_heads[2], kernel_size=kernel_size,network_depth=sum(depths),
                               dilations=None if dilations is None else dilations[2], mlp_ratio=self.mlp_ratios[2],
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               # drop_path=dpr[sum(depths[:2]): sum(depths[:3])],drop=drop_rate, attn_drop=attn_drop_rate,
                               norm_layer=norm_layer[2], downsample=False, layer_scale=layer_scale,
                               )

        self.patch_split1 = ConvUpsampler(dim=embed_dims[2])

        assert embed_dims[1] == embed_dims[3]
        self.fusion1 = SKFusion(embed_dims[3])
        # i_layer = i_layer - 1
        self.layer4 = NATBlock(dim=embed_dims[3], depth=depths[3], num_heads=num_heads[3], kernel_size=kernel_size,network_depth=sum(depths),
                               dilations=None if dilations is None else dilations[3], mlp_ratio=self.mlp_ratios[3],
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               # drop_path=dpr[sum(depths[:3]): sum(depths[:4])],drop=drop_rate, attn_drop=attn_drop_rate,
                               norm_layer=norm_layer[3], downsample=False, layer_scale=layer_scale,
                               )

        self.patch_split2 = ConvUpsampler(dim=embed_dims[3])

        assert embed_dims[0] == embed_dims[4]
        self.fusion2 = SKFusion(embed_dims[4])

        self.layer5 = NATBlock(dim=embed_dims[4], depth=depths[4], num_heads=num_heads[4], kernel_size=kernel_size,network_depth=sum(depths),
                               dilations=None if dilations is None else dilations[4], mlp_ratio=self.mlp_ratios[4],
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               # drop_path=dpr[sum(depths[:4]): sum(depths[:5])],drop=drop_rate, attn_drop=attn_drop_rate,
                               norm_layer=norm_layer[4], downsample=False, layer_scale=layer_scale,
                               )


        self.patch_unembed =PatchUnEmbed(out_chans=out_chans, embed_dim=embed_dims[4], kernel_size=3)
        # self.patch_unembed = ReverseConvTokenizer(out_chans=out_chans, in_chans=embed_dims[4])


    def forward_features(self, x):
        x = self.patch_embed(x)
        x, skip1= self.layer1(x)
        skip1 = skip1.permute(0, 3, 1, 2)
        # skip1 = x
        #
        # x = self.patch_merge1(x)
        x, skip2= self.layer2(x)
        # skip2 = x
        skip2 = skip2.permute(0, 3, 1, 2)

        # x = self.patch_merge2(x)
        x = self.layer3(x)
        x = self.patch_split1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.fusion1([x, self.skip2(skip2)]) + x
        x = x.permute(0, 2, 3, 1)



        x = self.layer4(x)
        x = self.patch_split2(x)
        x = x.permute(0, 3, 1, 2)
        x = self.fusion2([x, self.skip1(skip1)]) + x

        x = x.permute(0, 2, 3, 1)
        x = self.layer5(x)
        x = x.permute(0, 3, 1, 2)
        x = self.patch_unembed(x)
        return x

    def forward(self, x):
        H, W = x.shape[2:]
        feat = self.forward_features(x)
        K, B = torch.split(feat, (1, 3), dim=1)

        x = K * x - B + x
        x = x[:, :, :H, :W]
        return x


def dehazeformer_t():
    return DehazeFormer(
        embed_dims=[24, 48, 96, 48, 24],
        mlp_ratios=[2., 4., 4., 2., 2.],
        depths=[4, 4, 4, 2, 2],
        num_heads=[2, 4, 6, 1, 1],
        attn_ratio=[0, 1 / 2, 1, 0, 0],
        conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'])


def dehazeformer_s():
    return DehazeFormer(
        embed_dims=[24, 48, 96, 48, 24],
        # embed_dims=[64, 128, 256, 128, 64],
        mlp_ratios=[2., 4., 4., 2., 2.],
        depths=[8, 8, 8, 4, 4],
        num_heads=[2, 4, 6, 1, 1],
        # drop_path_rate=0.3,
        layer_scale=1e-5,
        kernel_size=7)
        # attn_ratio=[1 / 4, 1 / 2, 3 / 4, 0, 0],
        # conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'])


def dehazeformer_b():
    return DehazeFormer(
        embed_dims=[24, 48, 96, 48, 24],
        mlp_ratios=[2., 4., 4., 2., 2.],
        depths=[16, 16, 16, 8, 8],
        num_heads=[2, 4, 6, 1, 1],
        attn_ratio=[1 / 4, 1 / 2, 3 / 4, 0, 0],
        conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'])


def dehazeformer_d():
    return DehazeFormer(
        embed_dims=[24, 48, 96, 48, 24],
        mlp_ratios=[2., 4., 4., 2., 2.],
        depths=[32, 32, 32, 16, 16],
        num_heads=[2, 4, 6, 1, 1],
        attn_ratio=[1 / 4, 1 / 2, 3 / 4, 0, 0],
        conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'])


def dehazeformer_w():
    return DehazeFormer(
        embed_dims=[48, 96, 192, 96, 48],
        mlp_ratios=[2., 4., 4., 2., 2.],
        depths=[16, 16, 16, 8, 8],
        num_heads=[2, 4, 6, 1, 1],
        attn_ratio=[1 / 4, 1 / 2, 3 / 4, 0, 0],
        conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'])


def dehazeformer_m():
    return DehazeFormer(
        embed_dims=[24, 48, 96, 48, 24],
        mlp_ratios=[2., 4., 4., 2., 2.],
        depths=[12, 12, 12, 6, 6],
        num_heads=[2, 4, 6, 1, 1],
        attn_ratio=[1 / 4, 1 / 2, 3 / 4, 0, 0],
        conv_type=['Conv', 'Conv', 'Conv', 'Conv', 'Conv'])


def dehazeformer_l():
    return DehazeFormer(
        embed_dims=[48, 96, 192, 96, 48],
        mlp_ratios=[2., 4., 4., 2., 2.],
        depths=[16, 16, 16, 12, 12],
        num_heads=[2, 4, 6, 1, 1],
        attn_ratio=[1 / 4, 1 / 2, 3 / 4, 0, 0],
        conv_type=['Conv', 'Conv', 'Conv', 'Conv', 'Conv'])
