import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from natten import NeighborhoodAttention2D as NeighborhoodAttention

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


class ConvDownsampler(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size - patch_size + 1) // 2, padding_mode='reflect')

    def forward(self, x):
        x = self.proj(x)
        return x


class ConvUpsampler(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans * patch_size ** 2, kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        x = self.proj(x)
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
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
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
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.BatchNorm2d(dim)

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, padding_mode='reflect')
        self.conv3_19 = nn.Conv2d(dim, dim, kernel_size=7, padding=9, groups=dim, dilation=3, padding_mode='reflect')
        self.conv3_13 = nn.Conv2d(dim, dim, kernel_size=5, padding=6, groups=dim, dilation=3, padding_mode='reflect')
        self.conv3_7 = nn.Conv2d(dim, dim, kernel_size=3, padding=3, groups=dim, dilation=3, padding_mode='reflect')

        self.mlp1 = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 4, 1),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim * 4, dim, 1)
        )
        self.attn1 = NeighborhoodAttention(
            dim,
            kernel_size=kernel_size,
            dilation=dilation,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.attn2 = NeighborhoodAttention(
            dim,
            kernel_size=5,
            dilation=8,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        # self.linear2 = nn.Linear(dim,dim*3)

        # Simple Channel Attention
        # self.Wv = nn.Sequential(
        #     nn.Conv2d(dim, dim, 1),
        #     nn.Conv2d(dim, dim, kernel_size=3, padding=3 // 2, groups=dim, padding_mode='reflect')
        # )
        # self.Wg = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(dim, dim, 1),
        #     nn.Sigmoid()
        # )

        # Channel Attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        # Pixel Attention
        self.pa = nn.Sequential(
            nn.Conv2d(dim, dim // 8, 1, padding=0, bias=True),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        # self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            network_depth,
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),

        )
        self.mlp2 = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim, dim, 1)
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




        shortcut = x
        x, rescale, rebias = self.norm1(x)
        x = x.permute(0, 2, 3, 1)  # B H W C
        x = self.attn1(x)
        x = x.permute(0, 3, 1, 2)  # B C H W
        x = x * rescale + rebias
        x = shortcut + x

        shortcut = x
        x, rescale, rebias = self.norm1(x)
        x = x.permute(0, 2, 3, 1)  # B H W C
        x = self.attn2(x)
        x = x.permute(0, 3, 1, 2)  # B C H W
        x = x * rescale + rebias
        x = shortcut + x
        # x = x * rescale + rebias
        # x = shortcut + x
        x, rescale, rebias = self.norm1(x)
        x = self.mlp(x)
        x = x * rescale + rebias
        x = shortcut + x



        shortcut = x  # bchw

        x = self.norm3(x)
        # x = torch.cat([self.Wv(x) * self.Wg(x), self.ca(x) * x, self.pa(x) * x], dim=1)

        x = self.pa(x) * x
        x = self.ca(x) * x
        x = self.mlp2(x)
        x = shortcut + x

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
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
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
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                    layer_scale=layer_scale,
                    network_depth=network_depth
                )

                for i in range(depth)
            ]
        )

        self.downsample = (
            None if not downsample else ConvDownsampler(patch_size=2, in_chans=dim, embed_dim=dim*2, kernel_size=3)
        )

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
            # x B H W C
        if self.downsample is None:
            return x
        return self.downsample(x), x
        # return x




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

class AFFFusion(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(AFFFusion, self).__init__()

        self.height = height
        d = max(int(dim / reduction), 4)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim * height, 1, bias=False)
        )
        self.conv3_19 = nn.Conv2d(dim, dim, kernel_size=9, padding=12, groups=dim, dilation=3,padding_mode='reflect')
        self.conv3_13 = nn.Conv2d(dim, dim, kernel_size=5, padding=6, groups=dim, dilation=3,padding_mode='reflect')
        self.conv3_7 = nn.Conv2d(dim, dim, kernel_size=3, padding=3, groups=dim, dilation=3, padding_mode='reflect')
        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape

        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)

        feats_sum = torch.sum(in_feats, dim=1)

        feats_sum = self.conv3_19(feats_sum)
        # feats_sum = self.conv3_13(feats_sum)
        feats_sum = self.conv3_7(feats_sum)
        # feats_sum = self.conv1(feats_sum)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        out = torch.sum(in_feats * attn, dim=1)
        return out




class MSCENAFormer(nn.Module):
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
                 attn_drop_rate=0.0,
                 drop_rate=0.0,
                 drop_path_rate=0.2
                 ):
        super(MSCENAFormer, self).__init__()

        # setting
        self.patch_size = 4

        self.kernel_size = kernel_size
        self.mlp_ratios = mlp_ratios

        # split image into overlapping patches
        self.patch_embed = ConvDownsampler(
            patch_size=1, in_chans=in_chans, embed_dim=embed_dims[0], kernel_size=3)

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]


        # backbone
        self.layer1 = NATBlock( dim=embed_dims[0],depth=depths[0],num_heads=num_heads[0],kernel_size=kernel_size,network_depth=sum(depths),
                                dilations=None if dilations is None else dilations[0],mlp_ratio=self.mlp_ratios[0],
                                qkv_bias=qkv_bias,qk_scale=qk_scale,drop=drop_rate,attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:0]) : sum(depths[:1])],
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
                               qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:1]): sum(depths[:2])],
                               norm_layer=norm_layer[1], downsample=True, layer_scale=layer_scale,
                               )
        # self.patch_merge1 = ConvDownsampler(dim=embed_dims[1], norm_layer=norm_layer[1])
        # self.patch_merge2 = PatchEmbed(
        #     patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])

        self.skip2 = nn.Conv2d(embed_dims[1], embed_dims[1], 1)
        # i_layer = i_layer - 1
        self.layer3 = NATBlock(dim=embed_dims[2], depth=depths[2], num_heads=num_heads[2], kernel_size=kernel_size,network_depth=sum(depths),
                               dilations=None if dilations is None else dilations[2], mlp_ratio=self.mlp_ratios[2],
                               qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:2]): sum(depths[:3])],
                               norm_layer=norm_layer[2], downsample=False, layer_scale=layer_scale,
                               )

        self.patch_split1 = ConvUpsampler(patch_size=2, out_chans=embed_dims[3], embed_dim=embed_dims[2])

        assert embed_dims[1] == embed_dims[3]
        self.fusion1 = AFFFusion(embed_dims[3])
        # self.aff1 = AFF(embed_dims[3])
        # i_layer = i_layer - 1
        self.layer4 = NATBlock(dim=embed_dims[3], depth=depths[3], num_heads=num_heads[3], kernel_size=kernel_size,network_depth=sum(depths),
                               dilations=None if dilations is None else dilations[3], mlp_ratio=self.mlp_ratios[3],
                               qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:3]): sum(depths[:4])],
                               norm_layer=norm_layer[3], downsample=False, layer_scale=layer_scale,
                               )

        self.patch_split2 = ConvUpsampler(   patch_size=2, out_chans=embed_dims[4], embed_dim=embed_dims[3])

        assert embed_dims[0] == embed_dims[4]
        self.fusion2 = AFFFusion(embed_dims[4])
        # self.aff2 = AFF(embed_dims[4])
        self.layer5 = NATBlock(dim=embed_dims[4], depth=depths[4], num_heads=num_heads[4], kernel_size=kernel_size,network_depth=sum(depths),
                               dilations=None if dilations is None else dilations[4], mlp_ratio=self.mlp_ratios[4],
                               qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:4]): sum(depths[:5])],
                               norm_layer=norm_layer[4], downsample=False, layer_scale=layer_scale,
                               )


        self.patch_unembed = ConvUpsampler(patch_size=1, out_chans=out_chans, embed_dim=embed_dims[4], kernel_size=3)


    def forward_features(self, x):
        x = self.patch_embed(x)
        x, skip1= self.layer1(x)
        # skip1 = skip1.permute(0, 3, 1, 2)
        # skip1 = x
        #
        # x = self.patch_merge1(x)
        x, skip2= self.layer2(x)
        # skip2 = x
        # skip2 = skip2.permute(0, 3, 1, 2)

        # x = self.patch_merge2(x)
        x = self.layer3(x)
        x = self.patch_split1(x)
        # x = x.permute(0, 3, 1, 2)
        x = self.fusion1([x, self.skip2(skip2)]) + x
        # x = self.aff1(self.fusion1([x, self.skip2(skip2)]), x)
        # x = self.aff1(x, self.skip2(skip2))
        # x = x.permute(0, 2, 3, 1)

        x = self.layer4(x)
        x = self.patch_split2(x)
        # x = x.permute(0, 3, 1, 2)
        x = self.fusion2([x, self.skip1(skip1)]) + x
        x = self.layer5(x)
        # x = x.permute(0, 3, 1, 2)
        x = self.patch_unembed(x)
        return x

    def forward(self, x):
        H, W = x.shape[2:]
        feat = self.forward_features(x)
        K, B = torch.split(feat, (1, 3), dim=1)

        x = K * x - B + x
        x = x[:, :, :H, :W]
        return x




def mscenaformer_s():
    return MSCENAFormer(
        embed_dims=[24, 48, 96, 48, 24],
        # embed_dims=[64, 128, 256, 128, 64],
        mlp_ratios=[2., 4., 4., 2., 2.],
        depths=[4, 4, 8, 4, 4],
        # depths=[2,2, 4, 2, 2],
        num_heads=[2, 4, 6, 1, 1],
        drop_path_rate=0.1,
        layer_scale=1e-3,
        kernel_size=7)


def mscenaformer_b():
    return MSCENAFormer(
        embed_dims=[24, 48, 96, 48, 24],
        mlp_ratios=[2., 4., 4., 2., 2.],
        depths=[16, 16, 16, 8, 8],
        num_heads=[2, 4, 6, 1, 1],
        drop_path_rate=0.1,
        layer_scale=1e-3,
        kernel_size=7)
        # attn_ratio=[1 / 4, 1 / 2, 3 / 4, 0, 0],
        # conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'])



def mscenaformer_l():
    return MSCENAFormer(
        embed_dims=[48, 96, 192, 96, 48],
        mlp_ratios=[2., 4., 4., 2., 2.],
        depths=[16, 16, 16, 12, 12],
        num_heads=[2, 4, 6, 1, 1],
        attn_ratio=[1 / 4, 1 / 2, 3 / 4, 0, 0],
        conv_type=['Conv', 'Conv', 'Conv', 'Conv', 'Conv'])
