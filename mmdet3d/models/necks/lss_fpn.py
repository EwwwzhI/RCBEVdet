import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer

from torch.utils.checkpoint import checkpoint
from mmdet3d.models.backbones.resnet import ConvModule
from mmdet.models import NECKS


@NECKS.register_module()
class FPN_LSS(nn.Module):

    def __init__(self,
                 in_channels,                   # 640+160=800
                 out_channels,                  # 256
                 scale_factor=4,                # 上采样扩大因子
                 input_feature_index=(0, 2),    # 输入层索引 选择主干第三阶段输出和主干第一阶段输出作为模型输入
                 norm_cfg=dict(type='BN'),
                 extra_upsample=2,
                 lateral=None,
                 use_input_conv=False,
                 with_cp=False,
                 ):
        super().__init__()
        self.with_cp = with_cp
        self.input_feature_index = input_feature_index
        self.extra_upsample = extra_upsample is not None
        self.up = nn.Upsample(
            scale_factor=scale_factor, mode='bilinear', align_corners=True)
        # assert norm_cfg['type'] in ['BN', 'SyncBN']
        channels_factor = 2 if self.extra_upsample else 1
        self.input_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels * channels_factor,   # 256*2
                kernel_size=1,
                padding=0,
                bias=False),
            build_norm_layer(
                norm_cfg, out_channels * channels_factor, postfix=0)[1],
            nn.ReLU(inplace=True),
        ) if use_input_conv else None
        if use_input_conv:
            in_channels = out_channels * channels_factor
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels * channels_factor,
                kernel_size=3,
                padding=1,
                bias=False),
            build_norm_layer(
                norm_cfg, out_channels * channels_factor, postfix=0)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels * channels_factor,
                out_channels * channels_factor,
                kernel_size=3,
                padding=1,
                bias=False),
            build_norm_layer(
                norm_cfg, out_channels * channels_factor, postfix=0)[1],
            nn.ReLU(inplace=True),
        )
        if self.extra_upsample:
            self.up2 = nn.Sequential(
                nn.Upsample(
                    scale_factor=extra_upsample,
                    mode='bilinear',
                    align_corners=True),
                nn.Conv2d(
                    out_channels * channels_factor,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False),
                build_norm_layer(norm_cfg, out_channels, postfix=0)[1],
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    out_channels, out_channels, kernel_size=1, padding=0),
            )
        self.lateral = lateral is not None
        if self.lateral:
            self.lateral_conv = nn.Sequential(
                nn.Conv2d(
                    lateral, lateral, kernel_size=1, padding=0, bias=False),
                build_norm_layer(norm_cfg, lateral, postfix=0)[1],
                nn.ReLU(inplace=True),
            )

    def forward(self, feats):
        x2, x1 = feats[self.input_feature_index[0]], \
                 feats[self.input_feature_index[1]]
        if self.lateral:
            if self.with_cp:
                x2 = checkpoint(self.lateral_conv, x2)
            else:
                x2 = self.lateral_conv(x2)
        if self.with_cp:
            x1 = checkpoint(self.up, x1)
        else:
            x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        if self.input_conv is not None:
            if self.with_cp:
                x = checkpoint(self.input_conv, x)
            else:
                x = self.input_conv(x)
        if self.with_cp:
            x = checkpoint(self.conv, x)
        else:
            x = self.conv(x)
        if self.extra_upsample:
            if self.with_cp:
                x = checkpoint(self.up2, x)
            else:
                x = self.up2(x)
        return x



@NECKS.register_module()
class LSSFPN3D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 with_cp=False):
        super().__init__()
        self.up1 =  nn.Upsample(
            scale_factor=2, mode='trilinear', align_corners=True)
        self.up2 =  nn.Upsample(
            scale_factor=4, mode='trilinear', align_corners=True)

        self.conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d', ),
            act_cfg=dict(type='ReLU',inplace=True))
        self.with_cp = with_cp

    def forward(self, feats):
        x_8, x_16, x_32 = feats
        x_16 = self.up1(x_16)
        x_32 = self.up2(x_32)
        x = torch.cat([x_8, x_16, x_32], dim=1)
        if self.with_cp:
            x = checkpoint(self.conv, x)
        else:
            x = self.conv(x)
        return x