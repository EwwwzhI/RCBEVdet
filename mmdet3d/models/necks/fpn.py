# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16

from ..builder import NECKS


@NECKS.register_module()
class CustomFPN(BaseModule):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,     # FPN每个尺寸的输入通道
                 out_channels,    # FPN每个尺寸的输出通道
                 num_outs,        # FPN输出层数数量（输出尺寸数量）
                 start_level=0,   # FPN开始构建特征金字塔的输入骨干网络的起始层级索引，值为0表示从骨干网络的第一层开始
                 end_level=-1,    # FPN构建特征金字塔的输入骨干网络的最终层级索引，值为-1表示骨干网络的所有层都检索
                 out_ids=[],
                 add_extra_convs=False,  # 额外输出层的特征图来源
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),  # FPN构建特征金字塔上采样模块的配置 最近邻插值算法
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(CustomFPN, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels    # 输入来自ResNet的第三、四层特征输出 第三层：（16，44，1024） 第四层：（8，22，2048）
        self.out_channels = out_channels  # 输出通道数：512
        self.num_ins = len(in_channels)   # 输入特征图层级：2
        self.num_outs = num_outs          # 输出层数（尺寸数）：1
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()  # 上采样参数
        self.out_ids = out_ids
        if end_level == -1:
            self.backbone_end_level = self.num_ins  # 来自输入骨干网络的所以特征层都索引
            # assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()  # 横向卷积层列表（在上采样前）
        self.fpn_convs = nn.ModuleList()      # fpn卷积层列表（在上采样后）

        for i in range(self.start_level, self.backbone_end_level):  # i=0,1
            l_conv = ConvModule(
                # 构造上采样前的卷积核
                in_channels[i],
                out_channels,
                1,  # 卷积核尺寸
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            if i in self.out_ids:  # 该算法只有一个尺寸的特征输出 0：第一个层
                fpn_conv = ConvModule(
                    # 构造上采样后的卷积核
                    out_channels,
                    out_channels,
                    3,  # 卷积核尺寸
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals  用来记录每一次卷积计算后的输出值，可以理解成是一个临时变量temp
        laterals = [
            lateral_conv(inputs[i + self.start_level])  # 输入通过已经构建的卷积核输出特征
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path 上采样过程
        used_backbone_levels = len(laterals)  # 2层数
        for i in range(used_backbone_levels - 1, 0, -1):  # i = 1
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            # scale_factor：按固定比例放大   size：直接指定目标尺寸
            # it cannot co-exist with `size` in `F.interpolate`. PyTorch的F.interpolate不允许同时指定scale_factor和size，故需分支判断：
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                # upsample 与相加的操作，可以理解成经过“upsample”与“+”的操作后
                prev_shape = laterals[i - 1].shape[2:]   # 获取低层特征图的尺寸（H, W）
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)  # 通过**self.upsample_cfg传递其他插值参数（如mode='nearest'或align_corners=True）

        # build outputs 建立输出
        # part 1: from original levels out_ids=[0]
        outs = [self.fpn_convs[i](laterals[i]) for i in self.out_ids]
        # part 2: add extra levels 如果设定输出层大于实际输出层 添加额外的层
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return outs[0]  # 有点疑问为什么是[0]，这个算法是只存在一个层，但是如果层数多感觉应该是输出outs
