# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmcv.runner import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from ..builder import BACKBONES
from ..utils import ResLayer
from torch.quantization import QuantStub, DeQuantStub


class BasicBlock(BaseModule):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None,
                 init_cfg=None):
        super(BasicBlock, self).__init__(init_cfg)
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg, planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


class Bottleneck(BaseModule):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None,
                 init_cfg=None):
        """Bottleneck block for ResNet.

        If style is "pytorch", the stride-two layer is the 3x3 conv layer, if
        it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__(init_cfg)
        assert style in ['pytorch', 'caffe']
        assert dcn is None or isinstance(dcn, dict)
        assert plugins is None or isinstance(plugins, list)
        if plugins is not None:
            allowed_position = ['after_conv1', 'after_conv2', 'after_conv3']
            assert all(p['position'] in allowed_position for p in plugins)

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.plugins = plugins
        self.with_plugins = plugins is not None

        if self.with_plugins:
            # collect plugins for conv1/conv2/conv3
            self.after_conv1_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv1'
            ]
            self.after_conv2_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv2'
            ]
            self.after_conv3_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv3'
            ]

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, planes * self.expansion, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        fallback_on_stride = False
        if self.with_dcn:
            fallback_on_stride = dcn.pop('fallback_on_stride', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = build_conv_layer(
                conv_cfg,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)
        else:
            assert self.conv_cfg is None, 'conv_cfg must be None for DCN'
            self.conv2 = build_conv_layer(
                dcn,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)

        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        if self.with_plugins:
            self.after_conv1_plugin_names = self.make_block_plugins(
                planes, self.after_conv1_plugins)
            self.after_conv2_plugin_names = self.make_block_plugins(
                planes, self.after_conv2_plugins)
            self.after_conv3_plugin_names = self.make_block_plugins(
                planes * self.expansion, self.after_conv3_plugins)

    def make_block_plugins(self, in_channels, plugins):
        """make plugins for block.

        Args:
            in_channels (int): Input channels of plugin.
            plugins (list[dict]): List of plugins cfg to build.

        Returns:
            list[str]: List of the names of plugin.
        """
        assert isinstance(plugins, list)
        plugin_names = []
        for plugin in plugins:
            plugin = plugin.copy()
            name, layer = build_plugin_layer(
                plugin,
                in_channels=in_channels,
                postfix=plugin.pop('postfix', ''))
            assert not hasattr(self, name), f'duplicate plugin {name}'
            self.add_module(name, layer)
            plugin_names.append(name)
        return plugin_names

    def forward_plugin(self, x, plugin_names):
        out = x
        for name in plugin_names:
            out = getattr(self, name)(out)
        return out

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: normalization layer after the third convolution layer"""
        return getattr(self, self.norm3_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv1_plugin_names)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv2_plugin_names)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv3_plugin_names)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


@BACKBONES.register_module()
class ResNet(BaseModule):
    """ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        stem_channels (int | None): Number of stem channels. If not specified,
            it will be the same as `base_channels`. Default: None.
        base_channels (int): Number of base channels of res layer. Default: 64.
        in_channels (int): Number of input image channels. Default: 3.
        num_stages (int): Resnet stages. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        norm_cfg (dict): Dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        plugins (list[dict]): List of plugins for stages, each dict contains:

            - cfg (dict, required): Cfg dict to build plugin.
            - position (str, required): Position inside block to insert
              plugin, options are 'after_conv1', 'after_conv2', 'after_conv3'.
            - stages (tuple[bool], optional): Stages to apply plugin, length
              should be same as 'num_stages'.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity.
        pretrained (str, optional): model pretrained path. Default: None
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None

    Example:
        >>> from mmdet.models import ResNet
        >>> import torch
        >>> self = ResNet(depth=18)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 64, 8, 8)
        (1, 128, 4, 4)
        (1, 256, 2, 2)
        (1, 512, 1, 1)
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 # 网络深度
                 depth,
                 # 输入图像的channel数
                 in_channels=3,
                 # 主干卷积层的channel数，默认等于base_channels
                 stem_channels=None,
                 base_channels=64,
                 # stage数量
                 num_stages=4,
                 # 每个stage残差层第一个残差块的stride参数
                 strides=(1, 2, 2, 2),
                 # 膨胀（空洞）卷积参数设置
                 dilations=(1, 1, 1, 1),
                 # 输出特征图的索引，每个stage对应一个
                 out_indices=(0, 1, 2, 3),
                 # 风格设置
                 style='pytorch',
                 # 是否用3个3×3的卷积核代替主干上1个7×7的卷积核
                 deep_stem=False,
                 # 是否使用平均池化代替stride为2的卷积操作进行下采样
                 avg_down=False,
                 # 冻结层数，-1表示不冻结
                 frozen_stages=-1,
                 # 构建卷积层的配置
                 conv_cfg=None,
                 # 构建归一化层的配置
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 # 是否使用dcn（可变形卷积）
                 dcn=None,
                 # 指定哪个stage使用dcn
                 stage_with_dcn=(False, False, False, False),
                 plugins=None,
                 with_cp=False,
                 # 是否对残差块进行0初始化
                 zero_init_residual=True,
                 # 预训练模型
                 pretrained=None,
                 # 指定预训练模型
                 init_cfg=None,
                 quantized=False):
        super(ResNet, self).__init__(init_cfg)
        self.zero_init_residual = zero_init_residual

        # 判断是否有该depth设置下的模型
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')

        block_init_cfg = None
        # 下面进行预训练模型设定
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        # 如果指定预训练模型，就会自动读取模型配置与参数
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        # 如果没有指定预训练模型(init_cfg is None)就会自动生成模型配置参数组装模型
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
                block = self.arch_settings[depth][0]
                if self.zero_init_residual:
                    if block is BasicBlock:
                        block_init_cfg = dict(
                            type='Constant',
                            val=0,
                            override=dict(name='norm2'))
                    elif block is Bottleneck:
                        block_init_cfg = dict(
                            type='Constant',
                            val=0,
                            override=dict(name='norm3'))
        else:
            raise TypeError('pretrained must be a str or None')

        self.depth = depth
        self.quant = QuantStub()  
        self.dequant = DeQuantStub()  
        self.quantized = quantized
        if stem_channels is None:
            stem_channels = base_channels
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
        self.plugins = plugins
        self.block, stage_blocks = self.arch_settings[depth]  # self.block：Bottleneck 每个阶段（stage）中使用的块（block）的数量
        self.stage_blocks = stage_blocks[:num_stages]  # 从预定义的 stage_blocks 列表中截取前 num_stages 个元素，作为每个阶段（stage）中使用的块（block）的数量
        self.inplanes = stem_channels  # 每一个残差层的输入通道数

        # 使用self._make_stem_layer方法构造stem_layer
        self._make_stem_layer(in_channels, stem_channels)

        # 下面使用self.make_stage_plugins构造res_layer残差层
        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):  # 根据每个stage内block的数量构造res_layer残差层
            stride = strides[i]
            dilation = dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            if plugins is not None:
                stage_plugins = self.make_stage_plugins(plugins, i)
            else:
                stage_plugins = None
            planes = base_channels * 2**i
            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=self.inplanes,  # 残差层的输入通道数 ：64
                planes=planes,           # 残差层的输出通道数 第一个残差层：64   第二个残差层：128 第三个残差层：256 第四个残差层：512
                num_blocks=num_blocks,   # 残差层的残差块数量 第一个残差层：3    第二个残差层：4   第三个残差层：6   第四个残差层：3
                stride=stride,           # 残差层的步幅      第一个残差层：1    第二个残差层：2   第三个残差层：2   第四个残差层：2
                dilation=dilation,       # 残差层的扩张系数   第一个残差层：1    第二个残差层：1   第三个残差层：1   第四个残差层：1
                style=self.style,
                avg_down=self.avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                plugins=stage_plugins,
                init_cfg=block_init_cfg)
            self.inplanes = planes * self.block.expansion  # 扩张系数expansion = 4 系数在Bottleneck模型内
            layer_name = f'layer{i + 1}'   # layer_name = ['layer1', 'layer2', 'layer3', 'layer4']
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        # 固定指定stage的权重
        self._freeze_stages()

        self.feat_dim = self.block.expansion * base_channels * 2**(
            len(self.stage_blocks) - 1)
        
        if self.plugins is not None:  
            self.plugin_quants = nn.ModuleList()  
            self.plugin_dequants = nn.ModuleList()  
            for _ in range(3):  # after_conv1, after_conv2, after_conv3  
                self.plugin_quants.append(QuantStub())  
                self.plugin_dequants.append(DeQuantStub())  

    def make_stage_plugins(self, plugins, stage_idx):
        """Make plugins for ResNet ``stage_idx`` th stage.

        Currently we support to insert ``context_block``,
        ``empirical_attention_block``, ``nonlocal_block`` into the backbone
        like ResNet/ResNeXt. They could be inserted after conv1/conv2/conv3 of
        Bottleneck.

        An example of plugins format could be:

        Examples:
            >>> plugins=[
            ...     dict(cfg=dict(type='xxx', arg1='xxx'),
            ...          stages=(False, True, True, True),
            ...          position='after_conv2'),
            ...     dict(cfg=dict(type='yyy'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3'),
            ...     dict(cfg=dict(type='zzz', postfix='1'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3'),
            ...     dict(cfg=dict(type='zzz', postfix='2'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3')
            ... ]
            >>> self = ResNet(depth=18)
            >>> stage_plugins = self.make_stage_plugins(plugins, 0)
            >>> assert len(stage_plugins) == 3

        Suppose ``stage_idx=0``, the structure of blocks in the stage would be:

        .. code-block:: none

            conv1-> conv2->conv3->yyy->zzz1->zzz2

        Suppose 'stage_idx=1', the structure of blocks in the stage would be:

        .. code-block:: none

            conv1-> conv2->xxx->conv3->yyy->zzz1->zzz2

        If stages is missing, the plugin would be applied to all stages.

        Args:
            plugins (list[dict]): List of plugins cfg to build. The postfix is
                required if multiple same type plugins are inserted.
            stage_idx (int): Index of stage to build

        Returns:
            list[dict]: Plugins for current stage
        """
        stage_plugins = []
        for plugin in plugins:
            plugin = plugin.copy()
            stages = plugin.pop('stages', None)
            assert stages is None or len(stages) == self.num_stages
            # whether to insert plugin into current stage
            if stages is None or stages[stage_idx]:
                stage_plugins.append(plugin)

        return stage_plugins

    def make_res_layer(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer``."""
        return ResLayer(**kwargs)

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_channels):
        if self.deep_stem:
            self.stem = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels)[1],
                nn.ReLU(inplace=True))
        else:
        # stem_layer构造如下：
            # Conv2d(in_c = 3, out_c = 64, kenerl_size = 7, s = 2, p = 3)
            self.conv1 = build_conv_layer(
                self.conv_cfg,          # None, 在build_conv_layer中会被给予Conv2d
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)
            # BN_norm层
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            # 激活函数relu
            self.relu = nn.ReLU(inplace=True)
        # maxpool层
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x):
        if self.quantized:
            x = self.quant(x) 
        
        """Forward function."""
        if self.deep_stem:        # self.deep_stem = False
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        # print('ResNet start')
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:  # 网络输出来自 out_indices（2，3）阶段
                if self.quantized:
                    outs.append(self.dequant(x) if i == max(self.out_indices) else x)  
                else:
                    outs.append(x)
                
        if self.quantized:        
            if len(self.out_indices) == 1:  
                return self.dequant(tuple(outs)[0])  
        
        return tuple(outs)
        

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(ResNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
    
    def fuse_model(self):  
        """Fuse conv/bn/relu modules in resnet model."""  
        modules_to_fuse = []  
        if self.deep_stem:  
            for i in range(3):  
                modules_to_fuse.append(['stem.%d' % (i*3), 'stem.%d' % (i*3+1), 'stem.%d' % (i*3+2)])  
        else:  
            modules_to_fuse.append(['conv1', self.norm1_name, 'relu'])  
        
        for i, layer_name in enumerate(self.res_layers):  
            layer = getattr(self, layer_name)  
            for j in range(len(layer)):  
                if isinstance(layer[j], BasicBlock):  
                    modules_to_fuse.append(['%s.%d.conv1' % (layer_name, j),   
                                           '%s.%d.norm1' % (layer_name, j),   
                                           '%s.%d.relu' % (layer_name, j)])  
                    modules_to_fuse.append(['%s.%d.conv2' % (layer_name, j),   
                                           '%s.%d.norm2' % (layer_name, j)])  

                    if layer[j].downsample is not None:  
                        modules_to_fuse.append(['%s.%d.downsample.0' % (layer_name, j),   
                                               '%s.%d.downsample.1' % (layer_name, j)])  

                elif isinstance(layer[j], Bottleneck):  
                    modules_to_fuse.append(['%s.%d.conv1' % (layer_name, j),   
                                           '%s.%d.norm1' % (layer_name, j),   
                                           '%s.%d.relu' % (layer_name, j)])  
                    modules_to_fuse.append(['%s.%d.conv2' % (layer_name, j),   
                                           '%s.%d.norm2' % (layer_name, j),   
                                           '%s.%d.relu' % (layer_name, j)])  
                    modules_to_fuse.append(['%s.%d.conv3' % (layer_name, j),   
                                           '%s.%d.norm3' % (layer_name, j)])  

                    if layer[j].downsample is not None:  
                        modules_to_fuse.append(['%s.%d.downsample.0' % (layer_name, j),   
                                               '%s.%d.downsample.1' % (layer_name, j)])  
        
        for m in modules_to_fuse:  
            try:  
                torch.quantization.fuse_modules(self, m, inplace=True)  
                print(f"Successfully fused: {m}")  
            except Exception as e:  
                print(f"Failed to fuse: {m}, Error: {e}")  
                

@BACKBONES.register_module()
class ResNetV1d(ResNet):
    r"""ResNetV1d variant described in `Bag of Tricks
    <https://arxiv.org/pdf/1812.01187.pdf>`_.

    Compared with default ResNet(ResNetV1b), ResNetV1d replaces the 7x7 conv in
    the input stem with three 3x3 convs. And in the downsampling block, a 2x2
    avg_pool with stride 2 is added before conv, whose stride is changed to 1.
    """

    def __init__(self, **kwargs):
        super(ResNetV1d, self).__init__(
            deep_stem=True, avg_down=True, **kwargs)


def quantize_resnet(model, data_loader, backend='fbgemm'):  
    
    model.eval()  
    if backend == 'fbgemm':  
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')  
    else:  
        model.qconfig = torch.quantization.get_default_qconfig('qnnpack')  
    
    model.quantize = True  
    
    model.fuse_model()  
    
    torch.quantization.prepare(model, inplace=True)  
    
    with torch.no_grad():  
        for batch in data_loader:  
            if isinstance(batch, list) or isinstance(batch, tuple):  
                x = batch[0]  
            else:  
                x = batch  
            model(x)  
    
    torch.quantization.convert(model, inplace=True)  
    
    return model  

if __name__ == "__main__":
    
    model = ResNet(depth=18, quantized=True)
    model.load_state_dict(torch.load('resnet50_weights.pth'))  
    
    # 准备校准数据  
    calibration_dataset = YourDataset(...)  
    calibration_loader = torch.utils.data.DataLoader(  
        calibration_dataset,  
        batch_size=32,  
        shuffle=False,  
        num_workers=4  
    )  
    
    # 量化模型  
    model_quantized = quantize_resnet(model, calibration_loader, backend='fbgemm')  
    
    # 评估量化模型  
    val_dataset = YourValDataset(...)  
    val_loader = torch.utils.data.DataLoader(  
        val_dataset,  
        batch_size=32,  
        shuffle=False,  
        num_workers=4  
    )  
    
    evaluate_model(model_quantized, val_loader)  
    
    # 保存量化模型  
    torch.save(model_quantized.state_dict(), 'resnet50_quantized.pth')  
    
    