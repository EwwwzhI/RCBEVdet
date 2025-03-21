# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner import auto_fp16
from torch import nn

from ..builder import MIDDLE_ENCODERS

from mmdet3d.core import draw_heatmap_gaussian, draw_heatmap_gaussian_feat


@MIDDLE_ENCODERS.register_module()
class PointPillarsScatter(nn.Module):
    """Point Pillar's Scatter.

    Converts learned features from dense tensor to sparse pseudo image.

    Args:
        in_channels (int): Channels of input features.
        output_shape (list[int]): Required output shape of features.
    """

    def __init__(self, in_channels, output_shape):
        super().__init__()
        self.output_shape = output_shape
        self.ny = output_shape[0]  # 512
        self.nx = output_shape[1]  # 512
        self.in_channels = in_channels  # 64
        self.fp16_enabled = False

    @auto_fp16(apply_to=('voxel_features', ))
    def forward(self, voxel_features, coors, batch_size=None):
        """Foraward function to scatter features."""
        # TODO: rewrite the function in a batch manner
        # no need to deal with different batch cases
        if batch_size is not None:
            return self.forward_batch(voxel_features, coors, batch_size)
        else:
            return self.forward_single(voxel_features, coors)

    def forward_single(self, voxel_features, coors):
        """Scatter features of single sample.

        Args:
            voxel_features (torch.Tensor): Voxel features in shape (N, C).
            coors (torch.Tensor): Coordinates of each voxel.
                The first column indicates the sample ID.
        """
        # Create the canvas for this sample 创建一个全零的二维张量，用于存储散射后的 BEV 伪图像
        canvas = torch.zeros(
            self.in_channels,   # 64
            self.nx * self.ny,  # 512*512 = 262144
            dtype=voxel_features.dtype,
            device=voxel_features.device)

        indices = coors[:, 2] * self.nx + coors[:, 3]   # 将二维坐标 (Y, X) 转换为一维索引[90000]
        indices = indices.long()                        # 转换为整数索引
        voxels = voxel_features.t()                     # 转置后形状 [64, 90000]
        # Now scatter the blob back to the canvas.
        canvas[:, indices] = voxels                     # 按索引填充画布
        # Undo the column stacking to final 4-dim tensor
        canvas = canvas.view(1, self.in_channels, self.ny, self.nx)  # 将 [64, 262144] 的二维画布转换为 [1, 64, 512, 512] 的四维张量
        return canvas

    def forward_batch(self, voxel_features, coors, batch_size):
        """Scatter features of single sample.

        Args:
            voxel_features (torch.Tensor): Voxel features in shape (N, C).
            coors (torch.Tensor): Coordinates of each voxel in shape (N, 4).
                The first column indicates the sample ID.
            batch_size (int): Number of samples in the current batch.
        """
        # batch_canvas will be the final output.
        batch_canvas = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(
                self.in_channels,
                self.nx * self.ny,
                dtype=voxel_features.dtype,
                device=voxel_features.device)

            # Only include non-empty pillars
            batch_mask = coors[:, 0] == batch_itt
            this_coors = coors[batch_mask, :]
            indices = this_coors[:, 2] * self.nx + this_coors[:, 3]
            indices = indices.type(torch.long)
            voxels = voxel_features[batch_mask, :]
            voxels = voxels.t()

            # Now scatter the blob back to the canvas.
            canvas[:, indices] = voxels

            # Append to a list for later stacking.
            batch_canvas.append(canvas)

        # Stack to 3-dim tensor (batch-size, in_channels, nrows*ncols)
        batch_canvas = torch.stack(batch_canvas, 0)

        # Undo the column stacking to final 4-dim tensor
        batch_canvas = batch_canvas.view(batch_size, self.in_channels, self.ny,
                                         self.nx)

        return batch_canvas

@MIDDLE_ENCODERS.register_module()
class PointPillarsScatterRCS(PointPillarsScatter):

    def __init__(self, in_channels, output_shape):
        super(PointPillarsScatterRCS, self).__init__(in_channels, output_shape)
        self.compress = nn.Conv2d(in_channels*2, in_channels, 3, padding=1)      # 128 64
        self.rcs_att = nn.Conv2d(2, in_channels, 1)
        # self.rcs_att = nn.Conv2d(2+1, in_channels, 1)

    def forward(self, voxel_features, coors, batch_size=None):
        point_features, rcs = voxel_features
        features = super().forward(point_features, coors, batch_size)  # [1,64,512,512] BEV 伪图像

        # 初始化热力图
        heatmap = point_features.new_zeros((batch_size, self.ny,self.nx))     # 创建一个全零的三维张量，形状为【batch_size，512，512】 存储高斯热图，表示雷达反射的强度分布
        heatmap_feat = point_features.new_zeros((batch_size, 1, self.ny,self.nx))  # 创建一个全零的四维张量，形状为【batch_size，1，512，512】用于存储带有特征的高斯热图

        r = rcs[:, 0]**2 + rcs[:, 1]**2    # 计算半径
        true_rcs = rcs[:, -2] * r          # rcs * r
        true_rcs = torch.nn.functional.relu(true_rcs)

        radius = true_rcs + 1  # 高斯热图的半径 radius 加 1 是为了确保半径始终大于 0

        for i in range(coors.shape[0]):
            batch, _, y, x = coors[i]   # 从 coors 张量中提取当前体素的批次索引、y 坐标和 x 坐标
            # 在基础热力图上绘制高斯分布
            draw_heatmap_gaussian(heatmap[batch], [x, y], int(radius[i].data.item()))  # ???代码是不是有问题 这样heatmap不是始终为全零的三维张量
            # 在特征热力图上绘制带 RCS 权值的高斯分布
            heatmap_feat[batch] = draw_heatmap_gaussian_feat(heatmap_feat[batch], [x, y], int(radius[i].data.item()), rcs[i, -2])

        rcs_att = self.rcs_att(torch.cat([heatmap.unsqueeze(dim=1), heatmap_feat],dim=1))  # 【batch_size=1，64，512，512】

        features_att = self.compress(torch.cat([features, rcs_att], dim=1))  # [1,64,512,512]
        return features_att