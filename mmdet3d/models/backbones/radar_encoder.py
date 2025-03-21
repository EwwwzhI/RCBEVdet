from torch import nn

from typing import Any, Dict
from functools import partial
import torch
from mmcv.cnn import build_norm_layer
from torch import nn
from torch.nn import functional as F
from timm.models.layers import DropPath, Mlp, to_2tuple
from mmdet3d.models.builder import build_backbone
from mmdet3d.models.builder import BACKBONES
import time
from mmcv.cnn.bricks.transformer import MultiheadAttention, FFN

def get_paddings_indicator(actual_num, max_num, axis=0):
    """Create boolean mask by actually number of a padded tensor.
    创建一个布尔掩码（boolean mask），用于指示一个填充张量（padded tensor）中哪些元素是实际数据，哪些元素是填充值（padding）
    Args:
        actual_num ([type]): [description]
        max_num ([type]): [description]
    Returns:
        [type]: [description]
    """
    # actual_num:[a1,a2,a3,...,an] a1:第一个体素包含实际雷达点数 n:体素数
    actual_num = torch.unsqueeze(actual_num, axis + 1)  # 增加一个维度--->actual_num:[[a1],[a2],[a3],...,[an]]
    # tiled_actual_num: [N, M, 1]
    max_num_shape = [1] * len(actual_num.shape)  # 创建一个包含多个 1 的列表，列表的长度等于 actual_num 的维度数量,actual_num 的形状是 (N, 1)，max_num_shape = [1, 1]
    max_num_shape[axis + 1] = -1  # max_num_shape = [1, -1],-1 在 PyTorch 中表示该维度的大小由 PyTorch 自动推断
    # 创建一个形状为 max_num_shape 的张量 max_num(二维)
    # 其中包含了从 0 到 max_num - 1 的整数序列。
    # max_num 是 10，axis 是 0，且 actual_num 的形状是 (n, 1)，那么 max_num 的形状将是 (n, 10)，又因为max_num_shape 形状 [1, -1]，所以max_num形状[1,10] 且值为 [[0, 1, 2, 3, 4,5,6,7,8,9]]
    max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(
        max_num_shape
    )
    # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]  相当于a1=3,a2=4,a3=2
    # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
    # 如果 actual_num 中的元素大于 max_num 中的对应元素，则 paddings_indicator 中对应的元素为 True，表示该元素是实际值；否则为 False，表示该元素是填充数据
    # a1分别和0，1，2，3，4作比较
    paddings_indicator = actual_num.int() > max_num
    # paddings_indicator shape: [batch_size, max_num]
    return paddings_indicator  # [n,10]


class RFNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg=None, last_layer=False):
        """
        Pillar Feature Net Layer.
        The Pillar Feature Net could be composed of a series of these layers, but the PointPillars paper results only
        used a single PFNLayer. This layer performs a similar role as second.pytorch.voxelnet.VFELayer.
        :param in_channels: <int>. Number of input channels.
        :param out_channels: <int>. Number of output channels.
        :param last_layer: <bool>. If last_layer, there is no concatenation of features.
        """

        super().__init__()
        self.name = "RFNLayer"
        self.last_vfe = last_layer  # false
        
        self.units = out_channels

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)
        self.norm_cfg = norm_cfg

        self.linear = nn.Linear(in_channels, self.units, bias=False)
        self.norm = build_norm_layer(self.norm_cfg, self.units)[1]

    def forward(self, inputs):

        x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        torch.backends.cudnn.enabled = True
        x = F.relu(x)

        if self.last_vfe:
            x_max = torch.max(x, dim=1, keepdim=True)[0]
            return x_max
        else:
            return x


class PointEmbed(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.in_c = in_c                # 11
        self.out_c = out_c              # 32
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_c, out_c, 1),
            nn.BatchNorm1d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_c, out_c, 1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_c*2, out_c*2, 1),
            nn.BatchNorm1d(out_c*2),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_c*2, out_c, 1)
        )

    def forward(self, points):

        bs, n, c = points.shape
        feature = self.conv1(points.transpose(2, 1))  # bs c n
        # 在点数维度（dim=2）上取最大值，得到每个通道的全局特征
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # bs c 1
        
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # bs c*2 n
        feature = self.conv2(feature)

        return feature.transpose(2, 1)

class Extractor(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 with_cffn=True, cffn_ratio=0.25, drop=0., drop_path=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), with_cp=False):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = CrossAttention(dim, num_heads, qkv_bias=False, attn_drop=drop, proj_drop=drop) # 32,2
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        if with_cffn:    # cffn_ratio==1
            self.ffn = Mlp(in_features=dim, hidden_features=int(dim * cffn_ratio), act_layer=nn.GELU, drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, query, feat):
        
        def _inner_forward(query, feat):
            
            attn = self.attn(self.query_norm(query), self.feat_norm(feat))
            query = query + attn
            
            query = self.drop_path(self.ffn(self.ffn_norm(query)))
            return query
        
        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)
        
        return query


class Injector(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0., with_cp=False, drop=0.):
        super().__init__()
        self.with_cp = with_cp
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = CrossAttention(dim, num_heads, qkv_bias=False, attn_drop=drop, proj_drop=drop)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
    
    def forward(self, query, feat):  # query=self.compress(c_tmp)  feat=self.compress(x_tmp)
        
        def _inner_forward(query, feat):
            
            attn = self.attn(self.query_norm(query), self.feat_norm(feat))
            return self.gamma * attn
        
        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)
        
        return query

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads    # 2
        head_dim = dim // num_heads   # 16
        self.scale = head_dim**-0.5   # 1/4

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)  # [1,90000,32]--->[1,90000,64]
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, c):  # c_tmp=[1,90000,32],x_tmp=[1,90000,32]   # x_temp ,c_temp
        B, N, C = x.shape
        kv = self.kv(c).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # [2,1,2,90000,16]
        k, v = kv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)  # [1,2,90000,16]

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # (1, 2, 90000, 16)

        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale  # 形状 (1, 2, 90000, 90000)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # 恢复形状 (1, 90000, 32)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 with_cffn=True, cffn_ratio=0.25, drop=0., drop_path=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), with_cp=False):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.attn = DMSA(dim, num_heads, dropout=drop)  # dim=32,num_heads=2
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        if with_cffn:
            self.ffn = Mlp(in_features=dim, hidden_features=int(dim * 2), act_layer=nn.GELU, drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, feat, points):
        # feat=compress(c_tmp):[1,90000,32];
        # points=compress(points_coors):[1,90000,3]
        
        def _inner_forward(feat, points):
            identity = feat
            feat = self.query_norm(feat)
            feat = self.attn(points, feat)
            feat = feat + identity
            
            feat = self.drop_path(self.ffn(self.ffn_norm(feat)))
            return feat
        
        # if self.with_cp and query.requires_grad:
        #     query = cp.checkpoint(_inner_forward, query, feat)
        # else:
        query = _inner_forward(feat, points)
        
        return query

class DMSA(nn.Module):
    def __init__(self, embed_dims=256, num_heads=8, dropout=0.1):  # embed_dims=32,num_heads=2,dropout=0
        super().__init__()
        self.attention = MultiheadAttention(embed_dims, num_heads, dropout, batch_first=True)  # embed_dims=32,num_heads=2,dropout=0
        self.beta = nn.Linear(embed_dims, num_heads)

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.beta.weight)
        nn.init.uniform_(self.beta.bias, 0.0, 2.0)

    def inner_forward(self, query_bbox, query_feat, pre_attn_mask):  #  query_bbox:[1,90000,3];query_feat:[1,90000,32]
        dist = self.center_dists(query_bbox)   # [1, ,90000,90000]
        beta = self.beta(query_feat)           # [1,90000,2]

        beta = beta.permute(0, 2, 1)           # [1,2,90000, ]
        attn_mask = dist[:, None, :, :] * beta[..., None]  # [1,2,90000,90000]
        if pre_attn_mask is not None:
            attn_mask[:, :, pre_attn_mask] = float('-inf')
        attn_mask = attn_mask.flatten(0, 1)
        return self.attention(query_feat, attn_mask=attn_mask)

    def forward(self, query_bbox, query_feat, pre_attn_mask=None):
        return self.inner_forward(query_bbox, query_feat, pre_attn_mask)

    @torch.no_grad()
    def center_dists(self, points):  # 计算每个 Batch 中点之间的欧氏距离
        centers = points[..., :2]   # 从 points 中提取 x, y 坐标[1,90000,2]
        dist = []  # 初始化一个空列表，用于存储每个批次的距离
        for b in range(centers.shape[0]):  # 遍历每个批次
            dist_b = torch.norm(centers[b].reshape(-1, 1, 2) - centers[b].reshape(1, -1, 2), dim=-1)  # 计算批次 b 的成对距离  [90000,1,2]-[1,90000,2]=[90000,90000]
            dist.append(dist_b[None, ...])  # 将批次距离添加到列表中[1,90000,90000]

        dist = torch.cat(dist, dim=0)  # 连接所有批次的距离
        dist = -dist  # 对距离取反

        return dist  # [1,90000,90000]


@BACKBONES.register_module()
class RadarBEVNet(nn.Module):
    def __init__(
        self,
        in_channels=4,                                      # in_channels=7
        feat_channels=(64,),                                # feat_channels=[32, 64]
        with_distance=False,
        voxel_size=(0.2, 0.2, 4),                           # [0.2, 0.2, 8]
        point_cloud_range=(0, -40, -3, 70.4, 40, 1),        # [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        norm_cfg=None,
        with_pos_embed=False,                               # True
        return_rcs=False,                                   # True
        drop=0.0,
    ):

        super().__init__()
        self.return_rcs = return_rcs
        assert len(feat_channels) > 0

        self.in_channels = in_channels
        in_channels = in_channels + 2  # 9
        self._with_distance = with_distance

        # Create PillarFeatureNet layers
        feat_channels = [in_channels] + list(feat_channels)  # [9,32,64]
        point_block = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i < len(feat_channels) - 2:
                last_layer = False
            else:
                last_layer = False
            point_block.append(
                RFNLayer(
                    in_filters, out_filters, norm_cfg=norm_cfg, last_layer=last_layer
                )
            )
        self.point_block = nn.ModuleList(point_block)

        num_heads = 2
        extractor = []
        for i in range(1, len(feat_channels)):
            extractor.append(
                Extractor(feat_channels[i], num_heads=num_heads, cffn_ratio=1,drop=drop, drop_path=drop)
            )
        self.extractor = nn.ModuleList(extractor)

        injector = []
        for i in range(1, len(feat_channels)):
            injector.append(
                Injector(feat_channels[i], num_heads=num_heads,drop=drop)
            )
        self.injector = nn.ModuleList(injector)

        transformer_block = []
        for i in range(1, len(feat_channels)):
            transformer_block.append(
                SelfAttentionBlock(feat_channels[i], num_heads=num_heads, cffn_ratio=1,drop=drop, drop_path=drop)
            )
        self.transformer_block = nn.ModuleList(transformer_block)

        linear_module = []
        for i in range(1, len(feat_channels)-1):
            linear_module.append(
                nn.Linear(feat_channels[i], feat_channels[i+1])
            )
        self.linear_module = nn.ModuleList(linear_module)

        self.out_linear = nn.Linear(feat_channels[-1]*2, feat_channels[-1])


        # Need pillar (voxel) size and x/y offset in order to calculate pillar offset
        # 为了计算支柱偏移，需要支柱（体素）尺寸和 x/y 偏移
        # 偏移量用于移动体素网格，以便网格的原点与点云正确对齐!!! 第一个体素的中心点作为坐标原点
        self.vx = voxel_size[0]    # 0.2
        self.vy = voxel_size[1]    # 0.2
        self.x_offset = self.vx / 2 + point_cloud_range[0]    # -51.1
        self.y_offset = self.vy / 2 + point_cloud_range[1]    # -51.1
        self.pc_range = point_cloud_range                     # [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

        if with_pos_embed:  # 位置编码配置
            embed_dims = feat_channels[1]  # 32
            self.pos_embed = nn.Sequential(
                        nn.Linear(3, embed_dims), 
                        nn.LayerNorm(embed_dims),
                        nn.ReLU(inplace=True),
                        nn.Linear(embed_dims, embed_dims),
                        nn.LayerNorm(embed_dims),
                        nn.ReLU(inplace=True),
                    )
        self.with_pos_embed = with_pos_embed
        
        self.point_embed = PointEmbed(in_channels+2, feat_channels[1])  # 11,32
    
    def compress(self, x):  # 把一个体素内所有雷达特征的最大值提取出来  [90000,10,32]---->[1,90000,32]
        x = x.max(dim=1)[0]
        x = x.unsqueeze(dim=0)
        return x

    def forward(self, features, num_voxels, coors):  # 输入voxels, num_points, coors
        dtype = features.dtype  # 数据类型
        f_center = torch.zeros_like(features[:, :, :2])  # 创建一个新的张量 f_center，这个新张量的形状和数据类型与 features[:, :, :2] 完全相同，但是它的所有元素都被初始化为 0，一般用作初始化坐标偏移量
        f_center[:, :, 0] = features[:, :, 0] - (
            coors[:, 1].to(dtype).unsqueeze(1) * self.vx + self.x_offset
        )   # 计算了每个点相对于其所在体素左下角的x偏移量，并将这些偏移量存储在 f_center 中
        f_center[:, :, 1] = features[:, :, 1] - (
            coors[:, 2].to(dtype).unsqueeze(1) * self.vy + self.y_offset
        )   # 计算了每个点相对于其所在体素左下角的y偏移量，并将这些偏移量存储在 f_center 中

        # normalize x,y,z to [0, 1] 把坐标值归一化到[0,1] （坐标值-起始值）/（终点值-起始值）
        features[:, :, 0:1] = (features[:, :, 0:1] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
        features[:, :, 1:2] = (features[:, :, 1:2] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
        features[:, :, 2:3] = (features[:, :, 2:3] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])

        voxel_count = features.shape[1]  # 每个体素雷达点数10（被max截取过或者max导致空的数据）
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)  # 生成数据掩码[N,10] true：实际数据 false:填充数据
        mask = torch.unsqueeze(mask, -1).type_as(features)  # 将维度拓展成features--->[N,10,1]
        
        features_mean = torch.zeros_like(features[:, :, :2])  # 创建一个新的张量 f_center，这个新张量的形状和数据类型与 features[:, :, :2] 完全相同，但是它的所有元素都被初始化为 0

        # features 的 x 坐标和 y 坐标进行均值偏移      mask 是布尔张量，会将无效点(false)的 x 坐标置为 0   x坐标-有效点x坐标均值
        features_mean[:, :, 0] = features[:, :, 0] - ((features[:, :, 0] * mask.squeeze()).sum(dim=1) / mask.squeeze().sum(dim=1)).unsqueeze(1)
        features_mean[:, :, 1] = features[:, :, 1] - ((features[:, :, 1] * mask.squeeze()).sum(dim=1) / mask.squeeze().sum(dim=1)).unsqueeze(1)

        rcs_features = features.clone()
        c = torch.cat([features, features_mean, f_center], dim=-1)   # 通道融合初始坐标，均值偏移后坐标，原点偏移后坐标 features[90000,10,7] features_mean[90000,10,2] f_center[90000,10,2]
        x = torch.cat([features, f_center], dim=-1)                   # 通道融合初始坐标，原点偏移后坐标

        # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.

        #清除无效填充点
        x *= mask  # [90000,10,9]
        c *= mask  # [90000,10,11]

        c = self.point_embed(c)  # 点嵌入
        if self.with_pos_embed:  # 位置嵌入
            c = c + self.pos_embed(features[:, :, 0:3])
        points_coors = features[:, :, 0:3].detach()  # 将张量从当前计算图中分离，生成一个不追踪梯度的新张量

        batch_size = coors[-1, 0] + 1  # 从 coors 张量中提取 batch size
        # bs_list：列表，记录每个批次（Batch）的体素数（如 [30000, 20000, 40000]）
        if batch_size>1:
            bs_list = [0]
            bs_info = coors[:, 0]
            pre = bs_info[0]
            for i in range(1, len(bs_info)):
                if pre != bs_info[i]:
                    bs_list.append(i)
                    pre = bs_info[i]
            bs_list.append(len(bs_info))
            bs_list = [bs_list[i+1]-bs_list[i] for i in range(len(bs_list)-1)]
        elif batch_size == 1:
            bs_list = [len(coors[:, 0])]
        else:
            assert False

        points_coors_split = torch.split(points_coors, bs_list)  # 按批次分割features[:, :, 0:3]即每个雷达点的坐标（x,y,z）

        i = 0

        # 逐层提取局部特征，输出维度逐步变化（9 → 32 → 64）
        for rfn in self.point_block:
            x = rfn(x)
            x_split = torch.split(x, bs_list)  # 按批次分割特征
            c_split = torch.split(c, bs_list)  # 按批次分割上下文特征
            
            x_out_list = []
            c_out_list = []
            for bs in range(len(x_split)):
                c_tmp = c_split[bs]   # 当前批次的上下文特征
                x_tmp = x_split[bs]   # 当前批次的上下文特征
                points_coors_tmp = points_coors_split[bs]  # 当前批次的坐标
                # 注入雷达特征到上下文特征
                c_tmp = c_tmp + self.injector[i](self.compress(c_tmp), self.compress(x_tmp)).transpose(1, 0).expand_as(c_tmp)
                # 提取上下文特征到雷达特征
                x_tmp = x_tmp + self.extractor[i](self.compress(x_tmp), self.compress(c_tmp)).transpose(1, 0).expand_as(x_tmp)
                c_tmp = self.transformer_block[i](self.compress(c_tmp), self.compress(points_coors_tmp)).transpose(1, 0).expand_as(c_tmp)
                if i < len(self.point_block)-1:    # i=0
                    c_tmp = self.linear_module[i](c_tmp)
                
                c_out_list.append(c_tmp)
                x_out_list.append(x_tmp)
            
            x = torch.cat(x_out_list, dim=0)
            c = torch.cat(c_out_list, dim=0)
            i += 1
        c = self.out_linear(torch.cat([c, x], dim=-1))

        c = torch.max(c, dim=1, keepdim=True)[0]
        if not self.return_rcs:
            return c.squeeze()
        else:
            rcs = (rcs_features*mask).sum(dim=1)/mask.sum(dim=1)
            return c.squeeze(), rcs.squeeze()  # 移除张量 c 和 rcs 中大小为 1 的维度

