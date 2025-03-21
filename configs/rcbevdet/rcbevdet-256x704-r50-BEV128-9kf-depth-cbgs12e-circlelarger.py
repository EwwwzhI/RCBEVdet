# 继承原始配置文件 一般有4个基本组件类型，分别是数据集(datasets)、模型(models)、训练策略(schedules)和运行时的默认配置(default_runtime)
# rcbevdet模型:数据集配置文件继承自'../_base_/datasets/nus-3d.py';默认配置文件继承自'../_base_/default_runtime.py'
_base_ = ['../_base_/datasets/nus-3d.py', '../_base_/default_runtime.py']

# Global
# If point cloud range is changed, the models should also change their point
# cloud range accordingly 点云范围
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]  # 定义检测区域的三维边界

# 雷达体素化参数
radar_voxel_size = [0.2, 0.2, 8]  # 体素分辨率 (X/Y/Z轴)
# 使用雷达数据的维度索引（坐标、速度、RCS等）雷达点的特征维度
# x y z vx_comp vy_comp rcs timestamp
radar_use_dims = [0, 1, 2, 8, 9, 5, 18]

# For nuScenes we usually do 10-class detection
# 自动驾驶nuScenes数据集一般做10分类检测
# 类别标签：
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

# 数据配置格式：定义多摄像头输入参数及数据增强策略，提升模型鲁棒性
data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],  # 摄像头名称【左前、前、右前、左后、后、右后】
    'Ncams': 6,  # 摄像头数量
    'input_size': (256, 704),  # 模型输入尺寸
    'src_size': (900, 1600),   # 原始图像尺寸

    # Augmentation 数据增强参数
    'resize': (-0.06, 0.11),  # 随机缩放范围
    'rot': (-5.4, 5.4),       # 随机旋转角度范围（单位：度）
    'flip': True,             # 是否启用随机水平翻转
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

# Model
# BEV网格配置：将3D空间划分为规则的BEV网格，用于特征投影和检测
grid_config = {
    'x': [-51.2, 51.2, 0.8],  # X轴范围及分辨率
    'y': [-51.2, 51.2, 0.8],  # Y轴范围及分辨率
    'z': [-5, 3, 8],          # Z轴范围及分辨率
    'depth': [1.0, 60.0, 0.5],  # 深度范围及分辨率
}

# 体素尺寸 [0.1, 0.1, 0.2]
voxel_size = [0.1, 0.1, 0.2]
# 特征图下采样倍数
out_size_factor = 8
# 输出BEV特征图通道数
numC_Trans = 80

multi_adj_frame_id_cfg = (1, 8+1, 1)  # -->生成序列 [1, 2, 3, 4, 5, 6, 7, 8] 相邻帧的数量（时序融合）

# 模型架构 继承: BEVDepth4D_RC <--- BEVDet4D_RC <--- BEVDet_RC <--- CenterPoint <--- MVXTwoStageDetector <--- Base3DDetector <--- Base3DDetector(集成在mmdet库里面)
model = dict(
    type='BEVDepth4D_RC',  # 模型类型：BEVDepth4D_RC---多模态（雷达+摄像头）+时序的BEV检测  模型架构查看：RCBEVdet/mmdet3d/models/detectors/bevdet_rc.py
    freeze_img=True,       # 冻结图像主干网络（减少计算量）---继承自 BEVDet_RC 模型
    freeze_radar=False,    # 不冻结雷达分支（允许雷达特征学习）---继承自 BEVDet_RC 模型
    align_after_view_transfromation=False,  # 在视角变换（View Transformation）后拒绝对特征进行对齐---继承自 BEVDet4D_RC 模型
    num_adj=len(range(*multi_adj_frame_id_cfg)),  # 使用考虑相邻帧的数量（时序数据融合）---继承自 BEVDet4D_RC 模型

    # 图像处理架构
    img_backbone=dict(       # 模型图像骨干网络架构  ---继承自CenterPoint 模型
        pretrained='torchvision://resnet50',  # 模型预训练路径
        type='ResNet',       # 图像骨干网络模型：ResNet --->提取图像特征  模型架构查看:mmdetction-2.28.2/mmdet/model/backbones/ResNet.py
        depth=50,            # 骨干网络深度
        num_stages=4,        # ResNet网络阶段（残差层）数
        out_indices=(2, 3),  # 网络输出来自 2、3 阶段【4阶段数为：0，1，2，3】
        frozen_stages=-1,    # 要冻结的阶段（停止梯度并设置为评估模式） -1：表示不冻结任何参数
        norm_cfg=dict(type='BN', requires_grad=True),  # 用于构建和配置归一化层的字典
        norm_eval=False,     # 是否将归一化层设置为评估模式，即冻结运行统计（均值和方差）
        with_cp=False,       # 是否使用检查点，使用检查点会节省一些内存，但会降低训练速度
        style='pytorch'),    # 风格：`pytorch` 或 `caffe`，若设置为 "pytorch"，则步幅为 2 的层为 3x3 卷积层，否则步幅为 2 的层为第一个 1x1 卷积层
    img_neck=dict(           # 模型图像颈部架构  ---继承自CenterPoint 模型
        type='CustomFPN',    # 图像颈部模型：CustomFPN --->融合多尺度特征  模型架构查看:RCBEVdet/mmdet3d/model/necks/fpn.py
        in_channels=[1024, 2048],  # 每个尺度（或层级）的输入通道数，来自ResNet骨干网络的特征图在特定层级（该模型是2、3两层级）上的通道数
        out_channels=512,    # 每层特征图的输出通道数
        num_outs=1,          # FPN将产生的输出尺度（或层级）的数量
        start_level=0,       # FPN开始构建特征金字塔的输入骨干网络的起始层级索引，值为0表示从骨干网络的第一层开始
        out_ids=[0]),        # 输出ID列表，指定FPN输出层
    img_view_transformer=dict(                 # 模型图像转bev特征架构  ---继承自 BEVDet_RC 模型
        type='LSSViewTransformerBEVDepth',     # 模型图像转换架构，LSS算法投影图像到BEV camera_imgs--->鸟瞰图特征  2D->3D->压缩  模型架构查看:RCBEVdet/mmdet3d/models/necks/view_transformer.py
        grid_config=grid_config,               # BEV网格配置：将3D空间划分为规则的BEV网格，用于特征投影和检测
        input_size=data_config['input_size'],  # 输入特征图尺寸
        in_channels=512,                       # 输入特征图通道数
        out_channels=numC_Trans,               # 输出BEV特征图通道数：80
        depthnet_cfg=dict(use_dcn=False, aspp_mid_channels=96),  # 深度推断配置：不使用可变形卷积；ASPP中间通道数为96
        downsample=16),                        # 下采样因子 = input_size/特征图size(CustomFPN输出特征图尺寸)
    img_bev_encoder_backbone=dict(             # 模型bev特征提取架构  ---继承自 BEVDet_RC 模型
        type='CustomResNet',                   # 模型BEV特征提取架构，用于BEV特征的高层次特征提取  模型架构查看:RCBEVdet/mmdet3d/models/backbones/resnet.py
        numC_input=numC_Trans * (len(range(*multi_adj_frame_id_cfg))+1),   # 总输入通道数为 80 * (8+1) = 720（当前帧 + 8个相邻帧）时序融合---将当前帧与8个相邻帧的BEV特征拼接（通道维度）
        num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8]),    # 主干网络三个阶段的输出通道数（80*2, 80*4, 80*8），提取多尺度特征
    img_bev_encoder_neck=dict(                          # 模型bev特征颈部架构  ---继承自 BEVDet_RC 模型
        type='FPN_LSS',                                 # 改进的特征金字塔网络，用于多尺度特征融合  模型架构查看:mmdet3d/models/necks/lss_fpn.py
        in_channels=numC_Trans * 8 + numC_Trans * 2,    # 输入通道数，由640（主干第三阶段输出） + 160（主干第一阶段输出）组成
        out_channels=256),                              # 输出通道数，统一特征维度以适配检测头
    pre_process=dict(                           # 模型时序融合预处理架构  ---继承自 BEVDet4D_RC 模型   img_bev_encoder_backbone的前置工作
        type='CustomResNet',                    # 轻量级预处理网络  模型架构查看:RCBEVdet/mmdet3d/models/backbones/resnet.py
        numC_input=numC_Trans,                  # 输入通道数（单帧BEV特征图通道数）
        num_layer=[2,],                         # 包含第1个残差层两个残差快
        num_channels=[numC_Trans,],             # 每层输出通道保持80，不增加维度
        stride=[1,],                            # 不进行下采样，保留空间分辨率
        backbone_output_ids=[0,]),              # 输出层索引号
    

    #radar start 雷达处理架构
    radar_voxel_layer=dict(                     # 将原始点云转换为体素化表示 ---继承自 BEVDet_RC 模型
        max_num_points=10,                      # 单个体素（voxel）内允许包含的最大雷达点数
        voxel_size=radar_voxel_size,            # [0.2, 0.2, 8]  X和Y方向每个体素宽0.2米，Z方向高8米 X/Y轴高分辨率（0.2米）捕捉细节，Z轴大跨度（8米）适应雷达垂直稀疏性
        max_voxels=(90000, 120000),             # 训练时（第一个值）：限制为90,000个体素，防止显存溢出;测试时（第二个值）：放宽至120,000个体素，以保留更多信息
        point_cloud_range=point_cloud_range),   # [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]  雷达点云的有效检测范围 体素尺寸(512x512x1)
    radar_voxel_encoder=dict(                   # 雷达体素编码 ---继承自 BEVDet_RC 模型
        type='RadarBEVNet',                     # 模型雷达点云转换架构，投影点云到BEV 模型架构查看:RCBEVdet/mmdet3d/models/backbones/radar_encoder.py
        return_rcs=True,                        # 模型输出雷达截面积（RCS）特征
        in_channels=6+1,                        # 输入特征通道：7 [x, y, z, vx_comp, vy_comp, rcs, timestamp]
        feat_channels=[32, 64],                 # 特征提取层的通道数
        with_distance=False,                    # 不在特征提取中使用距离信息
        point_cloud_range=point_cloud_range,    # [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]  雷达点云的有效检测范围
        voxel_size=radar_voxel_size,            # [0.2, 0.2, 8]  X和Y方向每个体素宽0.2米，Z方向高8米 X/Y轴高分辨率（0.2米）捕捉细节，Z轴大跨度（8米）适应雷达垂直稀疏性
        norm_cfg=dict(                          # 归一层配置
            type='BN1d',                        # 使用一维批归一化
            eps=1.0e-3,                         # 归一化过程中用于数值稳定性的小常数
            momentum=0.01),                     # 归一化过程中用于计算运行均值和方差的动量参数
        with_pos_embed=True                     # 是否使用位置编码
    ),
    radar_middle_encoder=dict(                  # 将雷达体素特征转换为 BEV 伪图像 ---继承自 BEVDet_RC 模型
        type='PointPillarsScatterRCS',          # 改进的雷达体素散射模块 模型架构查看:RCBEVdet/mmdet3d/models/middle_encoders/pillar_scatter.py
        in_channels=64,                         # 输入体素特征通道：64
        output_shape=[512, 512],                # 输出伪图像的尺寸（高度 × 宽度），对应 BEV 空间分辨率
    ),

    radar_bev_backbone=dict(                    # 雷达bev提取特征骨干网络架构  ---继承自 BEVDet_RC 模型
        type='SECOND',                          # 模型雷达bev特征提取 模型架构查看:RCBEVdet/mmdet3d/models/backbones/second.py
        in_channels=64,                         # 输入特征通道：64
        out_channels=[64, 128, 256],            # 输出特征通道：[64,128,256]
        layer_nums=[3, 5, 5],                   # 每个卷积层包含卷积块的个数[3，5，5]
        layer_strides=[2, 2, 2],                # 每个卷积层第一个卷积块的步幅[3，5，5]
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),  # 归一层配置
        conv_cfg=dict(type='Conv2d', bias=False)),  # 卷积层配置
    radar_bev_neck=dict(                        # 雷达bev特征颈部架构  ---继承自 BEVDet_RC 模型
        type='SECONDFPN',                       # 雷达bev特征金字塔网络 模型架构查看:RCBEVdet/mmdet3d/models/necks/second_fpn.py
        in_channels=[64, 128, 256],             # bev特征输入通道数，64,128,256
        out_channels=[128, 128, 128],           # bev特征输出通道数，128,128,128
        upsample_strides=[0.5, 1, 2],           # 上采样步幅[0.5，1，2]
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),  # 归一层配置
        upsample_cfg=dict(type='deconv', bias=False),       # 上采样配置
        use_conv_for_no_stride=True),                       # 确保所有层级的特征均经过通道调整（如 64→128），维持网络结构一致性。
    rac=sum([128, 128, 128]),
    bev_size=128,

    #radar end

    pts_bbox_head=dict(     # bev特征检测头架构  ---继承自 MVXTwoStageDetector 模型
        type='CenterHead',  # 检测头模型 模型架构查看:RCBEVdet/mmdet3d/models/dense_heads/centerpoint_head.py
        in_channels=256,    # 输入BEV特征的通道数
        tasks=[             # 检测类别数（nuScenes数据集共10类）
            dict(num_class=10, class_names=['car', 'truck',
                                            'construction_vehicle',
                                            'bus', 'trailer',
                                            'barrier',
                                            'motorcycle', 'bicycle',
                                            'pedestrian', 'traffic_cone']),
        ],
        common_heads=dict(                       # 共享头部 中心点偏移回归,高度预测,尺寸预测,旋转角,速度预测
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(                        # 边界框编码
            type='CenterPointBBoxCoder',
            pc_range=point_cloud_range[:2],     # BEV范围 [-51.2, -51.2, 51.2, 51.2]
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],  # 后处理时保留预测框的范围
            max_num=500,                        # 最大保留预测框数量
            score_threshold=0.1,                # 置信度阈值（过滤低分预测）
            out_size_factor=8,                  # BEV特征图下采样倍数（与输入分辨率相关）
            voxel_size=voxel_size[:2],          # 体素尺寸 [0.1, 0.1]
            code_size=9),                       # 边界框编码维度（中心点xy、高度z、尺寸lwh、旋转角θ、速度vxvy）
        separate_head=dict(                     # 独立头部 初始化分类分支偏置（控制初始预测概率接近0.1，避免梯度爆炸） 最终卷积层的核大小
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean', loss_weight=6.),  # 分类损失 高斯焦点损失 损失计算方式为均值 分类损失权重
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=1.5),           # 回归损失 L1损失（用于中心点、尺寸、旋转角等回归） 回归损失权重
        norm_bbox=True),
    # model training and testing settings
    train_cfg=dict(                                         # 训练配置
        pts=dict(
            point_cloud_range=point_cloud_range,            # 点云范围
            grid_size=[1024, 1024, 40],                     # BEV网格划分（X/Y/Z轴）
            voxel_size=voxel_size,                          # 体素尺寸 [0.1, 0.1, 0.2]
            out_size_factor=8,                              # 特征图下采样倍数
            dense_reg=1,                                    # 密集回归策略（1表示启用）
            gaussian_overlap=0.1,                           # 高斯分布重叠阈值（正样本分配）
            max_objs=500,                                   # 单帧最大目标数
            min_radius=2,                                   # 高斯核最小半径
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])),  # 各回归项的损失权重（共9项）
    test_cfg=dict(
        pts=dict(
            pc_range=point_cloud_range[:2],                  # 预测框范围限制
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],    # 后处理过滤范围
            max_per_img=500,                                 # 每帧最大输出框数
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,                            # 置信度阈值
            out_size_factor=8,                              # 特征图下采样倍数
            voxel_size=voxel_size[:2],                      # 体素尺寸 [0.1, 0.1, 0.2]
            pre_max_size=1000,
            post_max_size=500,

            # Scale-NMS
            nms_type=['rotate'],                             # 旋转NMS（处理旋转框重叠）
            nms_thr=[0.2],                                   # NMS阈值
            nms_rescale_factor=[[1.0, 0.7, 0.7, 0.4, 0.55,
                                 1.1, 1.0, 1.0, 1.5, 3.5]]
        )
    )
)

# Data
dataset_type = 'NuScenesDatasetRC'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

train_pipeline = [
    dict(
        type='PrepareImageInputs',              # 准备图像输入数据，处理多摄像头图像输入，并进行数据增强（如随机翻转、旋转、缩放等）
        is_train=True,                          # 标记当前为训练模式，启用数据增强
        data_config=data_config,                # 引用全局定义的 data_config，包含摄像头名称、输入尺寸、增强参数等
        sequential=True),                       # 按时间序列处理多帧数据（用于时序模型）
    dict( #load radar
        type='LoadRadarPointsMultiSweeps',      # 雷达多帧点云加载模块的配置参数 加载多帧雷达点云数据，合并当前帧与历史帧的雷达点
        load_dim=18,                            # 原始雷达数据中每个点的特征维度数
        sweeps_num=8,                           # 加载的雷达扫描帧数（包含当前帧和历史帧）
        use_dim=radar_use_dims,                 # 实际使用的特征维度索引或名称
        max_num=1200, ),                        # 单样本中允许加载的最大雷达点数
    dict(
        type='LoadAnnotationsBEVDepth',         # 加载BEV视角下的3D标注信息（如边界框、类别标签），并进行数据增强
        bda_aug_conf=bda_aug_conf,              # 全局数据增强配置（随机旋转、缩放、翻转等）
        classes=class_names),                   # 指定有效的类别名称（过滤非目标类别）

    dict(type='GlobalRotScaleTrans_radar'),     # 对雷达点云和3D标注同时应用全局旋转、缩放、平移数据增强，增强模型鲁棒性

    dict(
        type='LoadPointsFromFile',              # 从文件加载LiDAR点云数据（可选步骤，可能用于多模态融合）
        coord_type='LIDAR',                     # 点云坐标系类型（LiDAR坐标系）
        load_dim=5,                             # 加载的LiDAR点云维度（x, y, z, intensity, timestamp）
        use_dim=5,                              # 实际使用的维度
        file_client_args=file_client_args),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),  # 将LiDAR点云投影到多摄像头视角，生成对应的深度图（用于辅助图像特征学习）
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),        # 根据点云范围过滤3D标注框，移除超出检测区域的目标
    dict(type='ObjectNameFilter', classes=class_names),                         # 根据类别名称过滤标注，仅保留指定类别的目标（如NuScenes的10类）
    dict(type='DefaultFormatBundle3D', class_names=class_names),                # 将数据转换为统一的张量格式，并添加必要的元数据（如标注框类型）
    dict(                                                                       # 收集所有处理后的数据项，过滤无关键值，构建最终输入字典
        type='Collect3D', keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d',
                                'gt_depth','radar'])
]

test_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config, sequential=True),  # 加载多摄像头图像，并进行基础预处理（如尺寸调整、归一化）
    dict(
        type='LoadRadarPointsMultiSweeps',
        load_dim=18,                                                            # 原始雷达点特征维度（x, y, z, vx, vy, rcs 等）
        sweeps_num=8,                                                           # 合并当前帧与历史 7 帧的雷达点云（共8帧）
        use_dim=radar_use_dims,                                                 # 实际使用的特征维度索引 [0,1,2,8,9,5,18]（坐标、速度、RCS等）
        max_num=1200, ),                                                        # 单帧最大雷达点数（防止显存溢出）
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,                                              # 全局数据增强配置（测试时可能不启用随机性）
        classes=class_names,                                                    # 指定有效类别（过滤非目标类别）
        is_train=False),                                                        # 关闭训练模式（不应用随机增强）

    dict(type='GlobalRotScaleTrans_radar'),                                     # 对雷达点云应用 固定的全局旋转、缩放、平移，与图像/BEV空间对齐

    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),        # 移除超出检测区域的标注框，确保模型仅预测有效范围内的目标
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',                                                     # 坐标系类型（LiDAR坐标系）
        load_dim=5,                                                             # 加载的LiDAR点云维度（x, y, z, intensity, timestamp）
        use_dim=5,                                                              # 实际使用的维度（与训练一致）
        file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),                                                  # 图像输入尺寸（未使用，仅占位符）
        pts_scale_ratio=1,                                                      # 点云缩放比例（测试时不增强）
        flip=False,                                                             # 禁用水平翻转（避免随机性）
        transforms=[
            dict(
                type='DefaultFormatBundle3D',                                   # 数据标准化（转为张量）
                class_names=class_names,
                with_label=False),                                              # 测试时不加载标注
            dict(type='Collect3D', keys=['points', 'img_inputs','radar'])       # 收集最终输入数据
        ])
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=True,
    use_map=False,
    use_external=False)

share_data_config = dict(
    type=dataset_type,
    classes=class_names,
    modality=input_modality,
    img_info_prototype='bevdet4d',
    multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
)

test_data_config = dict(
    pipeline=test_pipeline,
    ann_file=data_root + 'nuscenes_RC_infos_val.pkl')

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type='CBGSDataset',
        dataset=dict(
        data_root=data_root,
        ann_file=data_root + 'nuscenes_RC_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR')),
    val=test_data_config,
    test=test_data_config)

for key in ['val', 'test']:
    data[key].update(share_data_config)
data['train']['dataset'].update(share_data_config)

# Optimizer
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=1e-2) 
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='cyclic',
    target_ratio=(10, 0.0001),
    cyclic_times=1,
    step_ratio_up=0.4)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.8947368421052632, 1),
    cyclic_times=1,
    step_ratio_up=0.4)

runner = dict(type='EpochBasedRunner', max_epochs=12)
evaluation = dict(interval=4)

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
    dict(
        type='SequentialControlHook', 
        temporal_start_epoch=-1,
    ),
]
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

# fp16 = dict(loss_scale='dynamic')
# load_from='checkpoint/det-256x704-r50-BEV128-9kf-depth-hop.pth'（添加检查点：指基于该预训练的权重进行训练）
#checkpoint_config = dict(interval=6)

