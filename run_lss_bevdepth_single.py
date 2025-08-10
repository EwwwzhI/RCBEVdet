# run_lss_bevdepth.py
import torch
import os
import time
from mmdet3d.models.necks.view_transformer import LSSViewTransformerBEVDepth

def load_input_data(input_path):
    """加载保存的输入数据"""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件 {input_path} 不存在")
    
    data = torch.load(input_path)
    input_data = data["input_data"]
    input_data = input_data[0]
    
    # 验证输入结构
    if len(input_data) != 8:
        raise ValueError("输入数据应包含8个参数，实际检测到{}个".format(len(input_data)))
    
    return tuple(input_data)

def initialize_model(config):
    """初始化LSSViewTransformerBEVDepth模型"""
    model = LSSViewTransformerBEVDepth(
        grid_config=config['grid_config'],
        input_size=config['input_size'],
        downsample=config['downsample'],
        in_channels=config['in_channels'],
        out_channels=config['out_channels'],
        loss_depth_weight=config.get('loss_depth_weight', 3.0),
        depthnet_cfg=config.get('depthnet_cfg', {}),
        with_cp=config.get('with_cp', False),
        Depthwise_Separable=config.get('Depthwise_Separable', False)
    )
    
    model.eval()
    return model

def main():
    # 配置参数（需与实际训练配置一致）
    config = {
        'grid_config': {
            'x': [-51.2, 51.2, 0.8],
            'y': [-51.2, 51.2, 0.8],
            'z': [-5, 3, 8],
            'depth': [1.0, 60.0, 0.5]
        },
        'input_size': (256, 704),
        'downsample': 16,
        'in_channels': 512,
        'out_channels': 80,
        'checkpoint_path': None,#"bev_inputs/weights.pth",  # 可选预训练权重路径
        'Depthwise_Separable':True
    }
    data_path="input_1.pth"
    # 加载输入数据
    try:
        input_tuple = load_input_data(f"bev_inputs/{str(data_path)}")
    except Exception as e:
        print(f"加载输入数据失败: {str(e)}")
        return

    # 初始化模型
    model = initialize_model(config)
    
    depth_net_params = sum(p.numel() for p in model.depth_net.parameters())
    

    # 运行前向传播
    with torch.no_grad():
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            inputs = [tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor 
                      for tensor in input_tuple]
            
            # ========== 时间测试代码 ========== #
            # 预热（排除第一次运行的初始化时间）
            for _ in range(2):
                _ = model(inputs)
            
            # CUDA同步（确保准确计时）
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # 初始化时间统计变量
            total_depthnet_time = 0.0
            total_view_transform_time = 0.0
            total_forward_time = 0.0
            num_runs = 1000  # 测试次数
            
            for _ in range(num_runs):
                start_forward = time.perf_counter()
                
                # DepthNet计时
                start_depthnet = time.perf_counter()
                B, N, C, H, W = inputs[0].shape
                x = inputs[0].view(B * N, C, H, W)
                mlp_input = model.get_mlp_input(*inputs[1:7])
                if model.with_cp:
                    x = checkpoint(model.depth_net, x, mlp_input, None)
                else:
                    x = model.depth_net(x, mlp_input, None)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_depthnet = time.perf_counter()
                
                # ViewTransform计时
                start_view_transform = time.perf_counter()
                depth_digit = x[:, :model.D, ...]
                tran_feat = x[:, model.D:model.D + model.out_channels, ...]
                depth = depth_digit.softmax(dim=1)
                bev_feat, depth = model.view_transform(inputs, depth, tran_feat)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_view_transform = time.perf_counter()
                
                # 总时间
                end_forward = time.perf_counter()
                
                # 累积时间
                total_depthnet_time += (end_depthnet - start_depthnet) * 1000
                total_view_transform_time += (end_view_transform - start_view_transform) * 1000
                total_forward_time += (end_forward - start_forward) * 1000

            # 保存输出结果
            os.makedirs("bev_outputs", exist_ok=True)
            torch.save({
                'bev_feature': bev_feat.cpu(),
                'depth_pred': depth.cpu(),
                'time_metrics': {
                    'depthnet_avg_ms': total_depthnet_time / num_runs,
                    'view_transform_avg_ms': total_view_transform_time / num_runs,
                    'total_forward_avg_ms': total_forward_time / num_runs,
                }
            }, f"bev_outputs/{str(data_path)}")

            print(f"推理成功完成！输出已保存至 bev_outputs/{str(data_path)}")
            print(f"DepthNet平均耗时: {total_depthnet_time/num_runs:.2f}ms")
            print(f"ViewTransform平均耗时: {total_view_transform_time/num_runs:.2f}ms")
            print(f"总前向传播平均耗时: {total_forward_time/num_runs:.2f}ms")
            print(f"DepthNet参数量: {depth_net_params}")

        except RuntimeError as e:
            print(f"运行时错误: {str(e)}")
            if "CUDA out of memory" in str(e):
                print("显存不足，请尝试：\n1. 减小批量大小\n2. 使用CPU模式")

if __name__ == "__main__":
    main()