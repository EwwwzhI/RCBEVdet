# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import warnings
warnings.filterwarnings('ignore') # warning太多了。。。

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

import mmdet
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor

if mmdet.__version__ > '2.23.0':
    # If mmdet version > 2.23.0, setup_multi_processes would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import setup_multi_processes
else:
    from mmdet3d.utils import setup_multi_processes

try:
    # If mmdet version > 2.23.0, compat_cfg would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg

import torch.onnx
from mmdet3d.models.necks.view_transformer import LSSViewTransformerBEVDepth
import time
from collections import defaultdict

class TimeRecorder:
    def __init__(self, target_types=None):
        self.times = defaultdict(float)
        self.counts = defaultdict(int)
        self.hooks = []
        self.target_types = target_types or []
        
    def _record_time(self, module, input, output):
        if module.__class__.__name__ not in self.target_types:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = (time.time() - self.start_time) * 1000  # 单位：毫秒
        module_name = module.__class__.__name__
        self.times[module_name] += elapsed
        self.counts[module_name] += 1
        
    def _pre_hook(self, module, input):
        if module.__class__.__name__ not in self.target_types:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.time()
        
    def register_hooks(self, model):
        """仅注册目标模块的钩子"""
        for _, module in model.named_modules():
            if module.__class__.__name__ in self.target_types:
                pre_hook = module.register_forward_pre_hook(self._pre_hook)
                post_hook = module.register_forward_hook(self._record_time)
                self.hooks.append((pre_hook, post_hook))
            
    def remove_hooks(self):
        for pre_hook, post_hook in self.hooks:
            pre_hook.remove()
            post_hook.remove()
            
    def get_summary_and_save_to_csv(self, filename="module_timing.csv"):
        """将统计结果保存为CSV文件"""
        import csv
        summary = []
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Module", "Avg Time (ms)", "Call Count"])
            for name in self.times:
                avg_time = self.times[name] / self.counts[name]
                writer.writerow([name, f"{avg_time:.3f}", self.counts[name]])
                summary.append(f"{name}: {avg_time:.3f}ms per call (called {self.counts[name]} times)")  # 单位改为ms
        print(f"统计结果已保存至 {filename}")
        return "\n".join(summary)

class InputSaver:
    def __init__(self, save_dir="bev_inputs"):
        self.save_dir = save_dir
        self.index = 0
        os.makedirs(self.save_dir, exist_ok=True)

    def _save_input_hook(self, module, input):
        # 示例：手动生成符合要求的输入数据
        # input_data = (
        #     torch.randn(1, 6, 512, 16, 44),  # x
        #     torch.randn(1, 6, 4, 4),         # rots
        #     torch.randn(1, 6, 4, 4),         # trans
        #     torch.randn(1, 6, 3, 3),         # intrins
        #     torch.randn(1, 6, 3, 3),         # post_rots
        #     torch.randn(1, 6, 3),            # post_trans
        #     torch.randn(1, 3, 3),            # bda
        #     torch.randn(1, 6, 27)            # mlp_input
        # )

        # torch.save({'input_data': input_data}, "bev_inputs/randn_input.pth")
        try:
            def _to_tensor(data):
                if isinstance(data, torch.Tensor):
                    return data.detach().cpu()
                elif isinstance(data, (list, tuple)):
                    return type(data)([_to_tensor(x) for x in data])
                elif isinstance(data, dict):
                    return {k: _to_tensor(v) for k, v in data.items()}
                else:
                    return torch.tensor(data)  # 强制转换非张量数据

            # input是包含多个参数的元组，需逐个处理
            processed_input = tuple([_to_tensor(elem) for elem in input])
            
            save_path = os.path.join(self.save_dir, f"input_{self.index}.pth")
            self.index += 1

            save_dict = {
                'input_structure': str(type(input)),  # 记录原始类型
                'input_data': processed_input  # 保存为元组
            }
            torch.save(save_dict, save_path)
            print(f"成功保存输入数据（包含{len(input)}个参数）至 {save_path}")
        except Exception as e:
            print(f"保存失败: {str(e)}")

def get_target_modules_from_config(cfg_model):
    """提取配置中目标模块的type字段"""
    target_modules = set()
    
    # 定义需要监控的模块键名
    target_keys = [
        'img_backbone', 'img_neck', 'img_view_transformer',
        'img_bev_encoder_backbone', 'img_bev_encoder_neck', 'pre_process',
        'radar_voxel_encoder', 'radar_middle_encoder',
        'radar_bev_backbone', 'radar_bev_neck', 'pts_bbox_head'
    ]
    
    # 遍历目标键名
    for key in target_keys:
        if key in cfg_model:
            module_cfg = cfg_model[key]
            if isinstance(module_cfg, dict) and 'type' in module_cfg:
                target_modules.add(module_cfg['type'])
    
    return list(target_modules)

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--no-aavt',
        action='store_true',
        help='Do not align after view transformer.')
    parser.add_argument(
        '--aavt',
        action='store_true',
        help='Do not align after view transformer.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args

def main():
    # 解析命令行参数
    args = parse_args()

    # 断言至少指定一个操作（保存/评估/格式化/显示结果/保存结果）的参数
    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg = compat_cfg(cfg)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    test_num=5                                                        # 修改后的代码：截断数据集
    dataset = build_dataset(cfg.data.test)
    dataset.data_infos = dataset.data_infos[:test_num]                  # 仅保留前面test_num个样本
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    if args.no_aavt:
        if '4D' in cfg.model.type:
            cfg.model.align_after_view_transfromation=False
    elif args.aavt:
        if '4D' in cfg.model.type:
            cfg.model.align_after_view_transfromation=True
    else:  # default: align_after_view_transfromation=False
        if '4D' in cfg.model.type:
            cfg.model.align_after_view_transfromation=False
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))


    target_modules = get_target_modules_from_config(cfg.model)          # 获取配置中定义的目标模块类型

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    model.eval()# 关键：关闭Dropout和BatchNorm的训练行为
    
    # 初始化时间记录器（仅监控目标模块）
    time_recorder = TimeRecorder(target_types=target_modules)
    time_recorder.register_hooks(model)
    # ========== 统计模型参数量 ========== #
    def format_params(num_params):
        if num_params >= 1e6:
            return f"{num_params / 1e6:.2f}M"
        elif num_params >= 1e3:
            return f"{num_params / 1e3:.2f}K"
        else:
            return f"{num_params}"

    def format_size(bytes_size):
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_size < 1024:
                return f"{bytes_size:.2f} {unit}"
            bytes_size /= 1024
        return f"{bytes_size:.2f} TB"

    # 统计总参数量和可训练参数量和缓冲区
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_buffers = sum(b.numel() for b in model.buffers())

    # 获取模型数据类型
    try:
        dtype = next(model.parameters()).dtype
        bytes_per_element = 2 if dtype == torch.float16 else 4  # FP16=2字节，默认FP32=4字节
    except StopIteration:  # 模型无参数的情况
        bytes_per_element = 4

    # 计算内存占用
    model_memory = (total_params + total_buffers) * bytes_per_element

    print(f"[Model Summary] Total Parameters: {format_params(total_params)}")
    print(f"[Model Summary] Trainable Parameters: {format_params(trainable_params)}")
    print(f"[Model Memory] Estimated Memory Usage: {format_size(model_memory)}")
    # ========== 新增代码结束 ========== #

    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE

    if not distributed:
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)
        # 禁用梯度:
        with torch.no_grad():
            outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        
        # ========== 注册输入保存钩子（修改后） ========== #
        # input_saver = InputSaver()  # 初始化时指定保存目录
        # actual_model = model.module
        # # 注册钩子到目标模块
        # for name, module in actual_model.named_modules():
        #     if isinstance(module, LSSViewTransformerBEVDepth):
        #         # 注册前向预处理钩子
        #         module.register_forward_pre_hook(input_saver._save_input_hook)
        #         print(f"已在 {name} 注册输入保存钩子")
        # ========== 注册输入保存钩子（必须在此处！） ========== #
        # 禁用梯度:
        with torch.no_grad():
            outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)
    
    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        # ========== 打印层时间统计+保存为CSV========== #
        print("\n===== Layer Timing Summary =====")
        print(time_recorder.get_summary_and_save_to_csv())
        time_recorder.remove_hooks()  # 移除钩子
        # ========== 新增代码结束 ========== #

        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == '__main__':
    main()
