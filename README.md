# RCBEVdet 算法复现 

graduation_project --- HDU wwwzh 

## Getting Started

### environment复现环境

代码在如下环境实现复现（只展示部分重要依赖库版本，详情可见requirements.txt）:

```
python                       3.8.20
cuda                         11.8
pytorch                      2.0.1+cu118
torchvision                  0.15.2+cu118
numpy                        1.23.4
mmcv-full                    1.6.0
mmcls                        0.25.0
mmdet                        2.28.2
mmsegmentation               0.25.0
nuscenes-devkit              1.1.11
av2                          0.2.1
detectron2                   0.6
```

如果在下载依赖软件包时遇到下载速度慢或超时的问题，你需要考虑先从镜像网站安装依赖包，然后再执行安装（setup.py里面没有直接写明库版本号）：
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install {Find the dependencies in setup.py:setup(install_requires=[...]) and write them down here} -i https://pypi.tuna.tsinghua.edu.cn/simple
python setup.py develop
```

个人复现代码最推荐的步骤是：

1. 通过 anaconda 创建对应 Python 版本的虚拟环境.安装与机器的 CUDA 版本相对应[PyTorch](https://pytorch.org/get-started/previous-versions/)； ex:`conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia`

2. 安装与 PyTorch 和 CUDA 版本相对应的 [mmcv](https://github.com/open-mmlab/mmcv)；推荐先安装官方集成的mim安装工具`pip install -U openmim`,再安装mmcv等固定版本的库

3. 安装 mmdet 的其他依赖项并安装 [mmdet](https://github.com/open-mmlab/mmdetection)；依赖项包括：mmsegmentation、mmcls等

4. 安装本项目的其他依赖项（ 请根据 requirements.txt 中的版本安装 ），然后安装本项目：`python setup.py develop`；

5. 手动编译一些运算符，生成特殊编译算子流程:
```bash
cd mmdet3d/ops/csrc
python setup.py build_ext --inplace
cd ../deformattn
python setup.py build install
```
6. 安装其余 detectron2 的依赖库并安装 [detectron2](https://github.com/facebookresearch/detectron2);因为算法代码使用的 detectron2 的版本较老，需要通过下载源码，从而进行源码编译环境.
具体操作可参考如下：
```
1. 从官网或者直接git下载detectron2-0.6到本地文件夹。
2. 使用detectron2-0.6作为目标文件夹，并进入文件夹。
3. 在该文件下地址下进入创建的虚拟环境（ex:rcbev），安装该项目：`python -m pip install -e detectron2-0.6`
4. 通过conda list可以发现在该虚拟环境下存在detectron2-0.6库
```

### 数据集准备
可以参考以下链接准备nuscenes数据集（略有不同）：[Fast-BEV代码复现实践](https://blog.csdn.net/h904798869/article/details/130317240)
如果您的文件夹结构与以下不同，可能需要更改配置文件中的相应路径。

```
├── mmdet3d
├── tools
├── configs
├── data
│   ├── nuscenes
│   │   ├── maps
│   │   │   ├── basemap
│   │   │   ├── expansion
│   │   │   ├── prediction
│   │   │   ├── *.png
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
```

对于 RCBEVDet 算法而言, 准备 nuscenes 数据通过运行：
```bash
python tools/create_data_nuscenes_RC.py
```


### training训练指令
$config_path：配置文件的地址，包含模型架构、配置指令等 ex:configs/rcbevdet/rcbevdet-256x704-r50-BEV128-9kf-depth-cbgs12e-circlelarger.py

$gpus：调用GPU的个数 ex:1、2或者更多
```bash
./tools/dist_train.sh $config_path $gpus
```

### testing测试推理指令

testing on validation set：在验证集上进行测试

```bash
./tools/dist_test.sh $config_path $checkpoint_path $gpus --eval bbox
```

testing on test set：在测试集上测试

$checkpoint_path：权重参数文件的地址
```bash
./tools/dist_test.sh $config_path $checkpoint_path $gpus --format-only --eval-options 'jsonfile_prefix=work_dirs'
mv work_dirs/pts_bbox/results_nusc.json work_dirs/pts_bbox/{$name}.json
```

benchmarking test latency：基准测试延迟

```bash
python tools/analysis_tools/benchmark_sequential.py $config 1 --fuse-conv-bn
```

TTA or Ensemble

```bash
python tools/merge_result_json.py --paths a.json,b.json,c.json
mv work_dirs/ens/results.json work_dirs/ens/{$name}.json
```

json文件可视化生成mp4格式文件
```bash
python tools/analysis_tools/vis.py ./work_dirs/pts_bbox/{$name}.json
```

如果您有任何其他问题，请参阅 
<a href='https://mmdetection3d.readthedocs.io/en/v1.0.0rc1/'>mmdet3d docs</a>.

