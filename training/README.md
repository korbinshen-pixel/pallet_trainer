# 托盘位姿估计训练框架

## 上活盘位姿估计

基于 RGB 图像 (8-bit PNG) + 深度图像 (16-bit PNG) 使用大视野 CNN 学习托盘的位置（三维坐标）和姿态（四元数旋转）。

## 项目结构

```
training/
├── config.py              # 配置文件
├── dataset.py             # 数据加载器 (RGB 8-bit + Depth 16-bit PNG)
├── model.py               # 模型定义 (ResNet50 双分支)
├── train.py               # 训练脚本
├── eval.py                # 评估脚本
├── infer.py               # 推理脚本
├── checkpoints/           # 检查点保存目录
├── models/                # 模型保存目录
├── logs/                  # 日志目录 (TensorBoard)
├── results/               # 结果保存目录
└── README.md              # 本文档
```

## 数据集执残模式

### 目录结构

```
~/pallet_dataset/
└── 20251210_162120/       # 改为你的数据集时间戳 (YYYYMMDD_HHMMSS)
    ├── rgb/                # RGB 图像目录 (8-bit PNG)
    │   ├── rgb_000000.png
    │   ├── rgb_000001.png
    │   └── ...
    ├── depth/              # 深度图像目录 (16-bit PNG)
    │   ├── depth_000000.png
    │   ├── depth_000001.png
    │   └── ...
    └── poses.txt          # 位姿标注文件
```

### poses.txt 格式

每行是一个托盘的位姿，格式如下：

```
frame_id x y z qx qy qz qw
000000 0.123 0.456 0.789 0.0 0.0 0.707 0.707
000001 0.124 0.457 0.790 0.001 0.001 0.707 0.707
...
```

- `frame_id`: 整数了，与图像文件名可搞应
- `x, y, z`: 位置 (m)
- `qx, qy, qz, qw`: 四元数 (袜绫旋转)

## 使用方法

### 1. 安装依赖

```bash
pip install torch torchvision scikit-learn pillow tqdm tensorboard
```

### 2. 配置数据集

编辑 `config.py`设置数据集路径：

```python
DATASET_ROOT = os.path.expanduser('~/pallet_dataset')
```

### 3. 训练

```bash
python train.py
```

训练会自动找最新的数据集目录。

**配置参数修改** (可选)：

```python
# config.py
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
WARMUP_EPOCHS = 5
PATIENCE = 20  # Early stopping
```

### 4. 评估

```bash
# 评估测试集
python eval.py --model models/best_model.pth --split test --output results/metrics.json

# 评估你的数据集
python eval.py --model models/best_model.pth --dataset_dir ~/pallet_dataset/20251210_162120 --split test
```

### 5. 推理 (inference)

**单个图像：**

```bash
python infer.py --model models/best_model.pth \
                --rgb_path test_rgb.png \
                --depth_path test_depth.png
```

**批量推理：**

```bash
python infer.py --model models/best_model.pth \
                --rgb_dir ~/pallet_dataset/20251210_162120/rgb \
                --depth_dir ~/pallet_dataset/20251210_162120/depth \
                --output results/predictions.txt
```

## 模型体系

### 架构

```
PalletPoseEstimator
├── RGB Branch (ResNet-50 ImageNet 预训练)
│   └── 2048D 特征
├── Depth Branch (ResNet-50, 未预训练)
│   └── 2048D 特征 (单通道在 3 通道中复制)
├── Feature Fusion
│   └── 4096D -> 1024D
├── Position Head
│   └── 3D 位置 (x, y, z)
└── Rotation Head
    └── 4D 四元数 (qx, qy, qz, qw)
```

### 损失函数

- **位置损失**: MSE Loss
- **旋转损失**: Cosine Distance Loss (四元数相似度)
- **总損失**: position_weight * pos_loss + rotation_weight * rot_loss

## 整合器配置

### 优化器

- Adam optimizer
- Learning rate: 1e-4
- Weight decay: 1e-5

### 学习率调度

- **预热（Warmup）**: 线性从 0.1 * LR 递优到 LR (5 epochs)
- **衰减（Cosine Annealing）**: 余弦衰减到 1e-6 (45 epochs)

## 结果文件

### TensorBoard 可视化

```bash
tensorboard --logdir logs/
```

打开浏览器中访问 `http://localhost:6006`

### 检查点

- `checkpoints/checkpoint_epoch_XXX.pth`: 每个 epoch 的检查点
- `models/best_model.pth`: 最优模型 (early stopping)

## 批量处理脚本示例

### 创建批量训练脚本 `train_batch.sh`

```bash
#!/bin/bash
python train.py \
    --dataset_dir ~/pallet_dataset/20251210_162120 \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --num_epochs 100
```

## 常见问题

### 深度图像处理

16-bit PNG 深度图像抽取后为 f32 类型。预处理：

```python
depth = np.array(Image.open(depth_path), dtype=np.float32)  # 单位: mm
# 推荐正一化 (z-score)
if config.DEPTH_NORMALIZE:
    depth = (depth - mean) / std
```

### 不匹配的囸量

RGB 盯融吸收 ImageNet 预训练当加入专法:

```python
if pretrained:
    self.rgb_backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
else:
    self.rgb_backbone = resnet50(weights=None)
```

深度分支不采用预训练权重（不匹配）。

## 转接到其他像素或数据集

RGB/Depth 图像率先改正为上述的汽浅格式，有两个主要修改点：

1. **`dataset.py`**: 改写 `_load_poses()` 处理你的标注格式
2. **`config.py`**: 调整正一化参数 (RGB_MEAN, RGB_STD, 等)

## 就业/感谢

- 本框架基于 `pallet_trainer` 项目，整体掞化。
- 特对应用: 16-bit PNG 深度图像支持、托盘 6-DOF 位姿估计

## 许可证

MIT License - 自由使用。
