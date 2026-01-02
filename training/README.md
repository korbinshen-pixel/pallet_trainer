# 托盘位姿估计模型训练框架

## 快速开始

### 1. 数据准备

确保你已经用 `data_collector` 采集了数据，数据放在 `~/pallet_dataset/` 目录下。

目录结构：

~/pallet_dataset/
20251210_153102/
rgb/
rgb_000000.png
rgb_000001.png
...
depth/
depth_000000.npy
depth_000001.npy
...
poses.txt

text

### 2. 测试数据加载器

python3 dataset.py

text

如果输出显示 `DataLoader test passed!`，说明数据加载器工作正常。

### 3. 测试模型

python3 model.py

text

如果输出显示 `Model test passed!`，说明模型定义正确。

### 4. 开始训练

python3 train.py

text

训练过程会：
- 保存检查点到 `checkpoints/`
- 保存最好模型到 `models/best_model.pth`
- 记录日志到 `logs/`

### 5. 实时查看训练进度（可选）

在另一个终端运行：

tensorboard --logdir=logs/

text

然后在浏览器打开 `http://localhost:6006`

### 6. 评估模型

python3 eval.py

text

输出位置误差和旋转误差统计。

### 7. 推理

python3 infer.py

text

对新数据进行推理并输出位姿。

## 文件说明

| 文件 | 说明 |
|------|------|
| `config.py` | 所有超参数配置 |
| `dataset.py` | 数据加载器 |
| `model.py` | 模型定义 + 损失函数 |
| `train.py` | 训练脚本 |
| `eval.py` | 评估脚本 |
| `infer.py` | 推理脚本 |

## 调整超参数

编辑 `config.py`：

- `BATCH_SIZE`: 批大小（如果显存不足改小）
- `LEARNING_RATE`: 学习率（默认 1e-4）
- `NUM_EPOCHS`: 训练轮数
- `PATIENCE`: Early stopping 耐心（多少 epoch 没改进就停止）
- `POSITION_LOSS_WEIGHT`, `ROTATION_LOSS_WEIGHT`: 损失函数权重

## 常见问题

### Q: CUDA out of memory
A: 在 `config.py` 中降低 `BATCH_SIZE`

### Q: 训练太慢
A: 确保你的 GPU 被使用了：

nvidia-smi # 查看 GPU 使用情况

text

### Q: 模型精度不高
A: 
1. 增加训练数据
2. 调整超参数（学习率、损失函数权重）
3. 尝试更复杂的模型架构

## 输出文件

training/
├── checkpoints/ # 训练检查点
├── models/
│ └── best_model.pth # 最好模型权重
├── logs/
│ └── run_*/ # TensorBoard 日志
├── results/
│ └── eval_results.txt # 评估结果
└── *.py # 脚本文件

text

## 下一步

1. 使用 `infer.py` 在实际机器人上进行推理
2. 如果精度不够，考虑升级到更复杂的模型（见推荐方案 3-5）
3. 收集更多真实数据进行微调（fine-tuning）