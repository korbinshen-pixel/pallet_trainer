# config.py
import os
import torch

# ============ 数据路径 ============
DATASET_ROOT = os.path.expanduser('~/pallet_dataset')
# 假设你的最新数据集在 ~/pallet_dataset/20251210_153102 这样的目录
# 运行时会自动找最新的目录

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')
RESULT_DIR = os.path.join(OUTPUT_DIR, 'results')

# 创建目录
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ============ 数据集配置 ============
TRAIN_SPLIT = 0.8          # 80% 训练
VAL_SPLIT = 0.1            # 10% 验证
TEST_SPLIT = 0.1           # 10% 测试

# RGB 图像的归一化参数（ImageNet 统计）
RGB_MEAN = [0.485, 0.456, 0.406]
RGB_STD = [0.229, 0.224, 0.225]

# 深度图像的归一化方式（z-score）
DEPTH_NORMALIZE = True  # 用均值和标准差归一化

# ============ 模型配置 ============
MODEL_NAME = 'PalletPoseEstimator'
BACKBONE = 'resnet50'       # ResNet-50 作为主干
PRETRAINED = True           # RGB分支用预训练权重
FEATURE_DIM = 4096          # 融合后特征维度（2048+2048）

# ============ 训练超参数 ============
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 50
WARMUP_EPOCHS = 5           # 前5个 epoch 预热学习率

# 损失函数权重
POSITION_LOSS_WEIGHT = 1.0  # 位置损失权重
ROTATION_LOSS_WEIGHT = 1.0  # 旋转损失权重

# ============ 评估和保存 ============
EVAL_INTERVAL = 5           # 每5个 epoch 评估一次
SAVE_INTERVAL = 5           # 每5个 epoch 保存一次检查点
PATIENCE = 20               # Early stopping 耐心（多少个epoch没有改进就停止）

# ============ 推理配置 ============
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
INFER_BATCH_SIZE = 1

# ============ 数据增强 ============
USE_AUGMENTATION = True
AUGMENT_RGB_BRIGHTNESS = 0.2      # 亮度变化范围
AUGMENT_RGB_CONTRAST = 0.2         # 对比度变化范围
AUGMENT_DEPTH_NOISE_SIGMA = 0.01   # 深度噪声标准差
AUGMENT_ROTATION_RANGE = 10        # 旋转增强范围（度）
