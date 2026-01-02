# dataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob
from sklearn.model_selection import train_test_split
import config


def get_latest_dataset_dir(root_dir=config.DATASET_ROOT):
    """获取最新采集的数据集目录"""
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Dataset root dir not found: {root_dir}")
    
    # 找所有时间戳目录（格式：20251210_162120）
    subdirs = [d for d in os.listdir(root_dir) 
               if os.path.isdir(os.path.join(root_dir, d))]
    
    if not subdirs:
        raise FileNotFoundError(f"No dataset directories found in {root_dir}")
    
    subdirs.sort()  # 按字母序（时间戳自然排序）
    latest = os.path.join(root_dir, subdirs[-1])
    print(f"Using latest dataset: {latest}")
    return latest


class PalletPoseDataset(Dataset):
    """
    托盘位姿数据集加载器
    
    期望的目录结构：
        dataset_dir/
        ├── rgb/
        │   ├── rgb_000000.png
        │   ├── rgb_000001.png
        │   └── ...
        ├── depth/
        │   ├── depth_000000.npy
        │   ├── depth_000001.npy
        │   └── ...
        └── poses.txt  (每行：frame_id x y z qx qy qz qw)
    """
    
    def __init__(self, dataset_dir, split='train', transform_rgb=None, 
                 transform_depth=None, augmentation=False):
        """
        Args:
            dataset_dir: 数据集目录
            split: 'train', 'val', 或 'test'
            transform_rgb: RGB 图像的数据增强/归一化
            transform_depth: 深度图像的处理
            augmentation: 是否启用数据增强
        """
        self.dataset_dir = dataset_dir
        self.split = split
        self.transform_rgb = transform_rgb
        self.transform_depth = transform_depth
        self.augmentation = augmentation
        
        # 加载 RGB 和 Depth 文件列表
        self.rgb_dir = os.path.join(dataset_dir, 'rgb')
        self.depth_dir = os.path.join(dataset_dir, 'depth')
        self.pose_file = os.path.join(dataset_dir, 'poses.txt')
        
        if not os.path.exists(self.rgb_dir):
            raise FileNotFoundError(f"RGB dir not found: {self.rgb_dir}")
        if not os.path.exists(self.depth_dir):
            raise FileNotFoundError(f"Depth dir not found: {self.depth_dir}")
        if not os.path.exists(self.pose_file):
            raise FileNotFoundError(f"Pose file not found: {self.pose_file}")
        
        # 读取 RGB 文件列表（按名字排序）
        self.rgb_files = sorted(glob.glob(os.path.join(self.rgb_dir, 'rgb_*.png')))
        self.depth_files = sorted(glob.glob(os.path.join(self.depth_dir, 'depth_*.npy')))
        
        # 读取位姿数据
        self.poses = self._load_poses()
        
        # 确保三个数据集大小一致
        assert len(self.rgb_files) == len(self.depth_files) == len(self.poses), \
            "RGB, Depth, and Pose counts mismatch"
        
        print(f"Loaded {len(self.rgb_files)} samples from {dataset_dir}")
        
        # 分割数据集
        indices = np.arange(len(self.rgb_files))
        train_size = int(len(indices) * config.TRAIN_SPLIT)
        val_size = int(len(indices) * config.VAL_SPLIT)
        
        train_idx, test_idx = train_test_split(
            indices, test_size=(1 - config.TRAIN_SPLIT), random_state=42
        )
        val_idx, test_idx = train_test_split(
            test_idx, test_size=config.TEST_SPLIT / (config.VAL_SPLIT + config.TEST_SPLIT),
            random_state=42
        )
        
        if split == 'train':
            self.indices = train_idx
        elif split == 'val':
            self.indices = val_idx
        elif split == 'test':
            self.indices = test_idx
        else:
            raise ValueError(f"Unknown split: {split}")
        
        print(f"Split '{split}': {len(self.indices)} samples")
    
    def _load_poses(self):
        """加载 poses.txt，格式：frame_id x y z qx qy qz qw"""
        poses = {}
        with open(self.pose_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                frame_id = parts[0]
                pose = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                # pose 应该是 7D：位置3D + 四元数4D
                assert len(pose) == 7, f"Pose should be 7D, got {len(pose)}"
                poses[frame_id] = pose
        return poses
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """返回 (rgb, depth, position, rotation)"""
        actual_idx = self.indices[idx]
        
        # 加载 RGB 图像
        rgb_path = self.rgb_files[actual_idx]
        rgb = Image.open(rgb_path).convert('RGB')
        
        # 加载深度图像
        # 加载深度图像
        depth_path = self.depth_files[actual_idx]
        depth = np.load(depth_path).astype(np.float32)  # (H, W)

        # 数据增强（如果有的话），这里建议暂时不要改 depth 的类型
        if self.augmentation and self.split == 'train':
            rgb, depth = self._augment(rgb, depth)

        # 转换为张量 / 归一化
        if self.transform_depth:
            depth = self.transform_depth(depth)   # 让它接 numpy 或 tensor 自己处理
        else:
            depth = torch.from_numpy(depth).unsqueeze(0)  # (1, H, W)


        
        # 提取位姿（从文件名提取 frame_id）
        frame_id = os.path.basename(rgb_path).replace('rgb_', '').replace('.png', '')
        pose = self.poses[frame_id]  # (7,)
        
        position = pose[:3]  # (x, y, z)
        rotation = pose[3:]  # (qx, qy, qz, qw)
        
        # 数据增强（可选）
        if self.augmentation and self.split == 'train':
            rgb, depth = self._augment(rgb, depth)
        
        # 转换为张量
        if self.transform_rgb:
            rgb = self.transform_rgb(rgb)
        else:
            rgb = transforms.ToTensor()(rgb)
        
        if self.transform_depth:
            depth = self.transform_depth(depth)
        else:
            depth = torch.from_numpy(depth)
        
        # 返回数据
        return {
            'rgb': rgb,                    # (3, H, W)
            'depth': depth,                # (1, H, W)
            'position': torch.from_numpy(position),  # (3,)
            'rotation': torch.from_numpy(rotation),  # (4,)
            'frame_id': frame_id
        }
    
    def _augment(self, rgb, depth):
        """简单的数据增强"""
        # RGB 亮度和对比度增强
        if np.random.rand() > 0.5:
            brightness_factor = 1 + np.random.uniform(-config.AUGMENT_RGB_BRIGHTNESS, 
                                                       config.AUGMENT_RGB_BRIGHTNESS)
            rgb = Image.new('RGB', rgb.size)
            # 简单亮度调整...（这里为了简洁省略详细实现）
        
        return rgb, depth


def get_dataloaders(dataset_dir=None, batch_size=config.BATCH_SIZE):
    """
    获取 train/val/test 数据加载器
    
    Args:
        dataset_dir: 数据集目录，如果为 None 则自动找最新的
        batch_size: 批大小
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    if dataset_dir is None:
        dataset_dir = get_latest_dataset_dir()
    
    # RGB 图像的标准化（ImageNet 统计）
    rgb_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=config.RGB_MEAN, std=config.RGB_STD)
    ])
    
    # 深度图像的处理（简单转张量）
    def depth_transform(depth):

        """
        通用深度预处理：
        - 输入可以是 np.ndarray(H, W) 或 Tensor(H, W) 或 Tensor(1, H, W)
        - 输出统一为 Tensor(1, H, W)
        """
        import torch

        # 1. 统一成 torch.Tensor
        if isinstance(depth, torch.Tensor):
            depth_tensor = depth
        else:
            # 假设是 numpy
            depth_tensor = torch.from_numpy(depth)

        # 2. 去掉多余的 batch 维，保证最多 3 维
        # 例如有时候可能是 (1, H, W)，有时候 (H, W)
        if depth_tensor.dim() == 3 and depth_tensor.size(0) != 1:
            # 极端情况 (H, W, C) 这种，不太可能出现在当前管线，先简单处理
            # 这里取第一个通道
            depth_tensor = depth_tensor[0, ...]
        
        # 3. 确保现在是 (H, W) 或 (1, H, W)
        if depth_tensor.dim() == 2:
            # (H, W) -> (1, H, W)
            depth_tensor = depth_tensor.unsqueeze(0)
        elif depth_tensor.dim() == 3 and depth_tensor.size(0) == 1:
            # 已经是 (1, H, W)，OK
            pass
        else:
            raise ValueError(f"Unexpected depth shape in depth_transform: {depth_tensor.shape}")

        # 4. 归一化：在 (1, H, W) 上做
        if config.DEPTH_NORMALIZE:
            mean = depth_tensor.mean()
            std = depth_tensor.std()
            if std > 0:
                depth_tensor = (depth_tensor - mean) / std
        else:
            maxv = depth_tensor.max()
            if maxv > 0:
                depth_tensor = depth_tensor / maxv

        return depth_tensor


    
    # 创建三个数据集
    train_dataset = PalletPoseDataset(
        dataset_dir, split='train',
        transform_rgb=rgb_transform,
        transform_depth=depth_transform,
        augmentation=config.USE_AUGMENTATION
    )
    
    val_dataset = PalletPoseDataset(
        dataset_dir, split='val',
        transform_rgb=rgb_transform,
        transform_depth=depth_transform,
        augmentation=False
    )
    
    test_dataset = PalletPoseDataset(
        dataset_dir, split='test',
        transform_rgb=rgb_transform,
        transform_depth=depth_transform,
        augmentation=False
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True  # <-- 加上这一行！
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # 测试数据加载器
    print("Testing DataLoader...")
    train_loader, val_loader, test_loader = get_dataloaders()
    
    # 获取一个批次
    batch = next(iter(train_loader))
    print(f"Batch RGB shape: {batch['rgb'].shape}")
    print(f"Batch Depth shape: {batch['depth'].shape}")
    print(f"Batch Position shape: {batch['position'].shape}")
    print(f"Batch Rotation shape: {batch['rotation'].shape}")
    print("DataLoader test passed!")
