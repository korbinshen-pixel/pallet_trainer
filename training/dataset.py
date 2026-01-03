import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob
from sklearn.model_selection import train_test_split
import config


def euler_to_quaternion(roll, pitch, yaw):
    """
    将欧拉角 (roll, pitch, yaw) 转换为四元数 (qx, qy, qz, qw)
    
    Args:
        roll: 绕 X 轴旋转 (弧度)
        pitch: 绕 Y 轴旋转 (弧度)
        yaw: 绕 Z 轴旋转 (弧度)
    
    Returns:
        q: numpy array [qx, qy, qz, qw]
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    
    return np.array([qx, qy, qz, qw], dtype=np.float32)


def quaternion_to_euler(qx, qy, qz, qw):
    """
    将四元数 (qx, qy, qz, qw) 转换为欧拉角 (roll, pitch, yaw)
    
    Args:
        qx, qy, qz, qw: 四元数分量
    
    Returns:
        roll, pitch, yaw: 欧拉角 (弧度)
    """
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = np.sign(sinp) * np.pi / 2  # use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw


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
        │   ├── depth_000000.png
        │   ├── depth_000001.png
        │   └── ...
        └── poses.txt  (每行: frame_id x y z roll pitch yaw)
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
        self.depth_files = sorted(glob.glob(os.path.join(self.depth_dir, 'depth_*.png')))
        
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
        """
        加载 poses.txt，支持两种格式：
        1. 欧拉角格式: frame_id x y z roll pitch yaw (弧度)
        2. 四元数格式: frame_id x y z qx qy qz qw
        
        自动检测格式，如果是欧拉角则转换为四元数
        """
        poses = {}
        with open(self.pose_file, 'r') as f:
            lines = f.readlines()
            
        if not lines:
            raise ValueError("poses.txt is empty")
        
        # 检测第一行来判断格式
        first_line = lines[0].strip().split()
        
        # 判断是否为欧拉角格式 (通过检查数值范围)
        # 欧拉角通常在 [-π, π] 或 [-180, 180] 范围
        # 四元数的模应该接近 1
        if len(first_line) == 7:  # frame_id + 6 个数值
            test_values = [float(x) for x in first_line[1:]]
            rotation_values = test_values[3:6]  # 后 3 个值
            
            # 如果后三个值都在 [-2π, 2π] 范围内，判定为欧拉角格式
            is_euler = all(abs(v) <= 2 * np.pi + 0.5 for v in rotation_values)
            
            if is_euler:
                print("Detected Euler angle format (roll, pitch, yaw in radians)")
                format_type = 'euler'
            else:
                print("Detected quaternion format (qx, qy, qz, qw) - but only 6 values!")
                raise ValueError("Invalid format: expected 7 values for quaternion (x y z qx qy qz qw)")
        elif len(first_line) == 8:  # frame_id + 7 个数值
            print("Detected quaternion format (x, y, z, qx, qy, qz, qw)")
            format_type = 'quaternion'
        else:
            raise ValueError(f"Invalid pose format: expected 7 or 8 values, got {len(first_line)}")
        
        # 解析所有行
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
                
            frame_id = parts[0]
            values = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            
            if format_type == 'euler':
                # 格式: frame_id x y z roll pitch yaw
                assert len(values) == 6, f"Euler format expects 6 values, got {len(values)}"
                position = values[:3]  # x, y, z
                euler_angles = values[3:6]  # roll, pitch, yaw (弧度)
                
                # 转换为四元数
                rotation = euler_to_quaternion(euler_angles[0], euler_angles[1], euler_angles[2])
                pose = np.concatenate([position, rotation])  # [x, y, z, qx, qy, qz, qw]
            else:
                # 格式: frame_id x y z qx qy qz qw
                assert len(values) == 7, f"Quaternion format expects 7 values, got {len(values)}"
                pose = values  # 已经是 [x, y, z, qx, qy, qz, qw]
            
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
        
        # 加载深度图像（16bit PNG）
        depth_path = self.depth_files[actual_idx]
        depth_img = Image.open(depth_path)
        depth = np.array(depth_img, dtype=np.float32)  # (H, W) in mm
        
        # 提取位姿（从文件名提取 frame_id）
        frame_id = os.path.basename(rgb_path).replace('rgb_', '').replace('.png', '')
        pose = self.poses[frame_id]  # (7,) [x, y, z, qx, qy, qz, qw]
        
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
            depth = torch.from_numpy(depth).unsqueeze(0)  # (1, H, W)
        
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
            enhancer = transforms.RandomAdjustSharpness(0.5)
        
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
    
    # 深度图像的处理
    def depth_transform(depth):
        """
        通用深度预处理：
        - 输入可以是 np.ndarray(H, W) 或 Tensor(H, W) 或 Tensor(1, H, W)
        - 输出统一为 Tensor(1, H, W)
        """
        # 1. 统一成 torch.Tensor
        if isinstance(depth, torch.Tensor):
            depth_tensor = depth
        else:
            # 假设是 numpy
            depth_tensor = torch.from_numpy(depth)

        # 2. 保证是 2D 或 3D
        if depth_tensor.dim() == 3 and depth_tensor.size(0) != 1:
            depth_tensor = depth_tensor[0, ...]
        
        # 3. 确保现在是 (H, W) 或 (1, H, W)
        if depth_tensor.dim() == 2:
            # (H, W) -> (1, H, W)
            depth_tensor = depth_tensor.unsqueeze(0)
        elif depth_tensor.dim() == 3 and depth_tensor.size(0) == 1:
            # 已经是 (1, H, W)，OK
            pass
        else:
            raise ValueError(f"Unexpected depth shape: {depth_tensor.shape}")

        # 4. 归一化
        if config.DEPTH_NORMALIZE:
            # 按 (1, H, W) 形状做 z-score 归一化
            mean = depth_tensor.mean()
            std = depth_tensor.std()
            if std > 0:
                depth_tensor = (depth_tensor - mean) / std
        else:
            # 简单最大值归一化
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
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True
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
    
    # 测试欧拉角转换
    print("\nTesting Euler to Quaternion conversion...")
    roll, pitch, yaw = 0.1, 0.2, 0.3  # 弧度
    q = euler_to_quaternion(roll, pitch, yaw)
    print(f"Euler (rad): roll={roll}, pitch={pitch}, yaw={yaw}")
    print(f"Quaternion: qx={q[0]}, qy={q[1]}, qz={q[2]}, qw={q[3]}")
    print(f"Quaternion norm: {np.linalg.norm(q)}")
    
    # 反向测试
    roll_back, pitch_back, yaw_back = quaternion_to_euler(q[0], q[1], q[2], q[3])
    print(f"Back to Euler: roll={roll_back}, pitch={pitch_back}, yaw={yaw_back}")
    
    print("\nDataLoader test passed!")
