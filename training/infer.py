# infer.py
import torch
import numpy as np
from PIL import Image
import os
import logging

import config
from model import PalletPoseEstimator


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PoseEstimator:
    """推理器"""
    
    def __init__(self, model_path=None, device=config.DEVICE):
        """
        Args:
            model_path: 模型权重文件路径，如果为 None 则使用默认的最好模型
            device: 计算设备
        """
        self.device = device
        
        # 加载模型
        self.model = PalletPoseEstimator()
        
        if model_path is None:
            model_path = os.path.join(config.MODEL_DIR, 'checkpoint_epoch_055.pth')
        
        if not os.path.exists(model_path):
            logger.warning(f'Model not found: {model_path}')
            logger.info('Using untrained model')
        else:
            checkpoint = torch.load(model_path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f'Model loaded from {model_path}')
        
        self.model.to(device)
        self.model.eval()
    
    def infer(self, rgb_path, depth_path):
        """
        从 RGB 和深度图像推理位姿
        
        Args:
            rgb_path: RGB 图像路径
            depth_path: 深度图像路径（.npy 格式）
        
        Returns:
            position: (3,) - 位置 (x, y, z)
            rotation: (4,) - 四元数 (qx, qy, qz, qw)
        """
        # 加载图像
        rgb = Image.open(rgb_path).convert('RGB')
        depth = np.load(depth_path).astype(np.float32)
        
        # 预处理
        rgb_tensor = self._preprocess_rgb(rgb)
        depth_tensor = self._preprocess_depth(depth)
        
        # 推理
        with torch.no_grad():
            position, rotation = self.model(rgb_tensor, depth_tensor)
        
        return position.cpu().numpy(), rotation.cpu().numpy()
    
    def _preprocess_rgb(self, rgb):
        """RGB 图像预处理"""
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=config.RGB_MEAN, std=config.RGB_STD)
        ])
        
        rgb_tensor = transform(rgb)  # (3, H, W)
        rgb_tensor = rgb_tensor.unsqueeze(0).to(self.device)  # (1, 3, H, W)
        
        return rgb_tensor
    
    def _preprocess_depth(self, depth):
        """深度图像预处理"""
        # Z-score 归一化
        if config.DEPTH_NORMALIZE:
            depth_mean = depth.mean()
            depth_std = depth.std()
            if depth_std > 0:
                depth = (depth - depth_mean) / depth_std
        else:
            depth_max = depth.max()
            if depth_max > 0:
                depth = depth / depth_max
        
        depth_tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        depth_tensor = depth_tensor.to(self.device)
        
        return depth_tensor


def main():
    """测试推理"""
    logger.info('=' * 80)
    logger.info('Pallet Pose Estimation Inference')
    logger.info('=' * 80)
    
    # 创建推理器
    estimator = PoseEstimator()
    
    # 测试推理（使用数据集中的一个样本）
    dataset_dir = os.path.expanduser('~/pallet_dataset')
    subdirs = sorted([d for d in os.listdir(dataset_dir) 
                      if os.path.isdir(os.path.join(dataset_dir, d))])
    
    if not subdirs:
        logger.error('No dataset found')
        return
    
    latest_dataset = os.path.join(dataset_dir, subdirs[-1])
    rgb_file = os.path.join(latest_dataset, 'rgb', 'rgb_000000.png')
    depth_file = os.path.join(latest_dataset, 'depth', 'depth_000000.npy')
    
    if not (os.path.exists(rgb_file) and os.path.exists(depth_file)):
        logger.error(f'Sample files not found in {latest_dataset}')
        return
    
    logger.info(f'Inferring on: {rgb_file}, {depth_file}')
    
    position, rotation = estimator.infer(rgb_file, depth_file)
    
    logger.info('=' * 80)
    logger.info('Inference Results:')
    logger.info('=' * 80)
    logger.info(f'Position (x, y, z): {position}')
    logger.info(f'  x: {position[0]:.6f} m')
    logger.info(f'  y: {position[1]:.6f} m')
    logger.info(f'  z: {position[2]:.6f} m')
    logger.info('')
    logger.info(f'Rotation (Quaternion): {rotation}')
    logger.info(f'  qx: {rotation[0]:.6f}')
    logger.info(f'  qy: {rotation[1]:.6f}')
    logger.info(f'  qz: {rotation[2]:.6f}')
    logger.info(f'  qw: {rotation[3]:.6f}')
    logger.info('=' * 80)


if __name__ == '__main__':
    main()
