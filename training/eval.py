import os
import torch
import numpy as np
from tqdm import tqdm
import argparse
import json

import config
from dataset import get_dataloaders
from model import PalletPoseEstimator, PalletPoseLoss


class Evaluator:
    """模型评估类"""
    
    def __init__(self, model_path, device=config.DEVICE):
        """
        Args:
            model_path: 保存的模型路径
            device: 运行设备
        """
        self.device = torch.device(device)
        
        # 加载模型
        self.model = PalletPoseEstimator(pretrained=False)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # 损失函数
        self.criterion = PalletPoseLoss(
            position_weight=config.POSITION_LOSS_WEIGHT,
            rotation_weight=config.ROTATION_LOSS_WEIGHT
        )
        
        print(f"Model loaded from {model_path}")
    
    def evaluate(self, data_loader):
        """评估数据集
        
        Args:
            data_loader: PyTorch DataLoader
        
        Returns:
            metrics: dict with evaluation metrics
        """
        total_loss = 0.0
        total_pos_loss = 0.0
        total_rot_loss = 0.0
        
        position_errors = []
        rotation_errors = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc='Evaluating'):
                rgb = batch['rgb'].to(self.device)
                depth = batch['depth'].to(self.device)
                position_gt = batch['position'].to(self.device)
                rotation_gt = batch['rotation'].to(self.device)
                
                # 前向传播
                position_pred, rotation_pred = self.model(rgb, depth)
                
                # 计算损失
                loss, pos_loss, rot_loss = self.criterion(
                    position_pred, rotation_pred,
                    position_gt, rotation_gt
                )
                
                total_loss += loss.item()
                total_pos_loss += pos_loss.item()
                total_rot_loss += rot_loss.item()
                
                # 计算误差
                pos_error = torch.norm(position_pred - position_gt, dim=1)  # (B,)
                rot_error = torch.norm(rotation_pred - rotation_gt, dim=1)  # (B,)
                
                position_errors.extend(pos_error.cpu().numpy().tolist())
                rotation_errors.extend(rot_error.cpu().numpy().tolist())
        
        # 统计指标
        metrics = {
            'total_loss': total_loss / len(data_loader),
            'position_loss': total_pos_loss / len(data_loader),
            'rotation_loss': total_rot_loss / len(data_loader),
            'position_error_mean': np.mean(position_errors),
            'position_error_std': np.std(position_errors),
            'position_error_max': np.max(position_errors),
            'rotation_error_mean': np.mean(rotation_errors),
            'rotation_error_std': np.std(rotation_errors),
            'rotation_error_max': np.max(rotation_errors),
        }
        
        return metrics
    
    def print_metrics(self, metrics, title='Evaluation Results'):
        """打印评估指标"""
        print(f"\n{'='*60}")
        print(f"{title}")
        print(f"{'='*60}")
        print(f"Total Loss: {metrics['total_loss']:.6f}")
        print(f"Position Loss: {metrics['position_loss']:.6f}")
        print(f"Rotation Loss: {metrics['rotation_loss']:.6f}")
        print(f"\nPosition Error (m):")
        print(f"  Mean: {metrics['position_error_mean']:.6f}")
        print(f"  Std: {metrics['position_error_std']:.6f}")
        print(f"  Max: {metrics['position_error_max']:.6f}")
        print(f"\nRotation Error:")
        print(f"  Mean: {metrics['rotation_error_mean']:.6f}")
        print(f"  Std: {metrics['rotation_error_std']:.6f}")
        print(f"  Max: {metrics['rotation_error_max']:.6f}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Pallet Pose Estimator')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the trained model')
    parser.add_argument('--dataset_dir', type=str, default=None,
                        help='Path to the dataset directory')
    parser.add_argument('--device', type=str, default=config.DEVICE,
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--split', type=str, default='test',
                        help='Data split to evaluate (train/val/test)')
    parser.add_argument('--output', type=str, default=None,
                        help='Save metrics to JSON file')
    
    args = parser.parse_args()
    
    # 加载数据集
    print(f"Loading {args.split} dataset...")
    from dataset import PalletPoseDataset
    from torch.utils.data import DataLoader
    from torchvision import transforms
    
    # RGB 美化变换
    rgb_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=config.RGB_MEAN, std=config.RGB_STD)
    ])
    
    # 深度预处理
    def depth_transform(depth):
        if isinstance(depth, torch.Tensor):
            depth_tensor = depth
        else:
            depth_tensor = torch.from_numpy(depth)
        
        if depth_tensor.dim() == 2:
            depth_tensor = depth_tensor.unsqueeze(0)
        
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
    
    # 加载数据集
    if args.dataset_dir is None:
        from dataset import get_latest_dataset_dir
        args.dataset_dir = get_latest_dataset_dir()
    
    dataset = PalletPoseDataset(
        args.dataset_dir, split=args.split,
        transform_rgb=rgb_transform,
        transform_depth=depth_transform,
        augmentation=False
    )
    
    data_loader = DataLoader(
        dataset, batch_size=config.INFER_BATCH_SIZE,
        shuffle=False, num_workers=4
    )
    
    # 创建评估器
    evaluator = Evaluator(args.model, args.device)
    
    # 评估
    metrics = evaluator.evaluate(data_loader)
    evaluator.print_metrics(metrics, f'{args.split.upper()} Evaluation Results')
    
    # 保存结果
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to {args.output}")


if __name__ == '__main__':
    main()
