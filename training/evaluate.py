import os
import torch
import numpy as np
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

import config
from model import PalletPoseEstimator
from dataset import get_dataloaders, quaternion_to_euler


class PalletPoseEvaluator:
    """托盘位姿评估器"""
    
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
        
        print(f"Model loaded from {model_path}")
        if 'epoch' in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}")
        if 'val_loss' in checkpoint:
            print(f"  Validation Loss: {checkpoint['val_loss']:.6f}")
    
    def evaluate(self, test_loader, output_euler=True):
        """
        对测试集进行评估
        
        Args:
            test_loader: 测试数据加载器
            output_euler: 是否输出欧拉角误差（True）或四元数误差（False）
        
        Returns:
            metrics: dict 包含各种评价指标
        """
        position_errors = []
        rotation_errors = []  # 如果 output_euler=True，存储欧拉角误差；否则存储四元数误差
        
        position_errors_xyz = []  # 分别记录 x, y, z 误差
        
        if output_euler:
            rotation_errors_rpy = []  # roll, pitch, yaw 误差
        
        print("Evaluating on test set...")
        
        with torch.no_grad():
            for batch in tqdm(test_loader):
                rgb = batch['rgb'].to(self.device)
                depth = batch['depth'].to(self.device)
                position_gt = batch['position'].to(self.device)
                rotation_gt = batch['rotation'].to(self.device)
                
                # 推理
                position_pred, rotation_pred = self.model(rgb, depth)
                
                # 计算位置误差 (L2 距离)
                pos_error = torch.norm(position_pred - position_gt, dim=1)
                position_errors.extend(pos_error.cpu().numpy())
                
                # 记录 x, y, z 分别的误差
                pos_diff = torch.abs(position_pred - position_gt).cpu().numpy()
                position_errors_xyz.append(pos_diff)
                
                if output_euler:
                    # 转换为欧拉角计算误差
                    rotation_pred_np = rotation_pred.cpu().numpy()
                    rotation_gt_np = rotation_gt.cpu().numpy()
                    
                    batch_size = rotation_pred_np.shape[0]
                    for i in range(batch_size):
                        # 预测的四元数转欧拉角
                        qp = rotation_pred_np[i]
                        roll_p, pitch_p, yaw_p = quaternion_to_euler(qp[0], qp[1], qp[2], qp[3])
                        
                        # GT 四元数转欧拉角
                        qg = rotation_gt_np[i]
                        roll_g, pitch_g, yaw_g = quaternion_to_euler(qg[0], qg[1], qg[2], qg[3])
                        
                        # 计算欧拉角误差 (弧度)
                        euler_diff = np.array([
                            self._angle_diff(roll_p, roll_g),
                            self._angle_diff(pitch_p, pitch_g),
                            self._angle_diff(yaw_p, yaw_g)
                        ])
                        
                        rotation_errors_rpy.append(euler_diff)
                        
                        # 总旋转误差（欧拉角的 L2 范数）
                        rotation_errors.append(np.linalg.norm(euler_diff))
                else:
                    # 计算四元数误差（角度距离）
                    # 使用公式: theta = 2 * arccos(|q1 · q2|)
                    dot_product = torch.sum(rotation_pred * rotation_gt, dim=1)
                    dot_product = torch.clamp(dot_product, -1.0, 1.0)
                    rot_error = 2 * torch.acos(torch.abs(dot_product))
                    rotation_errors.extend(rot_error.cpu().numpy())
        
        # 转换为 numpy 数组
        position_errors = np.array(position_errors)
        rotation_errors = np.array(rotation_errors)
        position_errors_xyz = np.concatenate(position_errors_xyz, axis=0)
        
        # 计算统计指标
        metrics = {
            'position': {
                'mean': position_errors.mean(),
                'std': position_errors.std(),
                'median': np.median(position_errors),
                'max': position_errors.max(),
                'min': position_errors.min(),
            },
            'rotation': {
                'mean': rotation_errors.mean(),
                'std': rotation_errors.std(),
                'median': np.median(rotation_errors),
                'max': rotation_errors.max(),
                'min': rotation_errors.min(),
            },
            'position_xyz': {
                'x_mean': position_errors_xyz[:, 0].mean(),
                'y_mean': position_errors_xyz[:, 1].mean(),
                'z_mean': position_errors_xyz[:, 2].mean(),
                'x_std': position_errors_xyz[:, 0].std(),
                'y_std': position_errors_xyz[:, 1].std(),
                'z_std': position_errors_xyz[:, 2].std(),
            }
        }
        
        if output_euler:
            rotation_errors_rpy = np.array(rotation_errors_rpy)
            metrics['rotation_rpy'] = {
                'roll_mean': rotation_errors_rpy[:, 0].mean(),
                'pitch_mean': rotation_errors_rpy[:, 1].mean(),
                'yaw_mean': rotation_errors_rpy[:, 2].mean(),
                'roll_std': rotation_errors_rpy[:, 0].std(),
                'pitch_std': rotation_errors_rpy[:, 1].std(),
                'yaw_std': rotation_errors_rpy[:, 2].std(),
                'roll_mean_deg': np.rad2deg(rotation_errors_rpy[:, 0].mean()),
                'pitch_mean_deg': np.rad2deg(rotation_errors_rpy[:, 1].mean()),
                'yaw_mean_deg': np.rad2deg(rotation_errors_rpy[:, 2].mean()),
            }
        
        # 存储原始误差数据用于绘图
        metrics['raw_errors'] = {
            'position': position_errors,
            'rotation': rotation_errors,
            'position_xyz': position_errors_xyz,
        }
        
        if output_euler:
            metrics['raw_errors']['rotation_rpy'] = rotation_errors_rpy
        
        return metrics
    
    def _angle_diff(self, angle1, angle2):
        """
        计算两个角度之间的最小差值（考虑周期性）
        返回范围: [-π, π]
        """
        diff = angle1 - angle2
        # 将差值映射到 [-π, π]
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        return abs(diff)
    
    def print_metrics(self, metrics, output_euler=True):
        """打印评估指标"""
        print("\n" + "="*60)
        print("Evaluation Metrics")
        print("="*60)
        
        print("\nPosition Error (meters):")
        print(f"  Mean:   {metrics['position']['mean']:.6f} m")
        print(f"  Std:    {metrics['position']['std']:.6f} m")
        print(f"  Median: {metrics['position']['median']:.6f} m")
        print(f"  Min:    {metrics['position']['min']:.6f} m")
        print(f"  Max:    {metrics['position']['max']:.6f} m")
        
        print("\nPosition Error by Axis (meters):")
        print(f"  X: {metrics['position_xyz']['x_mean']:.6f} ± {metrics['position_xyz']['x_std']:.6f} m")
        print(f"  Y: {metrics['position_xyz']['y_mean']:.6f} ± {metrics['position_xyz']['y_std']:.6f} m")
        print(f"  Z: {metrics['position_xyz']['z_mean']:.6f} ± {metrics['position_xyz']['z_std']:.6f} m")
        
        if output_euler:
            print("\nRotation Error (Euler angles):")
            print(f"  Overall Mean: {metrics['rotation']['mean']:.6f} rad ({np.rad2deg(metrics['rotation']['mean']):.2f}°)")
            print(f"  Overall Std:  {metrics['rotation']['std']:.6f} rad ({np.rad2deg(metrics['rotation']['std']):.2f}°)")
            
            print("\nRotation Error by Axis (radians / degrees):")
            print(f"  Roll:  {metrics['rotation_rpy']['roll_mean']:.6f} rad ({metrics['rotation_rpy']['roll_mean_deg']:.2f}°) ± {metrics['rotation_rpy']['roll_std']:.6f} rad")
            print(f"  Pitch: {metrics['rotation_rpy']['pitch_mean']:.6f} rad ({metrics['rotation_rpy']['pitch_mean_deg']:.2f}°) ± {metrics['rotation_rpy']['pitch_std']:.6f} rad")
            print(f"  Yaw:   {metrics['rotation_rpy']['yaw_mean']:.6f} rad ({metrics['rotation_rpy']['yaw_mean_deg']:.2f}°) ± {metrics['rotation_rpy']['yaw_std']:.6f} rad")
        else:
            print("\nRotation Error (Quaternion):")
            print(f"  Mean:   {metrics['rotation']['mean']:.6f} rad ({np.rad2deg(metrics['rotation']['mean']):.2f}°)")
            print(f"  Std:    {metrics['rotation']['std']:.6f} rad ({np.rad2deg(metrics['rotation']['std']):.2f}°)")
            print(f"  Median: {metrics['rotation']['median']:.6f} rad ({np.rad2deg(metrics['rotation']['median']):.2f}°)")
        
        print("="*60)
    
    def plot_error_distribution(self, metrics, save_path=None, output_euler=True):
        """绘制误差分布图"""
        position_errors = metrics['raw_errors']['position']
        rotation_errors = metrics['raw_errors']['rotation']
        position_errors_xyz = metrics['raw_errors']['position_xyz']
        
        if output_euler:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        else:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 位置误差分布
        axes[0, 0].hist(position_errors, bins=50, edgecolor='black')
        axes[0, 0].set_xlabel('Position Error (m)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Position Error Distribution')
        axes[0, 0].axvline(metrics['position']['mean'], color='r', 
                          linestyle='--', label=f"Mean: {metrics['position']['mean']:.4f} m")
        axes[0, 0].legend()
        
        # 旋转误差分布
        if output_euler:
            # 欧拉角总误差
            axes[0, 1].hist(rotation_errors, bins=50, edgecolor='black')
            axes[0, 1].set_xlabel('Rotation Error (rad)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Overall Rotation Error Distribution')
            axes[0, 1].axvline(metrics['rotation']['mean'], color='r',
                              linestyle='--', label=f"Mean: {metrics['rotation']['mean']:.4f} rad")
            axes[0, 1].legend()
        else:
            # 四元数误差
            axes[0, 1].hist(rotation_errors, bins=50, edgecolor='black')
            axes[0, 1].set_xlabel('Rotation Error (rad)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Rotation Error Distribution (Quaternion)')
            axes[0, 1].axvline(metrics['rotation']['mean'], color='r',
                              linestyle='--', label=f"Mean: {metrics['rotation']['mean']:.4f} rad")
            axes[0, 1].legend()
        
        # X, Y, Z 误差分布
        if output_euler:
            axes[0, 2].hist(position_errors_xyz[:, 0], bins=30, alpha=0.5, label='X', edgecolor='black')
            axes[0, 2].hist(position_errors_xyz[:, 1], bins=30, alpha=0.5, label='Y', edgecolor='black')
            axes[0, 2].hist(position_errors_xyz[:, 2], bins=30, alpha=0.5, label='Z', edgecolor='black')
            axes[0, 2].set_xlabel('Position Error (m)')
            axes[0, 2].set_ylabel('Frequency')
            axes[0, 2].set_title('Position Error by Axis')
            axes[0, 2].legend()
            
            # Roll, Pitch, Yaw 误差分布
            rotation_errors_rpy = metrics['raw_errors']['rotation_rpy']
            
            axes[1, 0].hist(rotation_errors_rpy[:, 0], bins=30, edgecolor='black')
            axes[1, 0].set_xlabel('Roll Error (rad)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Roll Error Distribution')
            axes[1, 0].axvline(metrics['rotation_rpy']['roll_mean'], color='r',
                              linestyle='--', label=f"Mean: {metrics['rotation_rpy']['roll_mean']:.4f} rad")
            axes[1, 0].legend()
            
            axes[1, 1].hist(rotation_errors_rpy[:, 1], bins=30, edgecolor='black')
            axes[1, 1].set_xlabel('Pitch Error (rad)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Pitch Error Distribution')
            axes[1, 1].axvline(metrics['rotation_rpy']['pitch_mean'], color='r',
                              linestyle='--', label=f"Mean: {metrics['rotation_rpy']['pitch_mean']:.4f} rad")
            axes[1, 1].legend()
            
            axes[1, 2].hist(rotation_errors_rpy[:, 2], bins=30, edgecolor='black')
            axes[1, 2].set_xlabel('Yaw Error (rad)')
            axes[1, 2].set_ylabel('Frequency')
            axes[1, 2].set_title('Yaw Error Distribution')
            axes[1, 2].axvline(metrics['rotation_rpy']['yaw_mean'], color='r',
                              linestyle='--', label=f"Mean: {metrics['rotation_rpy']['yaw_mean']:.4f} rad")
            axes[1, 2].legend()
        else:
            # 仅显示 X, Y, Z 误差
            axes[1, 0].hist(position_errors_xyz[:, 0], bins=30, edgecolor='black')
            axes[1, 0].set_xlabel('X Error (m)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('X Position Error Distribution')
            
            axes[1, 1].hist(position_errors_xyz[:, 1], bins=30, edgecolor='black')
            axes[1, 1].set_xlabel('Y Error (m)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Y Position Error Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Error distribution plot saved to {save_path}")
        else:
            plt.show()


def main():
    parser = argparse.ArgumentParser(description='Pallet Pose Evaluation')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the trained model')
    parser.add_argument('--dataset_dir', type=str, default=None,
                        help='Path to the dataset directory')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default=config.DEVICE,
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--plot', action='store_true',
                        help='Plot error distributions')
    parser.add_argument('--plot_path', type=str, default='error_distribution.png',
                        help='Path to save the plot')
    parser.add_argument('--output_format', type=str, default='euler',
                        choices=['euler', 'quaternion'],
                        help='Output rotation format: euler or quaternion')
    
    args = parser.parse_args()
    
    # 加载数据
    print(f"Loading dataset from {args.dataset_dir or 'default location'}...")
    _, _, test_loader = get_dataloaders(args.dataset_dir, args.batch_size)
    
    # 创建评估器
    evaluator = PalletPoseEvaluator(args.model, args.device)
    
    # 评估
    output_euler = (args.output_format == 'euler')
    metrics = evaluator.evaluate(test_loader, output_euler=output_euler)
    
    # 打印结果
    evaluator.print_metrics(metrics, output_euler=output_euler)
    
    # 绘图
    if args.plot:
        evaluator.plot_error_distribution(metrics, args.plot_path, output_euler=output_euler)


if __name__ == '__main__':
    main()
