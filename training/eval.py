# eval.py
import torch
import numpy as np
from tqdm import tqdm
import logging
import os

import config
from dataset import get_dataloaders
from model import PalletPoseEstimator, PalletPoseLoss


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def quaternion_to_euler(q):
    """
    四元数转欧拉角
    Args:
        q: [qx, qy, qz, qw]
    Returns:
        [roll, pitch, yaw] in degrees
    """
    qx, qy, qz, qw = q
    
    # Roll (x-axis)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis)
    sinp = 2 * (qw * qy - qz * qx)
    sinp = np.clip(sinp, -1, 1)
    pitch = np.arcsin(sinp)
    
    # Yaw (z-axis)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.array([np.degrees(roll), np.degrees(pitch), np.degrees(yaw)])


def evaluate(model, test_loader, device=config.DEVICE):
    """评估模型"""
    model.eval()
    criterion = PalletPoseLoss()
    
    total_loss = 0.0
    position_errors = []
    rotation_errors = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Evaluating')
        for batch in pbar:
            rgb = batch['rgb'].to(device)
            depth = batch['depth'].to(device)
            position_gt = batch['position'].to(device)
            rotation_gt = batch['rotation'].to(device)
            
            # 前向传播
            position_pred, rotation_pred = model(rgb, depth)
            
            # 计算损失
            loss, _, _ = criterion(position_pred, rotation_pred, 
                                   position_gt, rotation_gt)
            total_loss += loss.item()
            
            # 计算位置误差（L2 距离，单位：米）
            pos_error = torch.norm(position_pred - position_gt, dim=1)
            position_errors.extend(pos_error.cpu().numpy())
            
            # 计算旋转误差（四元数距离）
            # 使用两个四元数之间的夹角作为旋转误差
            cos_sim = torch.nn.functional.cosine_similarity(
                rotation_pred, rotation_gt, dim=1, eps=1e-8
            )
            # 限制在 [-1, 1]
            cos_sim = torch.clamp(cos_sim, -1, 1)
            rot_angle = torch.acos(torch.abs(cos_sim))
            rot_error_deg = torch.rad2deg(rot_angle)
            rotation_errors.extend(rot_error_deg.cpu().numpy())
    
    # 统计信息
    position_errors = np.array(position_errors)
    rotation_errors = np.array(rotation_errors)
    
    results = {
        'avg_loss': total_loss / len(test_loader),
        'position_error_mean': position_errors.mean(),
        'position_error_std': position_errors.std(),
        'position_error_median': np.median(position_errors),
        'position_error_max': position_errors.max(),
        'position_error_min': position_errors.min(),
        'rotation_error_mean': rotation_errors.mean(),
        'rotation_error_std': rotation_errors.std(),
        'rotation_error_median': np.median(rotation_errors),
        'rotation_error_max': rotation_errors.max(),
        'rotation_error_min': rotation_errors.min(),
    }
    
    return results


def main():
    logger.info('=' * 80)
    logger.info('Pallet Pose Estimation Evaluation')
    logger.info('=' * 80)
    
    # 加载数据集
    logger.info('Loading dataset...')
    _, _, test_loader = get_dataloaders()
    
    # 加载模型
    logger.info('Loading model...')
    model = PalletPoseEstimator()
    
    best_model_path = os.path.join(config.MODEL_DIR, 'checkpoint_epoch_055.pth')
    if not os.path.exists(best_model_path):
        logger.error(f'Model not found: {best_model_path}')
        return
    
    checkpoint = torch.load(best_model_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.DEVICE)  # <-- 加这一行
    logger.info(f'Model loaded from {best_model_path}')
    
    # 评估
    logger.info('Evaluating...')
    results = evaluate(model, test_loader)
    
    # 打印结果
    logger.info('=' * 80)
    logger.info('Evaluation Results:')
    logger.info('=' * 80)
    logger.info(f'Average Loss: {results["avg_loss"]:.6f}')
    logger.info('')
    logger.info('Position Error (meters):')
    logger.info(f'  Mean:   {results["position_error_mean"]:.6f}')
    logger.info(f'  Std:    {results["position_error_std"]:.6f}')
    logger.info(f'  Median: {results["position_error_median"]:.6f}')
    logger.info(f'  Min:    {results["position_error_min"]:.6f}')
    logger.info(f'  Max:    {results["position_error_max"]:.6f}')
    logger.info('')
    logger.info('Rotation Error (degrees):')
    logger.info(f'  Mean:   {results["rotation_error_mean"]:.6f}')
    logger.info(f'  Std:    {results["rotation_error_std"]:.6f}')
    logger.info(f'  Median: {results["rotation_error_median"]:.6f}')
    logger.info(f'  Min:    {results["rotation_error_min"]:.6f}')
    logger.info(f'  Max:    {results["rotation_error_max"]:.6f}')
    logger.info('=' * 80)
    
    # 保存结果
    result_file = os.path.join(config.RESULT_DIR, 'eval_results.txt')
    with open(result_file, 'w') as f:
        f.write('Pallet Pose Estimation Evaluation Results\n')
        f.write('=' * 80 + '\n')
        f.write(f'Average Loss: {results["avg_loss"]:.6f}\n\n')
        f.write('Position Error (meters):\n')
        for key in ['position_error_mean', 'position_error_std', 'position_error_median', 
                    'position_error_min', 'position_error_max']:
            f.write(f'  {key}: {results[key]:.6f}\n')
        f.write('\nRotation Error (degrees):\n')
        for key in ['rotation_error_mean', 'rotation_error_std', 'rotation_error_median',
                    'rotation_error_min', 'rotation_error_max']:
            f.write(f'  {key}: {results[key]:.6f}\n')
    
    logger.info(f'Results saved to {result_file}')


if __name__ == '__main__':
    main()
