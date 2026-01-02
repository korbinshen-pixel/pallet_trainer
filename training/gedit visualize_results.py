# visualize_results.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import logging

import config
from dataset import get_dataloaders
from model import PalletPoseEstimator


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_detailed(model, test_loader, device=config.DEVICE, num_samples=100):
    """
    详细评估：返回逐样本的误差，用于可视化
    """
    model.eval()
    
    position_errors = []
    rotation_errors = []
    predictions = []
    ground_truths = []
    
    sample_count = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Detailed Evaluation'):
            rgb = batch['rgb'].to(device)
            depth = batch['depth'].to(device)
            position_gt = batch['position'].to(device)
            rotation_gt = batch['rotation'].to(device)
            
            # 前向传播
            position_pred, rotation_pred = model(rgb, depth)
            
            # 位置误差
            pos_error = torch.norm(position_pred - position_gt, dim=1)
            position_errors.extend(pos_error.cpu().numpy())
            
            # 旋转误差（四元数夹角）
            cos_sim = torch.nn.functional.cosine_similarity(
                rotation_pred, rotation_gt, dim=1, eps=1e-8
            )
            cos_sim = torch.clamp(cos_sim, -1, 1)
            rot_angle = torch.acos(torch.abs(cos_sim))
            rot_error_deg = torch.rad2deg(rot_angle)
            rotation_errors.extend(rot_error_deg.cpu().numpy())
            
            # 存储预测和真实值
            predictions.append({
                'position': position_pred.cpu().numpy(),
                'rotation': rotation_pred.cpu().numpy()
            })
            ground_truths.append({
                'position': position_gt.cpu().numpy(),
                'rotation': rotation_gt.cpu().numpy()
            })
            
            sample_count += len(pos_error)
            if sample_count >= num_samples:
                break
    
    return np.array(position_errors), np.array(rotation_errors), predictions, ground_truths


def plot_error_distributions(position_errors, rotation_errors, output_dir=config.RESULT_DIR):
    """绘制误差分布图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Evaluation - Error Distributions', fontsize=16, fontweight='bold')
    
    # 位置误差直方图
    axes[0, 0].hist(position_errors * 100, bins=50, color='blue', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(np.mean(position_errors) * 100, color='red', linestyle='--', 
                       label=f'Mean: {np.mean(position_errors)*100:.2f} cm')
    axes[0, 0].set_xlabel('Position Error (cm)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Position Error Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # 旋转误差直方图
    axes[0, 1].hist(rotation_errors, bins=50, color='green', alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(np.mean(rotation_errors), color='red', linestyle='--',
                       label=f'Mean: {np.mean(rotation_errors):.2f}°')
    axes[0, 1].set_xlabel('Rotation Error (degrees)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Rotation Error Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # 位置误差 CDF（累积分布函数）
    sorted_pos = np.sort(position_errors)
    cdf_pos = np.arange(1, len(sorted_pos) + 1) / len(sorted_pos)
    axes[1, 0].plot(sorted_pos * 100, cdf_pos, linewidth=2, color='blue')
    axes[1, 0].axvline(np.median(position_errors) * 100, color='red', linestyle='--',
                       label=f'Median: {np.median(position_errors)*100:.2f} cm')
    axes[1, 0].axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    axes[1, 0].set_xlabel('Position Error (cm)')
    axes[1, 0].set_ylabel('Cumulative Probability')
    axes[1, 0].set_title('Position Error CDF')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # 旋转误差 CDF
    sorted_rot = np.sort(rotation_errors)
    cdf_rot = np.arange(1, len(sorted_rot) + 1) / len(sorted_rot)
    axes[1, 1].plot(sorted_rot, cdf_rot, linewidth=2, color='green')
    axes[1, 1].axvline(np.median(rotation_errors), color='red', linestyle='--',
                       label=f'Median: {np.median(rotation_errors):.2f}°')
    axes[1, 1].axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    axes[1, 1].set_xlabel('Rotation Error (degrees)')
    axes[1, 1].set_ylabel('Cumulative Probability')
    axes[1, 1].set_title('Rotation Error CDF')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'error_distributions.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f'Error distribution plot saved to {output_path}')
    plt.close()


def plot_error_statistics(position_errors, rotation_errors, output_dir=config.RESULT_DIR):
    """绘制统计信息图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Model Evaluation - Statistics', fontsize=16, fontweight='bold')
    
    # 位置误差统计
    pos_stats = {
        'Mean': np.mean(position_errors) * 100,
        'Median': np.median(position_errors) * 100,
        'Std': np.std(position_errors) * 100,
        'Min': np.min(position_errors) * 100,
        'Max': np.max(position_errors) * 100,
        '95%ile': np.percentile(position_errors, 95) * 100,
    }
    
    axes[0].bar(pos_stats.keys(), pos_stats.values(), color='blue', alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('Position Error (cm)')
    axes[0].set_title('Position Error Statistics')
    axes[0].grid(alpha=0.3, axis='y')
    for i, (k, v) in enumerate(pos_stats.items()):
        axes[0].text(i, v + 0.1, f'{v:.2f}', ha='center', va='bottom')
    axes[0].tick_params(axis='x', rotation=45)
    
    # 旋转误差统计
    rot_stats = {
        'Mean': np.mean(rotation_errors),
        'Median': np.median(rotation_errors),
        'Std': np.std(rotation_errors),
        'Min': np.min(rotation_errors),
        'Max': np.max(rotation_errors),
        '95%ile': np.percentile(rotation_errors, 95),
    }
    
    axes[1].bar(rot_stats.keys(), rot_stats.values(), color='green', alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('Rotation Error (degrees)')
    axes[1].set_title('Rotation Error Statistics')
    axes[1].grid(alpha=0.3, axis='y')
    for i, (k, v) in enumerate(rot_stats.items()):
        axes[1].text(i, v + 0.2, f'{v:.2f}', ha='center', va='bottom')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'error_statistics.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f'Error statistics plot saved to {output_path}')
    plt.close()
    
    return pos_stats, rot_stats


def main():
    logger.info('=' * 80)
    logger.info('Detailed Model Evaluation')
    logger.info('=' * 80)
    
    # 加载数据集
    logger.info('Loading dataset...')
    _, _, test_loader = get_dataloaders()
    
    # 加载模型
    logger.info('Loading model...')
    model = PalletPoseEstimator()
    
    best_model_path = os.path.join(config.MODEL_DIR, 'best_model.pth')
    if not os.path.exists(best_model_path):
        logger.error(f'Model not found: {best_model_path}')
        return
    
    checkpoint = torch.load(best_model_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.DEVICE)  # <-- 加这一行！把模型移到 GPU/CPU
    logger.info(f'Model loaded from {best_model_path}')
    
    # 详细评估
    logger.info('Performing detailed evaluation...')
    position_errors, rotation_errors, _, _ = evaluate_detailed(model, test_loader, num_samples=1000)
    
    # 绘制分布图
    logger.info('Plotting error distributions...')
    plot_error_distributions(position_errors, rotation_errors)
    
    # 绘制统计图
    logger.info('Plotting error statistics...')
    pos_stats, rot_stats = plot_error_statistics(position_errors, rotation_errors)
    
    # 打印详细统计
    logger.info('=' * 80)
    logger.info('Detailed Statistics:')
    logger.info('=' * 80)
    logger.info('Position Error (cm):')
    for k, v in pos_stats.items():
        logger.info(f'  {k:10s}: {v:.4f}')
    
    logger.info('')
    logger.info('Rotation Error (degrees):')
    for k, v in rot_stats.items():
        logger.info(f'  {k:10s}: {v:.4f}')
    
    # 评估等级
    logger.info('')
    logger.info('=' * 80)
    logger.info('Evaluation Grade:')
    logger.info('=' * 80)
    
    pos_mean = pos_stats['Mean'] / 100  # 转换回米
    rot_mean = rot_stats['Mean']
    
    if pos_mean < 0.01 and rot_mean < 5:
        grade = '⭐⭐⭐⭐⭐ Excellent'
        advice = 'Ready for production use!'
    elif pos_mean < 0.05 and rot_mean < 15:
        grade = '⭐⭐⭐⭐ Good'
        advice = 'Can be used for most applications. Consider fine-tuning for better results.'
    elif pos_mean < 0.1 and rot_mean < 30:
        grade = '⭐⭐⭐ Fair'
        advice = 'Acceptable for non-critical applications. Recommend collecting more data and retraining.'
    else:
        grade = '⭐⭐ Poor'
        advice = 'Need significant improvements. Check data quality and model architecture.'
    
    logger.info(f'Grade: {grade}')
    logger.info(f'Advice: {advice}')
    logger.info('=' * 80)


if __name__ == '__main__':
    main()
