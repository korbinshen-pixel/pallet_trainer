import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
import logging

import config
from dataset import get_dataloaders
from model import PalletPoseEstimator, PalletPoseLoss


# ============ 日志配置 ============
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config.LOG_DIR, 'train.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Trainer:
    """训练器类，封装训练流程"""
    
    def __init__(self, model, train_loader, val_loader, device=config.DEVICE):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # 损失函数
        self.criterion = PalletPoseLoss(
            position_weight=config.POSITION_LOSS_WEIGHT,
            rotation_weight=config.ROTATION_LOSS_WEIGHT
        )
        
        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # 学习率调度器
        # 线性预热 + 余弦衰减
        self.warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=len(train_loader) * config.WARMUP_EPOCHS
        )
        
        self.cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=len(train_loader) * (config.NUM_EPOCHS - config.WARMUP_EPOCHS),
            eta_min=1e-6
        )
        
        # TensorBoard
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(
            os.path.join(config.LOG_DIR, f'run_{timestamp}')
        )
        
        # 统计信息
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.global_step = 0
    
    def train_epoch(self, epoch):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        total_pos_loss = 0.0
        total_rot_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{config.NUM_EPOCHS}')
        
        for batch_idx, batch in enumerate(pbar):
            rgb = batch['rgb'].to(self.device)
            depth = batch['depth'].to(self.device)
            position_gt = batch['position'].to(self.device)
            rotation_gt = batch['rotation'].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            position_pred, rotation_pred = self.model(rgb, depth)
            
            # 计算损失
            loss, pos_loss, rot_loss = self.criterion(
                position_pred, rotation_pred,
                position_gt, rotation_gt
            )
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 更新学习率
            if epoch < config.WARMUP_EPOCHS:
                self.warmup_scheduler.step()
            else:
                self.cosine_scheduler.step()
            
            # 统计
            total_loss += loss.item()
            total_pos_loss += pos_loss.item()
            total_rot_loss += rot_loss.item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': loss.item(),
                'pos_loss': pos_loss.item(),
                'rot_loss': rot_loss.item(),
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            # TensorBoard 记录（每100步）
            if batch_idx % 100 == 0:
                self.writer.add_scalar('Train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('Train/pos_loss', pos_loss.item(), self.global_step)
                self.writer.add_scalar('Train/rot_loss', rot_loss.item(), self.global_step)
                self.writer.add_scalar('Train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
            
            self.global_step += 1
        
        avg_loss = total_loss / len(self.train_loader)
        avg_pos_loss = total_pos_loss / len(self.train_loader)
        avg_rot_loss = total_rot_loss / len(self.train_loader)
        
        logger.info(f'Epoch {epoch+1} Train - Loss: {avg_loss:.6f}, '
                   f'Pos Loss: {avg_pos_loss:.6f}, Rot Loss: {avg_rot_loss:.6f}')
        
        # TensorBoard 记录 epoch 指标
        self.writer.add_scalar('Epoch/train_loss', avg_loss, epoch)
        self.writer.add_scalar('Epoch/train_pos_loss', avg_pos_loss, epoch)
        self.writer.add_scalar('Epoch/train_rot_loss', avg_rot_loss, epoch)
        
        return avg_loss
    
    def validate_epoch(self, epoch):
        """验证一个 epoch"""
        self.model.eval()
        total_loss = 0.0
        total_pos_loss = 0.0
        total_rot_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validating'):
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
        
        avg_loss = total_loss / len(self.val_loader)
        avg_pos_loss = total_pos_loss / len(self.val_loader)
        avg_rot_loss = total_rot_loss / len(self.val_loader)
        
        logger.info(f'Epoch {epoch+1} Valid - Loss: {avg_loss:.6f}, '
                   f'Pos Loss: {avg_pos_loss:.6f}, Rot Loss: {avg_rot_loss:.6f}')
        
        # TensorBoard 记录
        self.writer.add_scalar('Epoch/val_loss', avg_loss, epoch)
        self.writer.add_scalar('Epoch/val_pos_loss', avg_pos_loss, epoch)
        self.writer.add_scalar('Epoch/val_rot_loss', avg_rot_loss, epoch)
        
        # Early stopping 和模型保存
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.patience_counter = 0
            
            # 保存最好的模型
            best_model_path = os.path.join(config.MODEL_DIR, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_val_loss': self.best_val_loss,
            }, best_model_path)
            logger.info(f'Best model saved to {best_model_path}')
        else:
            self.patience_counter += 1
            logger.info(f'No improvement. Patience: {self.patience_counter}/{config.PATIENCE}')
        
        return avg_loss
    
    def save_checkpoint(self, epoch):
        """保存检查点"""
        checkpoint_path = os.path.join(
            config.CHECKPOINT_DIR,
            f'checkpoint_epoch_{epoch+1:03d}.pth'
        )
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
        }, checkpoint_path)
        logger.info(f'Checkpoint saved to {checkpoint_path}')
    
    def train(self):
        """完整训练循环"""
        logger.info('Starting training...')
        logger.info(f'Device: {self.device}')
        logger.info(f'Model: {self.model}')
        
        for epoch in range(config.NUM_EPOCHS):
            # 训练
            self.train_epoch(epoch)
            
            # 验证
            if (epoch + 1) % config.EVAL_INTERVAL == 0:
                self.validate_epoch(epoch)
            
            # 保存检查点
            if (epoch + 1) % config.SAVE_INTERVAL == 0:
                self.save_checkpoint(epoch)
            
            # Early stopping
            if self.patience_counter >= config.PATIENCE:
                logger.info(f'Early stopping at epoch {epoch+1}')
                break
        
        logger.info('Training finished!')
        self.writer.close()


def main():
    """主函数"""
    logger.info('=' * 80)
    logger.info('Pallet Pose Estimation Training')
    logger.info('=' * 80)
    
    # 加载数据集
    logger.info('Loading dataset...')
    train_loader, val_loader, test_loader = get_dataloaders()
    
    # 创建模型
    logger.info('Creating model...')
    model = PalletPoseEstimator(pretrained=config.PRETRAINED)
    logger.info(f'Model: {config.MODEL_NAME}')
    logger.info(f'Pretrained: {config.PRETRAINED}')
    
    # 创建训练器
    trainer = Trainer(model, train_loader, val_loader)
    
    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()
