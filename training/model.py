import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import config


class PalletPoseEstimator(nn.Module):
    """
    多任务学习的托盘位姿估计模型
    
    输入：RGB 图像 + 深度图像
    输出：3D 位置 (x,y,z) + 四元数 (qx,qy,qz,qw)
    """
    
    def __init__(self, pretrained=config.PRETRAINED):
        super(PalletPoseEstimator, self).__init__()
        
        # ============ RGB 分支 ============
        if pretrained:
            # 用 ImageNet 预训练的 ResNet-50
            self.rgb_backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.rgb_backbone = resnet50(weights=None)
        
        # 去掉最后的分类层，保留卷积特征
        self.rgb_backbone = nn.Sequential(*list(self.rgb_backbone.children())[:-1])
        
        # ============ 深度分支 ============
        # 用相同的 ResNet-50 架构，但不预训练（深度是单通道，不能直接用 ImageNet 预训练）
        self.depth_backbone = resnet50(weights=None)
        self.depth_backbone = nn.Sequential(*list(self.depth_backbone.children())[:-1])
        
        # ============ 特征融合 ============
        # RGB 和 Depth 各输出 2048 维特征，拼接后 4096 维
        self.feature_fusion = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3)
        )
        
        # ============ 位置回归头 ============
        self.position_head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3)  # x, y, z
        )
        
        # ============ 旋转回归头 ============
        self.rotation_head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4)  # qx, qy, qz, qw
        )
    
    def forward(self, rgb, depth):
        """
        Args:
            rgb: (B, 3, H, W) - RGB 图像
            depth: (B, 1, H, W) - 深度图像
        
        Returns:
            position: (B, 3) - 位置
            rotation: (B, 4) - 四元数
        """
        # RGB 特征提取
        rgb_feat = self.rgb_backbone(rgb)  # (B, 2048, 1, 1)
        rgb_feat = rgb_feat.view(rgb_feat.size(0), -1)  # (B, 2048)
        
        # 深度特征提取
        # 确保维度是 (B, 1, H, W)
        if depth.dim() == 3:
            # 可能是 (B, H, W)，补上通道维
            depth = depth.unsqueeze(1)
        elif depth.dim() == 2:
            # 单张 (H, W)，补上 batch 和通道维
            depth = depth.unsqueeze(0).unsqueeze(0)
        
        # 如果已经是 (B, 1, H, W)，保持不变
        assert depth.dim() == 4, f"Depth tensor must be 4D, got shape {depth.shape}"
        
        # ResNet 期望 3 通道输入：单通道复制到 3 通道
        if depth.size(1) == 1:
            depth_3ch = depth.repeat(1, 3, 1, 1)  # (B, 3, H, W)
        else:
            # 如果已经是 3 通道，就直接用
            depth_3ch = depth

        depth_feat = self.depth_backbone(depth_3ch)  # (B, 2048, 1, 1)
        depth_feat = depth_feat.view(depth_feat.size(0), -1)  # (B, 2048)
        
        # 特征融合
        fused = torch.cat([rgb_feat, depth_feat], dim=1)  # (B, 4096)
        fused = self.feature_fusion(fused)  # (B, 1024)
        
        # 位置和旋转预测
        position = self.position_head(fused)  # (B, 3)
        rotation = self.rotation_head(fused)  # (B, 4)
        
        # 四元数归一化（保证 ||q|| = 1）
        rotation = rotation / (torch.norm(rotation, dim=1, keepdim=True) + 1e-8)
        
        return position, rotation


# 损失函数定义
class PalletPoseLoss(nn.Module):
    """
    多任务损失：位置损失 + 旋转损失
    """
    
    def __init__(self, position_weight=config.POSITION_LOSS_WEIGHT,
                 rotation_weight=config.ROTATION_LOSS_WEIGHT):
        super(PalletPoseLoss, self).__init__()
        self.position_weight = position_weight
        self.rotation_weight = rotation_weight
        
        # 位置使用 MSE 损失
        self.position_loss_fn = nn.MSELoss()
        
        # 旋转使用 cosine distance 损失（四元数相似度）
        self.rotation_loss_fn = nn.CosineSimilarity(dim=1, eps=1e-8)
    
    def forward(self, pred_position, pred_rotation, 
                target_position, target_rotation):
        """
        Args:
            pred_position: (B, 3) - 预测位置
            pred_rotation: (B, 4) - 预测四元数
            target_position: (B, 3) - 目标位置
            target_rotation: (B, 4) - 目标四元数
        
        Returns:
            total_loss, position_loss, rotation_loss
        """
        # 位置损失
        position_loss = self.position_loss_fn(pred_position, target_position)
        
        # 旋转损失（四元数余弦相似度）
        cos_sim = self.rotation_loss_fn(pred_rotation, target_rotation)
        # 相似度范围 [-1, 1]，转换为距离损失
        rotation_loss = (1 - cos_sim.abs()).mean()
        
        # 加权总损失
        total_loss = (self.position_weight * position_loss + 
                      self.rotation_weight * rotation_loss)
        
        return total_loss, position_loss, rotation_loss


if __name__ == '__main__':
    # 测试模型
    print("Testing model...")
    model = PalletPoseEstimator()
    print(f"Model created: {model}")
    
    # 虚拟输入
    rgb = torch.randn(4, 3, 480, 640)
    depth = torch.randn(4, 1, 480, 640)
    
    pos, rot = model(rgb, depth)
    print(f"Position shape: {pos.shape}, Rotation shape: {rot.shape}")
    print(f"Position sample: {pos[0]}")
    print(f"Rotation sample: {rot[0]}, norm: {torch.norm(rot[0])}")
    print("Model test passed!")
