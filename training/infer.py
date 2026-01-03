import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import argparse
from pathlib import Path

import config
from model import PalletPoseEstimator
from dataset import quaternion_to_euler


class PalletPoseInference:
    """托盘位姿推理类"""
    
    def __init__(self, model_path, device=config.DEVICE, output_euler=True):
        """
        Args:
            model_path: 保存的模型路径
            device: 运行设备
            output_euler: 是否输出欧拉角（True）或四元数（False）
        """
        self.device = torch.device(device)
        self.output_euler = output_euler
        
        # 加载模型
        self.model = PalletPoseEstimator(pretrained=False)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded from {model_path}")
        print(f"Output format: {'Euler angles (roll, pitch, yaw)' if output_euler else 'Quaternion (qx, qy, qz, qw)'}")
        
        # RGB 预处理
        self.rgb_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=config.RGB_MEAN, std=config.RGB_STD)
        ])
    
    def preprocess_depth(self, depth):
        """深度预处理"""
        if isinstance(depth, torch.Tensor):
            depth_tensor = depth
        else:
            depth_tensor = torch.from_numpy(depth)
        
        if depth_tensor.dim() == 2:
            depth_tensor = depth_tensor.unsqueeze(0)
        
        # 归一化
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
    
    def infer(self, rgb_path, depth_path):
        """
        对单个 RGB/Depth 图像对进行推理
        
        Args:
            rgb_path: RGB 图像路径
            depth_path: 深度图像路径 (16bit PNG)
        
        Returns:
            position: (3,) numpy array - (x, y, z)
            rotation: (3,) or (4,) numpy array - Euler angles (roll, pitch, yaw) or Quaternion
        """
        # 加载图像
        rgb = Image.open(rgb_path).convert('RGB')
        depth_img = Image.open(depth_path)
        depth = np.array(depth_img, dtype=np.float32)
        
        # 预处理
        rgb_tensor = self.rgb_transform(rgb).unsqueeze(0).to(self.device)  # (1, 3, H, W)
        depth_tensor = self.preprocess_depth(depth).unsqueeze(0).to(self.device)  # (1, 1, H, W)
        
        # 推理
        with torch.no_grad():
            position, rotation_quat = self.model(rgb_tensor, depth_tensor)
        
        # 转换为 numpy
        position = position.cpu().numpy()[0]  # (3,)
        rotation_quat = rotation_quat.cpu().numpy()[0]  # (4,) [qx, qy, qz, qw]
        
        # 如果需要，转换为欧拉角
        if self.output_euler:
            roll, pitch, yaw = quaternion_to_euler(
                rotation_quat[0], rotation_quat[1], 
                rotation_quat[2], rotation_quat[3]
            )
            rotation = np.array([roll, pitch, yaw], dtype=np.float32)
        else:
            rotation = rotation_quat
        
        return position, rotation
    
    def infer_batch(self, rgb_dir, depth_dir, output_file=None):
        """
        对一个批次的图像执行推理
        
        Args:
            rgb_dir: RGB 图像目录
            depth_dir: 深度图像目录
            output_file: 输出结果文件路径
        
        Returns:
            results: list of (frame_id, position, rotation)
        """
        import glob
        
        rgb_files = sorted(glob.glob(os.path.join(rgb_dir, 'rgb_*.png')))
        results = []
        
        print(f"Processing {len(rgb_files)} images...")
        
        for i, rgb_path in enumerate(rgb_files):
            frame_id = os.path.basename(rgb_path).replace('rgb_', '').replace('.png', '')
            depth_path = os.path.join(depth_dir, f'depth_{frame_id}.png')
            
            if not os.path.exists(depth_path):
                print(f"Warning: Depth image not found for {frame_id}")
                continue
            
            position, rotation = self.infer(rgb_path, depth_path)
            results.append((frame_id, position, rotation))
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(rgb_files)} images")
        
        # 保存结果
        if output_file:
            self.save_results(results, output_file)
        
        return results
    
    def save_results(self, results, output_file):
        """
        保存推理结果
        
        格式:
        - 欧拉角: frame_id x y z roll pitch yaw
        - 四元数: frame_id x y z qx qy qz qw
        """
        with open(output_file, 'w') as f:
            for frame_id, position, rotation in results:
                line = f"{frame_id} "
                line += f"{position[0]:.6f} {position[1]:.6f} {position[2]:.6f} "
                
                if self.output_euler:
                    # 欧拉角格式
                    line += f"{rotation[0]:.6f} {rotation[1]:.6f} {rotation[2]:.6f}"
                else:
                    # 四元数格式
                    line += f"{rotation[0]:.6f} {rotation[1]:.6f} {rotation[2]:.6f} {rotation[3]:.6f}"
                
                f.write(line + '\n')
        
        format_name = "Euler angles" if self.output_euler else "Quaternion"
        print(f"Results saved to {output_file} (format: {format_name})")


def main():
    parser = argparse.ArgumentParser(description='Pallet Pose Inference')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the trained model')
    parser.add_argument('--rgb_dir', type=str, default=None,
                        help='Directory of RGB images for batch inference')
    parser.add_argument('--depth_dir', type=str, default=None,
                        help='Directory of depth images for batch inference')
    parser.add_argument('--rgb_path', type=str, default=None,
                        help='Path to a single RGB image')
    parser.add_argument('--depth_path', type=str, default=None,
                        help='Path to a single depth image')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file to save results')
    parser.add_argument('--device', type=str, default=config.DEVICE,
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--output_format', type=str, default='euler',
                        choices=['euler', 'quaternion'],
                        help='Output rotation format: euler (roll,pitch,yaw) or quaternion (qx,qy,qz,qw)')
    
    args = parser.parse_args()
    
    # 创建推理器
    output_euler = (args.output_format == 'euler')
    inferer = PalletPoseInference(args.model, args.device, output_euler=output_euler)
    
    if args.rgb_dir and args.depth_dir:
        # 批量推理
        print(f"Batch inference from {args.rgb_dir} and {args.depth_dir}")
        results = inferer.infer_batch(args.rgb_dir, args.depth_dir, args.output)
        
        print(f"\nInference results (first 5):")
        for frame_id, position, rotation in results[:5]:
            if output_euler:
                print(f"  {frame_id}: pos={position}, euler(rad)={rotation}")
                print(f"           euler(deg)=[{np.rad2deg(rotation[0]):.2f}, {np.rad2deg(rotation[1]):.2f}, {np.rad2deg(rotation[2]):.2f}]")
            else:
                print(f"  {frame_id}: pos={position}, quat={rotation}")
    
    elif args.rgb_path and args.depth_path:
        # 单图推理
        print(f"Inference from {args.rgb_path} and {args.depth_path}")
        position, rotation = inferer.infer(args.rgb_path, args.depth_path)
        
        print(f"\nInference result:")
        print(f"  Position (m): {position}")
        
        if output_euler:
            print(f"  Rotation (Euler, rad): {rotation}")
            print(f"  Rotation (Euler, deg): [{np.rad2deg(rotation[0]):.2f}, {np.rad2deg(rotation[1]):.2f}, {np.rad2deg(rotation[2]):.2f}]")
        else:
            print(f"  Rotation (Quaternion): {rotation}")
            print(f"  Quaternion norm: {np.linalg.norm(rotation):.6f}")
    
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  # Single image with Euler output")
        print("  python infer.py --model models/best_model.pth --rgb_path test_rgb.png --depth_path test_depth.png --output_format euler")
        print("\n  # Batch inference with Euler output")
        print("  python infer.py --model models/best_model.pth --rgb_dir ./rgb --depth_dir ./depth --output results.txt --output_format euler")
        print("\n  # Batch inference with Quaternion output")
        print("  python infer.py --model models/best_model.pth --rgb_dir ./rgb --depth_dir ./depth --output results.txt --output_format quaternion")


if __name__ == '__main__':
    main()
