import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import argparse
from pathlib import Path

import config
from model import PalletPoseEstimator


class PalletPoseInference:
    """托盘位姿推理類"""
    
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
        
        # RGB 美化变换
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
            position: (3,) numpy array
            rotation: (4,) numpy array
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
            position, rotation = self.model(rgb_tensor, depth_tensor)
        
        # 转换为 numpy
        position = position.cpu().numpy()[0]  # (3,)
        rotation = rotation.cpu().numpy()[0]  # (4,)
        
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
        """保存推理结果
        
        格式: frame_id x y z qx qy qz qw
        """
        with open(output_file, 'w') as f:
            for frame_id, position, rotation in results:
                line = f"{frame_id} "
                line += f"{position[0]} {position[1]} {position[2]} "
                line += f"{rotation[0]} {rotation[1]} {rotation[2]} {rotation[3]}"
                f.write(line + '\n')
        
        print(f"Results saved to {output_file}")


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
    
    args = parser.parse_args()
    
    # 创建推理器
    inferer = PalletPoseInference(args.model, args.device)
    
    if args.rgb_dir and args.depth_dir:
        # 批量推理
        print(f"Batch inference from {args.rgb_dir} and {args.depth_dir}")
        results = inferer.infer_batch(args.rgb_dir, args.depth_dir, args.output)
        
        print(f"\nInference results:")
        for frame_id, position, rotation in results[:5]:
            print(f"  {frame_id}: pos={position}, rot={rotation}")
    
    elif args.rgb_path and args.depth_path:
        # 单图推理
        print(f"Inference from {args.rgb_path} and {args.depth_path}")
        position, rotation = inferer.infer(args.rgb_path, args.depth_path)
        
        print(f"\nInference result:")
        print(f"  Position: {position}")
        print(f"  Rotation: {rotation}")
        print(f"  Rotation norm: {np.linalg.norm(rotation)}")
    
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  python infer.py --model models/best_model.pth --rgb_path test_rgb.png --depth_path test_depth.png")
        print("  python infer.py --model models/best_model.pth --rgb_dir ./rgb --depth_dir ./depth --output results.txt")


if __name__ == '__main__':
    main()
