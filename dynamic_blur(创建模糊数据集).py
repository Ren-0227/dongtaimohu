import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import torch
import torch.nn.functional as F
import glob
import os
class DynamicBlur:
    """动态模糊算法类"""
    
    def __init__(self, kernel_size=15, angle_range=(-45, 45), length_range=(5, 15)):
        """
        初始化动态模糊参数
        Args:
            kernel_size: 模糊核大小
            angle_range: 运动角度范围(度)
            length_range: 运动长度范围(像素)
        """
        self.kernel_size = kernel_size
        self.angle_range = angle_range
        self.length_range = length_range
        
    def generate_motion_kernel(self, angle, length):
        """
        生成运动模糊核
        Args:
            angle: 运动角度(度)
            length: 运动长度(像素)
        Returns:
            motion_kernel: 运动模糊核
        """
        # 创建空核
        kernel = np.zeros((self.kernel_size, self.kernel_size))
        
        # 计算中心点
        center = self.kernel_size // 2
        
        # 将角度转换为弧度
        angle_rad = np.deg2rad(angle)
        
        # 计算运动方向
        dx = length * np.cos(angle_rad)
        dy = length * np.sin(angle_rad)
        
        # 在核上绘制运动轨迹
        cv2.line(kernel, 
                (center, center),
                (int(center + dx), int(center + dy)),
                1, 1)
        
        # 归一化核
        kernel = kernel / np.sum(kernel)
        
        return kernel
    
    def apply_motion_blur(self, image, angle=None, length=None):
        """
        应用运动模糊
        Args:
            image: 输入图像 (H, W, C) 或 (B, C, H, W)
            angle: 运动角度(度)，如果为None则随机生成
            length: 运动长度(像素)，如果为None则随机生成
        Returns:
            blurred_image: 模糊后的图像
        """
        # 确保输入是numpy数组
        if isinstance(image, torch.Tensor):
            is_tensor = True
            if image.dim() == 4:  # (B, C, H, W)
                batch_size = image.size(0)
                image = image.cpu().numpy()
            else:  # (C, H, W)
                image = image.cpu().numpy()
                image = np.expand_dims(image, 0)
                batch_size = 1
        else:
            is_tensor = False
            if image.ndim == 3:  # (H, W, C)
                image = np.expand_dims(image, 0)
                batch_size = 1
            else:  # (B, H, W, C)
                batch_size = image.shape[0]
        
        # 随机生成运动参数
        if angle is None:
            angle = np.random.uniform(*self.angle_range)
        if length is None:
            length = np.random.uniform(*self.length_range)
        
        # 生成运动模糊核
        kernel = self.generate_motion_kernel(angle, length)
        
        # 对每个批次应用模糊
        blurred_images = []
        for i in range(batch_size):
            if is_tensor:
                # 处理PyTorch张量格式 (B, C, H, W)
                blurred = np.zeros_like(image[i])
                for c in range(image.shape[1]):
                    blurred[c] = cv2.filter2D(image[i, c], -1, kernel)
            else:
                # 处理NumPy数组格式 (B, H, W, C)
                blurred = cv2.filter2D(image[i], -1, kernel)
            blurred_images.append(blurred)
        
        # 合并结果
        blurred_images = np.stack(blurred_images)
        
        # 转换回原始格式
        if is_tensor:
            blurred_images = torch.from_numpy(blurred_images).float()
            if batch_size == 1 and image.dim() == 3:
                blurred_images = blurred_images.squeeze(0)
        else:
            if batch_size == 1:
                blurred_images = blurred_images.squeeze(0)
        
        return blurred_images
    
    def apply_random_blur(self, image, num_blurs=3):
        """
        应用多次随机运动模糊
        Args:
            image: 输入图像
            num_blurs: 模糊次数
        Returns:
            blurred_image: 多次模糊后的图像
        """
        blurred = image.copy()
        for _ in range(num_blurs):
            blurred = self.apply_motion_blur(blurred)
        return blurred
    
    def apply_complex_blur(self, image, num_paths=3):
        """
        应用复杂运动模糊（多条运动路径）
        Args:
            image: 输入图像
            num_paths: 运动路径数量
        Returns:
            blurred_image: 复杂模糊后的图像
        """
        # 生成多条运动路径的参数
        angles = np.random.uniform(*self.angle_range, num_paths)
        lengths = np.random.uniform(*self.length_range, num_paths)
        
        # 对每条路径应用模糊
        blurred = image.copy()
        for angle, length in zip(angles, lengths):
            blurred = self.apply_motion_blur(blurred, angle, length)
        
        return blurred

def create_blur_dataset(clear_images, blur_generator, output_dir):
    """
    创建模糊数据集
    Args:
        clear_images: 清晰图像列表或目录
        blur_generator: DynamicBlur实例
        output_dir: 输出目录
    """
    import os
    from PIL import Image
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'blur'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'sharp'), exist_ok=True)
    
    # 处理每张图像
    for i, image_path in enumerate(clear_images):
        # 读取图像
        image = Image.open(image_path)
        image = np.array(image)
        
        # 应用复杂模糊
        blurred = blur_generator.apply_complex_blur(image)
        
        # 保存图像
        Image.fromarray(blurred.astype(np.uint8)).save(
            os.path.join(output_dir, 'blur', f'blur_{i:04d}.png'))
        Image.fromarray(image).save(
            os.path.join(output_dir, 'sharp', f'sharp_{i:04d}.png'))

if __name__ == '__main__':
    blur_generator = DynamicBlur(
        kernel_size=15,
        angle_range=(-45, 45),
        length_range=(5, 15)
    )

    # 测试集原图路径
    test_images_dir = r'C:\Users\91144\Desktop\dongtaimohuhe\test_images'
    # 输出模糊数据集路径
    output_dir = r'C:\Users\91144\Desktop\dongtaimohuhe\test_blur_dataset'

    # 获取所有图片路径（支持jpg/png等）
    image_paths = glob.glob(os.path.join(test_images_dir, '*.jpg')) + \
                  glob.glob(os.path.join(test_images_dir, '*.png')) + \
                  glob.glob(os.path.join(test_images_dir, '*.jpeg'))

    print(f"共找到{len(image_paths)}张测试图片，开始生成模糊数据集...")

    create_blur_dataset(image_paths, blur_generator, output_dir)

    print("模糊数据集生成完毕！")