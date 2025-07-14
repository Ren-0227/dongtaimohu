import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

def center_crop(img, target_size):
    """中心裁剪图片"""
    h, w = img.shape[:2]
    th, tw = target_size
    
    # 计算裁剪的起始位置
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    
    return img[i:i+th, j:j+tw]

def preprocess_images(input_dir, output_dir, target_size=(512, 512)):
    """预处理图片：调整大小"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图片文件
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Found {len(image_files)} images in {input_dir}")
    
    # 定义图像变换
    transform = transforms.Compose([
        transforms.Resize(target_size, transforms.InterpolationMode.LANCZOS),
    ])
    
    # 处理每张图片
    for img_file in tqdm(image_files, desc="Processing images"):
        try:
            # 读取图片
            img_path = os.path.join(input_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to read image: {img_path}")
                continue
            
            # 转换为RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 转换为PIL图像
            img_pil = Image.fromarray(img)
            
            # 调整大小
            img_resized = transform(img_pil)
            
            # 保存处理后的图片
            output_path = os.path.join(output_dir, img_file)
            img_resized.save(output_path, quality=95)
            
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
            continue

def main():
    # 设置输入输出目录
    blur_dir = './dataset/blur'
    sharp_dir = './dataset/sharp'
    processed_blur_dir = './dataset/processed/blur'
    processed_sharp_dir = './dataset/processed/sharp'
    
    # 处理模糊图片
    print("\nProcessing blur images...")
    preprocess_images(blur_dir, processed_blur_dir)
    
    # 处理清晰图片
    print("\nProcessing sharp images...")
    preprocess_images(sharp_dir, processed_sharp_dir)
    
    print("\nPreprocessing completed!")

if __name__ == '__main__':
    main() 