import cv2
import numpy as np
import os
from tqdm import tqdm

def apply_motion_blur(image, angle=45, length=15):
    """应用运动模糊效果"""
    # 创建运动模糊核
    kernel = np.zeros((length, length))
    kernel[length//2, :] = 1
    kernel = cv2.warpAffine(kernel, 
                           cv2.getRotationMatrix2D((length//2, length//2), angle, 1.0),
                           (length, length))
    kernel = kernel / kernel.sum()
    
    # 应用模糊
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred

def generate_blurred_images():
    """生成模糊图片"""
    sharp_dir = 'dataset/sharp'
    blur_dir = 'dataset/blur'
    
    # 确保输出目录存在
    os.makedirs(blur_dir, exist_ok=True)
    
    # 获取所有清晰图片
    image_files = [f for f in os.listdir(sharp_dir) if f.endswith(('.jpg', '.png'))]
    
    for img_name in tqdm(image_files, desc="生成模糊图片"):
        # 读取清晰图片
        img_path = os.path.join(sharp_dir, img_name)
        img = cv2.imread(img_path)
        
        # 随机生成模糊参数
        angle = np.random.uniform(0, 360)  # 随机角度
        length = np.random.randint(10, 20)  # 随机长度
        
        # 应用模糊
        blurred = apply_motion_blur(img, angle, length)
        
        # 保存模糊图片
        output_path = os.path.join(blur_dir, img_name)
        cv2.imwrite(output_path, blurred)

if __name__ == '__main__':
    generate_blurred_images() 