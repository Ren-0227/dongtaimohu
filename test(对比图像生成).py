import torch
from torchvision import transforms
import os
from torchvision.utils import save_image
from PIL import Image
import numpy as np

# 导入模型定义
from innovative_deblur import InnovativeDeblurGAN

# 指定训练集中的一对图像
blur_image_path = r'C:\Users\91144\Desktop\dongtaimohuhe\dataset\processed\blur\a_frame_0008.jpg'
clear_image_path = r'C:\Users\91144\Desktop\dongtaimohuhe\dataset\processed\sharp\a_frame_0008.jpg'

# 统一transform（和训练时一致）
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 和训练时一致
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 模型信息
model_info = {
    'name': '多模块协同创新去模糊',
    'model_class': InnovativeDeblurGAN,
    'weight_path': 'training_output_new/checkpoint_best.pth'  # 使用最佳检查点
}

# 加载模型
print("正在加载模型...")
model = model_info['model_class']().to(device)

# 加载模型权重
print("正在加载模型权重...")
checkpoint = torch.load(model_info['weight_path'], map_location=device)

# 加载权重
if 'generator_state_dict' in checkpoint:
    print("\n加载生成器权重...")
    model.generator.load_state_dict(checkpoint['generator_state_dict'])
    print("生成器参数数量:", sum(p.numel() for p in model.generator.parameters()))
    
    if 'discriminator_state_dict' in checkpoint:
        print("加载判别器权重...")
        model.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        print("判别器参数数量:", sum(p.numel() for p in model.discriminator.parameters()))
    
    if 'motion_estimator_state_dict' in checkpoint:
        print("加载运动估计器权重...")
        model.motion_estimator.load_state_dict(checkpoint['motion_estimator_state_dict'])
        print("运动估计器参数数量:", sum(p.numel() for p in model.motion_estimator.parameters()))
else:
    print("\n警告：检查点中没有找到生成器权重！")
    print("尝试直接加载模型状态...")
    try:
        model.load_state_dict(checkpoint)
        print("模型状态加载成功")
    except Exception as e:
        print(f"模型状态加载失败: {str(e)}")
        raise

model.eval()
print("\n模型加载完成")

# 创建输出目录
output_dir = "test_results_images"
os.makedirs(output_dir, exist_ok=True)

# 加载模糊图像和清晰图像
blur_image = Image.open(blur_image_path).convert('RGB')
clear_image = Image.open(clear_image_path).convert('RGB')
blur_tensor = transform(blur_image).unsqueeze(0).to(device)
clear_tensor = transform(clear_image).unsqueeze(0).to(device)

# 使用零张量作为前一帧
previous_frame = torch.zeros_like(blur_tensor)

# 获取运动估计
motion_map = model.motion_estimator(blur_tensor, previous_frame)

# 获取生成器输出
deblurred_tensor = model.generator(blur_tensor, motion_map, previous_frame)

# 反归一化，将图像从 [-1, 1] 转换回 [0, 1]
blur_tensor = (blur_tensor + 1) / 2
deblurred_tensor = (deblurred_tensor + 1) / 2
clear_tensor = (clear_tensor + 1) / 2

# 创建对比图像（左：模糊，中：去模糊，右：清晰）
comparison = torch.cat([blur_tensor, deblurred_tensor, clear_tensor], dim=0)

# 保存对比图像
save_image(comparison, f"{output_dir}/comparison_train_sample.png", nrow=3, normalize=False)

print(f"\n训练集样本推理完成，结果保存在 {output_dir}/comparison_train_sample.png")