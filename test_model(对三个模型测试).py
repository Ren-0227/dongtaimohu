import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import csv
import os

# 导入各自的模型和数据集定义
from train_innovative import DeblurDataset, calculate_psnr, calculate_ssim
from innovative_deblur import InnovativeDeblurGAN
from single_scale_deblur import SingleScaleDeblurGAN
from fixed_deblur import FixedDeblurGAN

# 测试集路径
blur_dir = r'C:\Users\91144\Desktop\dongtaimohuhe\test_blur_dataset\blur'
sharp_dir = r'C:\Users\91144\Desktop\dongtaimohuhe\test_blur_dataset\sharp'

# 统一transform（和训练时一致）
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 统一使用256x256尺寸
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 测试集
test_dataset = DeblurDataset(blur_dir, sharp_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 测试函数
def test_model(model, device, test_loader):
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    num_images = 0
    with torch.no_grad():
        for batch in test_loader:
            blur = batch['blur'].to(device)
            sharp = batch['clear'].to(device)
            deblurred, _ = model(blur)
            deblurred = (deblurred + 1) / 2
            sharp = (sharp + 1) / 2
            for i in range(deblurred.size(0)):
                psnr = calculate_psnr(sharp[i], deblurred[i])
                ssim_val = calculate_ssim(sharp[i], deblurred[i])
                total_psnr += psnr
                total_ssim += ssim_val
                num_images += 1
    avg_psnr = total_psnr / num_images
    avg_ssim = total_ssim / num_images
    return avg_psnr, avg_ssim

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型信息
models_info = [
    {
        'name': '多模块协同创新去模糊',
        'model_class': InnovativeDeblurGAN,
        'weight_path': 'training_output_new/innovative_deblur_final.pth'
    },
    {
        'name': '单尺度注意力去模糊',
        'model_class': SingleScaleDeblurGAN,
        'weight_path': 'training_output_single_scale/checkpoint_best.pth'
    },
    {
        'name': '固定卷积去模糊',
        'model_class': FixedDeblurGAN,
        'weight_path': 'training_output_fixed/fixed_deblur_final.pth'
    }
]

results = []

for info in models_info:
    print(f"正在测试模型: {info['name']}")
    model = info['model_class']().to(device)
    checkpoint = torch.load(info['weight_path'], map_location=device)
    # 自动适配权重加载
    if isinstance(checkpoint, dict):
        # 优先尝试加载generator/discriminator/motion_estimator
        if 'generator_state_dict' in checkpoint:
            model.generator.load_state_dict(checkpoint['generator_state_dict'])
            if hasattr(model, 'discriminator') and 'discriminator_state_dict' in checkpoint:
                model.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            if hasattr(model, 'motion_estimator') and 'motion_estimator_state_dict' in checkpoint:
                model.motion_estimator.load_state_dict(checkpoint['motion_estimator_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # 直接尝试整体加载
            try:
                model.load_state_dict(checkpoint)
            except Exception as e:
                print(f"权重加载失败: {e}")
                continue
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    avg_psnr, avg_ssim = test_model(model, device, test_loader)
    print(f"{info['name']} 平均PSNR: {avg_psnr:.4f}, 平均SSIM: {avg_ssim:.4f}")
    results.append({
        '模型名称': info['name'],
        '平均PSNR': avg_psnr,
        '平均SSIM': avg_ssim
    })

# 保存为CSV
csv_path = 'test_results.csv'
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['模型名称', '平均PSNR', '平均SSIM'])
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"所有结果已保存到 {csv_path}")