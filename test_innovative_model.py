import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from innovative_deblur import InnovativeDeblurGAN
from torchvision.utils import save_image

def test_model(model_path, blur_image_paths, output_dir):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载模型
    print("加载模型...")
    model = InnovativeDeblurGAN().to(device)
    
    # 加载检查点
    checkpoint = torch.load(model_path, map_location=device)
    
    # 加载生成器
    model.generator.load_state_dict(checkpoint['generator_state_dict'])
    
    # 加载判别器 - 处理键名不匹配的问题
    discriminator_state_dict = checkpoint['discriminator_state_dict']
    new_discriminator_state_dict = {}
    for k, v in discriminator_state_dict.items():
        if k.startswith(('0.', '2.', '3.', '5.', '6.', '8.', '9.', '11.')):
            new_k = 'main.' + k
            new_discriminator_state_dict[new_k] = v
    model.discriminator.load_state_dict(new_discriminator_state_dict)
    
    # 加载运动估计器
    if 'motion_estimator_state_dict' in checkpoint and checkpoint['motion_estimator_state_dict'] is not None:
        model.motion_estimator.load_state_dict(checkpoint['motion_estimator_state_dict'])
    
    model.eval()

    # 图像预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 处理每张图片
    for blur_path in blur_image_paths:
        try:
            print(f"\n处理图片: {blur_path}")
            
            # 加载并预处理图片
            blur_image = Image.open(blur_path).convert('RGB')
            blur_tensor = transform(blur_image).unsqueeze(0).to(device)

            # 进行去模糊
            with torch.no_grad():
                deblurred, _ = model(blur_tensor)

            # 准备保存的图像
            blur_vis = (blur_tensor[0] + 1) / 2  # 从[-1,1]转换到[0,1]
            deblurred_vis = (deblurred[0] + 1) / 2

            # 拼接图像
            comparison = torch.cat([blur_vis, deblurred_vis], dim=2)

            # 保存结果
            output_filename = os.path.join(output_dir, f"comparison_{os.path.basename(blur_path)}")
            save_image(comparison, output_filename)
            print(f"结果已保存到: {output_filename}")

        except Exception as e:
            print(f"处理图片 {blur_path} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    # 模型路径
    model_path = r"C:\Users\91144\Desktop\dongtaimohu\training_output\checkpoint_latest_epoch_55_batch_200.pth"
    
    # 测试图片路径
    blur_image_paths = [
        r"C:\Users\91144\Desktop\dongtaimohu\dataset\processed\blur\a_frame_3623.jpg",
        r"C:\Users\91144\Desktop\dongtaimohu\test_blur_dataset\blur\blur_0197.png",
        r"C:\Users\91144\Desktop\dongtaimohu\test_blur_dataset\blur\blur_1260.png"
    ]
    
    # 输出目录
    output_dir = "test_results"
    
    # 运行测试
    test_model(model_path, blur_image_paths, output_dir) 