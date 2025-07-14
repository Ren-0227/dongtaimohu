import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import numpy as np
import os
import cv2
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
from math import sqrt

# 数据集准备
class DeblurringDataset(Dataset):
    def __init__(self, blur_dir, clear_dir, transform=None):
        self.blur_dir = blur_dir
        self.clear_dir = clear_dir
        self.transform = transform
        
        self.blur_files = sorted(os.listdir(blur_dir))
        self.clear_files = sorted(os.listdir(clear_dir))
        
    def __len__(self):
        return len(self.blur_files)
    
    def __getitem__(self, idx):
        blur_path = os.path.join(self.blur_dir, self.blur_files[idx])
        clear_path = os.path.join(self.clear_dir, self.clear_files[idx])
        
        blur_image = Image.open(blur_path).convert('RGB')
        clear_image = Image.open(clear_path).convert('RGB')
        
        if self.transform:
            blur_image = self.transform(blur_image)
            clear_image = self.transform(clear_image)
            
        return {'blur': blur_image, 'clear': clear_image}

# 生成器网络
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.encoder = nn.Sequential(
            # 输入: 3x256x256
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64x128x128
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 128x64x64
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 256x32x32
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.residual_blocks = nn.Sequential(
            *[self._create_residual_block(512) for _ in range(4)]
        )
        
        self.decoder = nn.Sequential(
            # 512x32x32
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 256x64x64
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 128x128x128
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 64x256x256
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        
    def _create_residual_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.residual_blocks(x)
        x = self.decoder(x)
        return x

# 判别器网络
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.main = nn.Sequential(
            # 输入: 3x256x256
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64x128x128
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 128x64x64
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 256x32x32
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 512x16x16
            nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.main(x)

# 计算PSNR
def calculate_psnr(original, generated):
    # 确保输入是numpy数组
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy()
    if isinstance(generated, torch.Tensor):
        generated = generated.detach().cpu().numpy()
    
    # 确保图像在[0, 255]范围内
    original = np.clip(original * 255, 0, 255).astype(np.uint8)
    generated = np.clip(generated * 255, 0, 255).astype(np.uint8)
    
    # 计算MSE
    mse = np.mean((original - generated) ** 2)
    if mse == 0:
        return float('inf')
    
    # 计算PSNR
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# 计算SSIM
def calculate_ssim(original, generated):
    # 确保输入是numpy数组
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy()
    if isinstance(generated, torch.Tensor):
        generated = generated.detach().cpu().numpy()
    
    # 确保图像在[0, 255]范围内
    original = np.clip(original * 255, 0, 255).astype(np.uint8)
    generated = np.clip(generated * 255, 0, 255).astype(np.uint8)
    
    # 检查图像尺寸
    if len(original.shape) == 4:  # 如果是批次数据，取第一个样本
        original = original[0]
        generated = generated[0]
    
    # 确保图像是HWC格式
    if original.shape[0] == 3:  # 如果是CHW格式，转换为HWC
        original = np.transpose(original, (1, 2, 0))
        generated = np.transpose(generated, (1, 2, 0))
    
    h, w = original.shape[:2]  # 获取高度和宽度
    
    # 确保图像尺寸足够大
    if h < 8 or w < 8:
        raise ValueError(f"图像尺寸太小 ({h}x{w})，至少需要8x8")
    
    # 计算合适的窗口大小
    win_size = min(7, min(h, w) - 1)  # 确保窗口大小小于图像尺寸
    if win_size % 2 == 0:  # 确保窗口大小是奇数
        win_size -= 1
    if win_size < 3:  # 如果图像太小，使用最小窗口大小
        win_size = 3
    
    # 计算SSIM
    try:
        return ssim(original, generated, 
                   channel_axis=2,  # 指定通道轴
                   data_range=255,
                   win_size=win_size)
    except Exception as e:
        print(f"SSIM计算错误: {str(e)}")
        print(f"图像尺寸: {original.shape}")
        print(f"窗口大小: {win_size}")
        raise e

# 验证模型
def validate_model(generator, dataloader, device):
    generator.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            try:
                blur_images = batch['blur'].to(device)
                clear_images = batch['clear'].to(device)
                
                # 检查输入数据
                if torch.isnan(blur_images).any() or torch.isnan(clear_images).any():
                    print("Warning: NaN detected in validation input")
                    continue
                
                # 检查图像尺寸
                h, w = blur_images.size(2), blur_images.size(3)
                if h < 8 or w < 8:  # 确保图像尺寸足够大
                    print(f"Warning: Image size {h}x{w} is too small, skipping")
                    continue
                
                generated_images = generator(blur_images)
                
                # 检查生成图像
                if torch.isnan(generated_images).any():
                    print("Warning: NaN detected in generated images")
                    continue
                
                # 将图像转换到[0,1]范围
                generated_images = (generated_images + 1) / 2
                clear_images = (clear_images + 1) / 2
                
                for i in range(generated_images.size(0)):
                    try:
                        # 计算PSNR和SSIM
                        psnr = calculate_psnr(clear_images[i], generated_images[i])
                        ssim_val = calculate_ssim(clear_images[i], generated_images[i])
                        
                        if np.isfinite(psnr) and 0 <= ssim_val <= 1:
                            total_psnr += psnr
                            total_ssim += ssim_val
                            num_samples += 1
                        else:
                            print(f"Warning: Invalid metrics - PSNR: {psnr}, SSIM: {ssim_val}")
                    except Exception as e:
                        print(f"Error processing sample {i}: {str(e)}")
                        continue
                
            except Exception as e:
                print(f"Error in validation batch: {str(e)}")
                continue
    
    if num_samples == 0:
        print("Warning: No valid samples were evaluated")
        return 0.0, 0.0
    
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples
    
    return avg_psnr, avg_ssim

def save_checkpoint(epoch, generator, discriminator, g_optimizer, d_optimizer, g_loss, d_loss, metrics, output_dir):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'g_optimizer_state_dict': g_optimizer.state_dict(),
        'd_optimizer_state_dict': d_optimizer.state_dict(),
        'g_loss': g_loss,
        'd_loss': d_loss,
        'metrics': metrics
    }
    
    # 保存最新检查点
    torch.save(checkpoint, f"{output_dir}/checkpoint_latest.pth")
    
    # 如果是最佳模型，也保存一份
    if metrics and metrics.get('psnr', 0) > 0:
        torch.save(checkpoint, f"{output_dir}/checkpoint_best.pth")
    
    # 每10个epoch保存一次
    if epoch % 10 == 0:
        torch.save(checkpoint, f"{output_dir}/checkpoint_epoch_{epoch}.pth")

def load_checkpoint(checkpoint_path, generator, discriminator, g_optimizer, d_optimizer):
    """加载检查点"""
    if not os.path.exists(checkpoint_path):
        print(f"检查点文件不存在: {checkpoint_path}")
        return 0
    
    print(f"加载检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
    d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
    
    print(f"已加载第 {checkpoint['epoch']} 轮的检查点")
    if 'metrics' in checkpoint:
        print(f"检查点指标: {checkpoint['metrics']}")
    
    return checkpoint['epoch']

# 训练函数
def train_gan(generator, discriminator, dataloader, val_dataloader, num_epochs, device, output_dir):
    generator.to(device)
    discriminator.to(device)
    
    adversarial_loss = nn.BCELoss()
    content_loss = nn.MSELoss()
    
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 尝试加载最新检查点
    start_epoch = load_checkpoint(f"{output_dir}/checkpoint_latest.pth", 
                                generator, discriminator, 
                                optimizer_G, optimizer_D)
    
    # 训练并定期验证
    for epoch in range(start_epoch, num_epochs):
        generator.train()
        discriminator.train()
        
        total_g_loss = 0
        total_d_loss = 0
        
        for i, batch in enumerate(dataloader):
            try:
                blur_images = batch['blur'].to(device)
                clear_images = batch['clear'].to(device)
                
                # 检查输入数据
                if torch.isnan(blur_images).any() or torch.isnan(clear_images).any():
                    print(f"Warning: NaN detected in training input at batch {i}")
                    continue
                
                # 真实和假标签
                real_labels = torch.ones((blur_images.size(0), 1, 8, 8), device=device)
                fake_labels = torch.zeros((blur_images.size(0), 1, 8, 8), device=device)
                
                # 训练判别器
                optimizer_D.zero_grad()
                
                # 真实图像
                real_outputs = discriminator(clear_images)
                d_real_loss = adversarial_loss(real_outputs, real_labels)
                
                # 生成图像
                generated_images = generator(blur_images)
                fake_outputs = discriminator(generated_images.detach())
                d_fake_loss = adversarial_loss(fake_outputs, fake_labels)
                
                d_loss = d_real_loss + d_fake_loss
                
                # 检查损失值
                if torch.isnan(d_loss):
                    print(f"Warning: NaN detected in discriminator loss at batch {i}")
                    continue
                
                d_loss.backward()
                optimizer_D.step()
                
                # 训练生成器
                optimizer_G.zero_grad()
                
                g_adversarial_loss = adversarial_loss(discriminator(generated_images), real_labels)
                g_content_loss = content_loss(generated_images, clear_images)
                g_loss = g_adversarial_loss + 0.01 * g_content_loss
                
                # 检查损失值
                if torch.isnan(g_loss):
                    print(f"Warning: NaN detected in generator loss at batch {i}")
                    continue
                
                g_loss.backward()
                optimizer_G.step()
                
                total_g_loss += g_loss.item()
                total_d_loss += d_loss.item()
                
                # 每10个批次打印一次
                if (i + 1) % 10 == 0:
                    print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(dataloader)}] "
                          f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
                
            except Exception as e:
                print(f"Error in training batch {i}: {str(e)}")
                continue
        
        # 计算平均损失
        avg_g_loss = total_g_loss / len(dataloader)
        avg_d_loss = total_d_loss / len(dataloader)
        
        # 每个epoch保存一些生成的图像
        if (epoch + 1) % 5 == 0:
            try:
                save_image(generated_images.data[:5], 
                          f"{output_dir}/generated_images_epoch_{epoch + 1}.png", 
                          nrow=5, normalize=True)
            except Exception as e:
                print(f"Error saving generated images: {str(e)}")
        
        # 每个epoch进行验证
        try:
            avg_psnr, avg_ssim = validate_model(generator, val_dataloader, device)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Validation PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")
            
            # 保存检查点
            metrics = {
                'psnr': avg_psnr,
                'ssim': avg_ssim,
                'g_loss': avg_g_loss,
                'd_loss': avg_d_loss
            }
            save_checkpoint(epoch + 1, generator, discriminator, 
                          optimizer_G, optimizer_D,
                          avg_g_loss, avg_d_loss,
                          metrics, output_dir)
            
            # 将结果记录到文件
            with open(f"{output_dir}/training_log.txt", "a") as f:
                f.write(f"Epoch {epoch + 1}: PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}, "
                       f"G_Loss={avg_g_loss:.4f}, D_Loss={avg_d_loss:.4f}\n")
        except Exception as e:
            print(f"Error in validation: {str(e)}")
    
    return generator, discriminator

def pre_training_check(generator, discriminator, dataloader, val_dataloader, device):
    """预训练检查函数"""
    print("开始预训练检查...")
    
    # 1. 检查GPU
    if torch.cuda.is_available():
        print(f"GPU可用: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("警告: 未检测到GPU，将使用CPU训练（训练速度会很慢）")
    
    # 2. 检查数据集
    print("\n检查数据集...")
    try:
        sample_batch = next(iter(dataloader))
        blur_images = sample_batch['blur']
        clear_images = sample_batch['clear']
        
        print(f"批次大小: {blur_images.size(0)}")
        print(f"图像尺寸: {blur_images.size(2)}x{blur_images.size(3)}")
        print(f"通道数: {blur_images.size(1)}")
        
        # 检查图像值范围
        print(f"模糊图像值范围: [{blur_images.min():.3f}, {blur_images.max():.3f}]")
        print(f"清晰图像值范围: [{clear_images.min():.3f}, {clear_images.max():.3f}]")
        
        if blur_images.size(2) < 8 or blur_images.size(3) < 8:
            raise ValueError("图像尺寸太小，至少需要8x8")
        
    except Exception as e:
        print(f"数据集检查失败: {str(e)}")
        return False
    
    # 3. 检查模型
    print("\n检查模型...")
    try:
        # 将模型移动到正确的设备
        generator = generator.to(device)
        discriminator = discriminator.to(device)
        
        # 检查生成器
        g_output = generator(blur_images.to(device))
        print(f"生成器输出尺寸: {g_output.size()}")
        print(f"生成器参数数量: {sum(p.numel() for p in generator.parameters())}")
        
        # 检查判别器
        d_output = discriminator(clear_images.to(device))
        print(f"判别器输出尺寸: {d_output.size()}")
        print(f"判别器参数数量: {sum(p.numel() for p in discriminator.parameters())}")
        
        # 检查输出值范围
        print(f"生成器输出范围: [{g_output.min():.3f}, {g_output.max():.3f}]")
        print(f"判别器输出范围: [{d_output.min():.3f}, {d_output.max():.3f}]")
        
    except Exception as e:
        print(f"模型检查失败: {str(e)}")
        return False
    
    # 4. 检查评估指标
    print("\n检查评估指标...")
    try:
        # 测试PSNR计算
        psnr = calculate_psnr(clear_images[0], g_output[0])
        print(f"PSNR测试值: {psnr:.2f}")
        
        # 测试SSIM计算
        ssim_val = calculate_ssim(clear_images[0], g_output[0])
        print(f"SSIM测试值: {ssim_val:.4f}")
        
        if not (0 <= ssim_val <= 1):
            raise ValueError(f"SSIM值超出范围: {ssim_val}")
        
    except Exception as e:
        print(f"评估指标检查失败: {str(e)}")
        return False
    
    # 5. 检查内存使用
    print("\n检查内存使用...")
    try:
        if torch.cuda.is_available():
            # 检查GPU内存
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            cached = torch.cuda.memory_reserved(0) / 1024**3
            print(f"GPU内存使用: {allocated:.2f} GB (已分配)")
            print(f"GPU内存缓存: {cached:.2f} GB (已保留)")
            
            # 确保有足够的内存
            if allocated > 0.9 * torch.cuda.get_device_properties(0).total_memory / 1024**3:
                print("警告: GPU内存使用率过高")
                return False
    except Exception as e:
        print(f"内存检查失败: {str(e)}")
        return False
    
    print("\n所有检查通过！可以开始训练。")
    return True

# 主函数
def main():
    # 设置超参数
    batch_size = 16
    num_epochs = 100
    image_size = 256
    output_dir = "training_output"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 数据预处理和加载
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 训练集和验证集
    train_dataset = DeblurringDataset(
        blur_dir=r'C:\Users\91144\Desktop\dongtaimohuhe\dataset\processed\blur',
        clear_dir=r'C:\Users\91144\Desktop\dongtaimohuhe\dataset\processed\sharp',
        transform=transform
    )
    
    # 假设将10%的数据用于验证
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 初始化网络
    generator = Generator()
    discriminator = Discriminator()
    
    # 预训练检查
    if not pre_training_check(generator, discriminator, train_dataloader, val_dataloader, device):
        print("预训练检查失败，请解决上述问题后再开始训练。")
        return
    
    # 训练
    trained_generator, trained_discriminator = train_gan(
        generator, discriminator, train_dataloader, val_dataloader, 
        num_epochs, device, output_dir
    )
    
    # 保存最终模型
    torch.save(trained_generator.state_dict(), f"{output_dir}/deblurring_generator_final.pth")
    torch.save(trained_discriminator.state_dict(), f"{output_dir}/deblurring_discriminator_final.pth")

if __name__ == "__main__":
    main()