import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from single_scale_deblur import SingleScaleDeblurGAN, SingleScaleDeblurLoss
from skimage.metrics import structural_similarity as ssim
from torchvision.utils import save_image

class DeblurDataset(Dataset):
    """图片去模糊数据集"""
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
            
        return {
            'blur': blur_image,
            'clear': clear_image
        }

def calculate_psnr(original, generated):
    """计算PSNR"""
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy()
    if isinstance(generated, torch.Tensor):
        generated = generated.detach().cpu().numpy()
    
    original = np.clip(original * 255, 0, 255).astype(np.uint8)
    generated = np.clip(generated * 255, 0, 255).astype(np.uint8)
    
    mse = np.mean((original - generated) ** 2)
    if mse == 0:
        return float('inf')
    
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_ssim(original, generated):
    """计算SSIM"""
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy()
    if isinstance(generated, torch.Tensor):
        generated = generated.detach().cpu().numpy()
    
    original = np.clip(original * 255, 0, 255).astype(np.uint8)
    generated = np.clip(generated * 255, 0, 255).astype(np.uint8)
    
    if len(original.shape) == 4:
        original = original[0]
        generated = generated[0]
    
    if original.shape[0] == 3:
        original = np.transpose(original, (1, 2, 0))
        generated = np.transpose(generated, (1, 2, 0))
    
    h, w = original.shape[:2]
    win_size = min(7, min(h, w) - 1)
    if win_size % 2 == 0:
        win_size -= 1
    if win_size < 3:
        win_size = 3
    
    return ssim(original, generated, 
                channel_axis=2,
                data_range=255,
                win_size=win_size)

def validate_model(model, dataloader, device):
    """验证模型"""
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            try:
                blur_images = batch['blur'].to(device)
                clear_images = batch['clear'].to(device)
                
                if torch.isnan(blur_images).any() or torch.isnan(clear_images).any():
                    print("Warning: NaN detected in validation input")
                    continue
                
                h, w = blur_images.size(2), blur_images.size(3)
                if h < 8 or w < 8:
                    print(f"Warning: Image size {h}x{w} is too small, skipping")
                    continue
                
                deblurred, _ = model(blur_images)
                
                if torch.isnan(deblurred).any():
                    print("Warning: NaN detected in generated images")
                    continue
                
                deblurred = (deblurred + 1) / 2
                clear_images = (clear_images + 1) / 2
                
                for i in range(deblurred.size(0)):
                    try:
                        psnr = calculate_psnr(clear_images[i], deblurred[i])
                        ssim_val = calculate_ssim(clear_images[i], deblurred[i])
                        
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

def save_checkpoint(epoch, model, g_optimizer, d_optimizer, g_loss, d_loss, metrics, output_dir):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': model.generator.state_dict(),
        'discriminator_state_dict': model.discriminator.state_dict(),
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

def load_checkpoint(checkpoint_path, model, g_optimizer, d_optimizer):
    """加载检查点"""
    if not os.path.exists(checkpoint_path):
        print(f"检查点文件不存在: {checkpoint_path}")
        return 0
    
    print(f"加载检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    
    # 加载各个组件的状态
    model.generator.load_state_dict(checkpoint['generator_state_dict'])
    model.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
    d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
    
    print(f"已加载第 {checkpoint['epoch']} 轮的检查点")
    if 'metrics' in checkpoint:
        print(f"检查点指标: {checkpoint['metrics']}")
    
    return checkpoint['epoch']

def train_single_scale_model():
    # 设置参数
    batch_size = 8
    num_epochs = 100
    learning_rate = 0.0002
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = "training_output_single_scale"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 加载数据集
    dataset = DeblurDataset(
        blur_dir=r'C:\Users\91144\Desktop\dongtaimohuhe\dataset\processed\blur',
        clear_dir=r'C:\Users\91144\Desktop\dongtaimohuhe\dataset\processed\sharp',
        transform=transform
    )
    
    # 划分训练集和验证集
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 初始化模型
    model = SingleScaleDeblurGAN().to(device)
    criterion = SingleScaleDeblurLoss()
    
    # 优化器
    g_optimizer = optim.Adam(model.generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(model.discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    
    # 尝试加载最新检查点
    start_epoch = load_checkpoint(f"{output_dir}/checkpoint_latest.pth", 
                                model, g_optimizer, d_optimizer)
    
    # 训练循环
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_g_loss = 0
        total_d_loss = 0
        
        for i, batch in enumerate(train_dataloader):
            # 获取数据
            blur_images = batch['blur'].to(device)
            clear_images = batch['clear'].to(device)
            
            # 训练判别器
            d_optimizer.zero_grad()
            
            # 真实图像
            real_output = model.discriminator(clear_images)
            d_real_loss = nn.BCELoss()(real_output, torch.ones_like(real_output))
            
            # 生成图像
            deblurred, _ = model(blur_images)
            fake_output = model.discriminator(deblurred.detach())
            d_fake_loss = nn.BCELoss()(fake_output, torch.zeros_like(fake_output))
            
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()
            
            # 训练生成器
            g_optimizer.zero_grad()
            
            # 生成图像
            deblurred, _ = model(blur_images)
            fake_output = model.discriminator(deblurred)
            
            # 计算损失
            g_loss = nn.BCELoss()(fake_output, torch.ones_like(fake_output))
            deblur_loss, loss_dict = criterion(deblurred, clear_images)
            
            total_g_loss = g_loss + deblur_loss
            total_g_loss.backward()
            g_optimizer.step()
            
            # 打印训练信息
            if i % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_dataloader)}], '
                      f'D_loss: {d_loss.item():.4f}, G_loss: {total_g_loss.item():.4f}, '
                      f'Recon_loss: {loss_dict["recon_loss"]:.4f}')
        
        # 计算平均损失
        avg_g_loss = total_g_loss / len(train_dataloader)
        avg_d_loss = total_d_loss / len(train_dataloader)
        
        # 每个epoch保存一些生成的图像
        if (epoch + 1) % 5 == 0:
            try:
                save_image(deblurred.data[:5], 
                          f"{output_dir}/generated_images_epoch_{epoch + 1}.png", 
                          nrow=5, normalize=True)
            except Exception as e:
                print(f"Error saving generated images: {str(e)}")
        
        # 每个epoch进行验证
        try:
            avg_psnr, avg_ssim = validate_model(model, val_dataloader, device)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Validation PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")
            
            # 保存检查点
            metrics = {
                'psnr': avg_psnr,
                'ssim': avg_ssim,
                'g_loss': avg_g_loss,
                'd_loss': avg_d_loss
            }
            save_checkpoint(epoch + 1, model, g_optimizer, d_optimizer,
                          avg_g_loss, avg_d_loss,
                          metrics, output_dir)
            
            # 将结果记录到文件
            with open(f"{output_dir}/training_log.txt", "a") as f:
                f.write(f"Epoch {epoch + 1}: PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}, "
                       f"G_Loss={avg_g_loss:.4f}, D_Loss={avg_d_loss:.4f}\n")
        except Exception as e:
            print(f"Error in validation: {str(e)}")
    
    # 保存最终模型
    torch.save(model.state_dict(), f"{output_dir}/single_scale_deblur_final.pth")

if __name__ == '__main__':
    train_single_scale_model() 