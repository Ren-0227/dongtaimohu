import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from innovative_deblur import (
    InnovativeDeblurGAN, 
    DeblurLoss,
    MotionAwareAttention,
    AdaptiveDeblurBlock,
    TemporalCoherenceModule
)
from skimage.metrics import structural_similarity as ssim
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import torchvision.models as models

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
    
    # 确保数值范围在[0,1]之间
    original = np.clip(original, 0, 1)
    generated = np.clip(generated, 0, 1)
    
    # 转换为[0,255]范围
    original = (original * 255).astype(np.uint8)
    generated = (generated * 255).astype(np.uint8)
    
    # 计算MSE
    mse = np.mean((original - generated) ** 2)
    if mse == 0:
        return float('inf')
    
    # 计算PSNR
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_ssim(original, generated):
    """计算SSIM"""
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy()
    if isinstance(generated, torch.Tensor):
        generated = generated.detach().cpu().numpy()
    
    # 确保数值范围在[0,1]之间
    original = np.clip(original, 0, 1)
    generated = np.clip(generated, 0, 1)
    
    # 转换为[0,255]范围
    original = (original * 255).astype(np.uint8)
    generated = (generated * 255).astype(np.uint8)
    
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
        'motion_estimator_state_dict': model.motion_estimator.state_dict(),
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
    
    # 打印保存信息
    print(f"\n保存检查点 - Epoch {epoch}")
    print(f"PSNR: {metrics.get('psnr', 0):.4f}")
    print(f"SSIM: {metrics.get('ssim', 0):.4f}")
    print(f"G_loss: {g_loss:.4f}")
    print(f"D_loss: {d_loss:.4f}")

def load_checkpoint(checkpoint_path, model, g_optimizer, d_optimizer):
    """加载检查点"""
    if not os.path.exists(checkpoint_path):
        print(f"检查点文件不存在: {checkpoint_path}")
        return 0
    
    print(f"加载检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    
    # 加载各个组件的状态，忽略不匹配的参数
    try:
        # 使用strict=False允许部分加载
        model.generator.load_state_dict(checkpoint['generator_state_dict'], strict=False)
        model.discriminator.load_state_dict(checkpoint['discriminator_state_dict'], strict=False)
        model.motion_estimator.load_state_dict(checkpoint['motion_estimator_state_dict'], strict=False)
        
        # 检查并打印加载状态
        print("模型参数加载状态:")
        print(f"- 生成器: {len(model.generator.state_dict())} 个参数")
        print(f"- 判别器: {len(model.discriminator.state_dict())} 个参数")
        print(f"- 运动估计器: {len(model.motion_estimator.state_dict())} 个参数")
        
        # 尝试加载优化器状态
        try:
            g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
            d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
            print("优化器状态加载成功")
        except Exception as e:
            print(f"优化器状态加载失败: {str(e)}")
            print("将使用新的优化器状态")
        
        print(f"已加载第 {checkpoint['epoch']} 轮的检查点")
        if 'metrics' in checkpoint:
            print(f"检查点指标: {checkpoint['metrics']}")
        
        return checkpoint['epoch']
        
    except Exception as e:
        print(f"加载检查点时出错: {str(e)}")
        print("将从头开始训练")
        return 0

def pre_training_check(model, dataloader, val_dataloader, device):
    """预训练检查"""
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
        model = model.to(device)
        
        # 检查生成器
        with torch.no_grad():
            # Pass a sample batch through the generator
            deblurred, motion_map = model(blur_images.to(device))
            
            print(f"生成器输出尺寸: {deblurred.size()}")
            print(f"运动图尺寸: {motion_map.size()}")
            print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
            
            # 检查判别器
            d_output = model.discriminator(clear_images.to(device))
            print(f"判别器输出尺寸: {d_output.size()}")
            
            # 检查输出值范围
            print(f"生成器输出范围: [{deblurred.min():.3f}, {deblurred.max():.3f}]")
            print(f"判别器输出范围: [{d_output.min():.3f}, {d_output.max():.3f}]")
        
    except Exception as e:
        print(f"模型检查失败: {str(e)}")
        return False

    # 4. 检查去模糊图像和运动图指标
    print("\n检查去模糊图像和运动图指标...")
    try:
        with torch.no_grad():
            # Calculate PSNR and SSIM for deblurred image
            # Ensure tensors are on the same device and in the correct range [0, 1]
            deblurred_norm = (deblurred.to(device) + 1) / 2
            clear_images_norm = (clear_images.to(device) + 1) / 2

            psnr = calculate_psnr(clear_images_norm[0], deblurred_norm[0])
            ssim_val = calculate_ssim(clear_images_norm[0], deblurred_norm[0])

            print(f"预训练去模糊图像 PSNR: {psnr:.2f}")
            print(f"预训练去模糊图像 SSIM: {ssim_val:.4f}")

            # Basic check for PSNR and SSIM (expect low but not zero/negative)
            if psnr < 5.0 or ssim_val < 0.05:
                print("警告: 预训练去模糊图像指标过低，可能存在问题。")
                # return False # Optionally return False to stop training

            # Check motion map values (expect values mostly around zero for untrained model)
            motion_map_abs_mean = torch.mean(torch.abs(motion_map.to(device)))
            motion_map_max = torch.max(torch.abs(motion_map.to(device)))

            print(f"预训练运动图平均绝对值: {motion_map_abs_mean:.4f}")
            print(f"预训练运动图最大绝对值: {motion_map_max:.4f}")

            # Basic check for motion map values (expect small values)
            if motion_map_abs_mean > 0.1 or motion_map_max > 0.5:
                print("警告: 预训练运动图数值异常，可能存在问题。")
                # return False # Optionally return False to stop training

    except Exception as e:
        print(f"去模糊图像和运动图指标检查失败: {str(e)}")
        return False
    
    # 5. 保存预训练可视化结果
    print("\n保存预训练可视化结果...")
    try:
        output_dir = "training_output_new"
        pre_training_viz_dir = os.path.join(output_dir, "pre_training_check_viz")
        os.makedirs(pre_training_viz_dir, exist_ok=True)
        
        # Use the save_visualizations function with epoch 0 (or any placeholder)
        save_visualizations(model, 0, pre_training_viz_dir, blur_images.to(device), clear_images.to(device))
        print(f"预训练可视化结果已保存到: {pre_training_viz_dir}")
        
    except Exception as e:
        print(f"保存预训练可视化结果失败: {str(e)}")
        return False
    
    # 5. 检查评估指标
    print("\n检查评估指标...")
    try:
        with torch.no_grad():
            # Assuming deblurred is already calculated from the model check
            psnr = calculate_psnr(clear_images[0].to(device), deblurred[0])
            print(f"PSNR测试值: {psnr:.2f}")
            
            ssim_val = calculate_ssim(clear_images[0].to(device), deblurred[0])
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
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            cached = torch.cuda.memory_reserved(0) / 1024**3
            print(f"GPU内存使用: {allocated:.2f} GB (已分配)")
            print(f"GPU内存缓存: {cached:.2f} GB (已保留)")
            
            if allocated > 0.9 * torch.cuda.get_device_properties(0).total_memory / 1024**3:
                print("警告: GPU内存使用率过高")
                return False
    except Exception as e:
        print(f"内存检查失败: {str(e)}")
        return False
    
    # 6. 检查数据加载器
    print("\n检查数据加载器...")
    try:
        # 检查训练数据加载器
        train_iter = iter(dataloader)
        train_batch = next(train_iter)
        print(f"训练数据批次大小: {train_batch['blur'].size(0)}")
        
        # 检查验证数据加载器
        val_iter = iter(val_dataloader)
        val_batch = next(val_iter)
        print(f"验证数据批次大小: {val_batch['blur'].size(0)}")
        
    except Exception as e:
        print(f"数据加载器检查失败: {str(e)}")
        return False
    
    # 7. 检查模型组件
    print("\n检查模型组件...")
    try:
        # 检查生成器组件
        print("检查生成器组件...")
        for name, module in model.generator.named_children():
            print(f"- {name}: {type(module).__name__}")
        
        # 检查判别器组件
        print("\n检查判别器组件...")
        for name, module in model.discriminator.named_children():
            print(f"- {name}: {type(module).__name__}")
        
        # 检查运动估计器组件
        print("\n检查运动估计器组件...")
        for name, module in model.motion_estimator.named_children():
            print(f"- {name}: {type(module).__name__}")
        
    except Exception as e:
        print(f"模型组件检查失败: {str(e)}")
        return False
    
    # 8. 检查优化器
    print("\n检查优化器...")
    try:
        # 创建临时优化器进行测试
        test_optimizer = optim.Adam(model.parameters(), lr=0.001)
        print("优化器创建成功")
        
    except Exception as e:
        print(f"优化器检查失败: {str(e)}")
        return False
    
    print("\n所有检查通过！可以开始训练。")
    return True

def save_visualizations(model, epoch, output_dir, blur_images, clear_images):
    """保存可视化结果"""
    model.eval()
    with torch.no_grad():
        # 获取去模糊结果
        deblurred, motion_map = model(blur_images)
        
        # 保存注意力图
        for i, block in enumerate(model.generator.motion_attention.modules()):
            if isinstance(block, MotionAwareAttention):
                if block.last_spatial_att is not None:
                    # 确保注意力图是3通道的
                    spatial_att = block.last_spatial_att[0].mean(dim=0, keepdim=True).repeat(3, 1, 1)
                    save_image(spatial_att, 
                             f"{output_dir}/spatial_attention_epoch_{epoch}.png")
                if block.last_motion_att is not None:
                    motion_att = block.last_motion_att[0].mean(dim=0, keepdim=True).repeat(3, 1, 1)
                    save_image(motion_att, 
                             f"{output_dir}/motion_attention_epoch_{epoch}.png")
                if block.last_fused_att is not None:
                    fused_att = block.last_fused_att[0].mean(dim=0, keepdim=True).repeat(3, 1, 1)
                    save_image(fused_att, 
                             f"{output_dir}/fused_attention_epoch_{epoch}.png")
        
        # 保存动态卷积核
        for i, block in enumerate(model.generator.deblur_blocks.modules()):
            if isinstance(block, AdaptiveDeblurBlock):
                if hasattr(block, 'last_kernel'):
                    # Get the kernel for the center pixel of the first image
                    b, c_out, kernel_size_sq, h, w = block.last_kernel.size()
                    center_h, center_w = h // 2, w // 2
                    kernel_center = block.last_kernel[0, 0, :, center_h, center_w]
                    kernel = kernel_center.view(3, 3) # Reshape to 3x3
                    
                    plt.figure(figsize=(5, 5))
                    plt.imshow(kernel.cpu().numpy(), cmap='viridis')
                    plt.colorbar()
                    plt.title(f'Dynamic Kernel Block {i} (Center)')
                    plt.savefig(f"{output_dir}/dynamic_kernel_block_{i}_epoch_{epoch}.png")
                    plt.close()
        
        # 保存时序注意力图
        for i, block in enumerate(model.generator.temporal_module.modules()):
            if isinstance(block, TemporalCoherenceModule):
                if block.last_temporal_att is not None:
                    temporal_att = block.last_temporal_att[0].mean(dim=0, keepdim=True).repeat(3, 1, 1)
                    save_image(temporal_att, 
                             f"{output_dir}/temporal_attention_epoch_{epoch}.png")
        
        # 保存对比图
        blur_images = (blur_images + 1) / 2
        clear_images = (clear_images + 1) / 2
        deblurred = (deblurred + 1) / 2
        
        for i in range(min(5, blur_images.size(0))):
            comparison = torch.cat([
                blur_images[i],
                deblurred[i],
                clear_images[i]
            ], dim=2)
            
            save_image(comparison, 
                      f"{output_dir}/comparison_epoch_{epoch}_sample_{i}.png")
        
        # 保存motion map
        for ch in range(motion_map.shape[1]):
            # 将motion map归一化到[0,1]范围
            mm = motion_map[0, ch].detach().cpu()
            mm = (mm - mm.min()) / (mm.max() - mm.min() + 1e-8)
            mm = mm.unsqueeze(0).repeat(3, 1, 1)  # 转换为3通道
            save_image(mm, f"{output_dir}/motion_map_epoch_{epoch}_ch{ch}.png")

def get_previous_frame(dataloader, current_batch_idx, batch_size):
    """获取前一帧"""
    if current_batch_idx > 0:
        # 获取前一帧
        prev_frame = dataloader.dataset[current_batch_idx - 1]['blur']
        
        # 确保维度正确
        if len(prev_frame.shape) == 3:  # [C, H, W]
            prev_frame = prev_frame.unsqueeze(0)  # [1, C, H, W]
        
        # 确保批次大小匹配
        if prev_frame.size(0) != batch_size:
            prev_frame = prev_frame.repeat(batch_size, 1, 1, 1)
            
        return prev_frame
    return None

# 添加去模糊损失函数类
class DeblurLoss(nn.Module):
    def __init__(self, recon_weight=1.0, motion_weight=1.0, temporal_weight=1.0, perceptual_weight=1.0):
        super(DeblurLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        # self.edge_loss = EdgeLoss()  # Uncomment if you implement EdgeLoss
        self.recon_weight = recon_weight
        self.motion_weight = motion_weight
        self.temporal_weight = temporal_weight
        self.perceptual_weight = perceptual_weight

        # Load VGG16 for perceptual loss
        self.vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:16].eval()
        for param in self.vgg.parameters():
            param.requires_grad = False # Freeze VGG parameters

    def forward(self, deblurred, target, motion_map, previous_frame=None):
        # 基础重建损失 (L1 loss)
        recon_loss = self.l1_loss(deblurred, target)

        # 运动损失 (L1 loss on motion map, penalizing non-zero motion)
        # We want motion_map to be close to 0 where there is no motion
        # So L1 loss towards zero motion
        motion_loss = torch.mean(torch.abs(motion_map)) # Penalize motion magnitude

        # 时序一致性损失 (Temporal L1 loss if previous frame is available)
        # Initialize as a zero tensor on the same device as deblurred
        temporal_loss = torch.tensor(0.0, device=deblurred.device)
        if previous_frame is not None:
            # Note: Re-evaluate if this is the correct temporal loss calculation
            try:
                if previous_frame.shape == deblurred.shape:
                     temporal_loss = self.l1_loss(deblurred, previous_frame.to(deblurred.device))
                else:
                     print("Warning: previous_frame shape mismatch with deblurred. Skipping temporal loss.")
                     temporal_loss = torch.tensor(0.0, device=deblurred.device) # Ensure it remains a tensor
            except Exception as e:
                print(f"Warning: Temporal loss calculation failed: {e}. Setting temporal_loss to 0.")
                temporal_loss = torch.tensor(0.0, device=deblurred.device) # Handle other errors

        # Perceptual Loss
        perceptual_loss = self.calculate_perceptual_loss(deblurred, target)

        # Combine losses with weights
        total_loss = self.recon_weight * recon_loss + \
                     self.motion_weight * motion_loss + \
                     self.temporal_weight * temporal_loss + \
                     self.perceptual_weight * perceptual_loss

        return total_loss, recon_loss, motion_loss, temporal_loss, perceptual_loss # Return individual losses for logging

    # Move the perceptual loss calculation inside the class
    def calculate_perceptual_loss(self, generated, target):
        # Ensure inputs are within [0,1] range and move to VGG device
        generated = (generated + 1) / 2
        target = (target + 1) / 2

        device = next(self.vgg.parameters()).device
        generated = generated.to(device)
        target = target.to(device)

        # Cast to float32 for VGG compatibility in mixed precision
        generated = generated.float()
        target = target.float()

        # Normalize input
        # Use the same normalization as in the original calculate_perceptual_loss function
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        generated = normalize(generated)
        target = normalize(target)

        # Extract features
        gen_features = self.vgg(generated)
        target_features = self.vgg(target)

        # Calculate L1 loss between features
        loss = self.l1_loss(gen_features, target_features)

        return loss

def validate_and_debug(model, val_dataloader, device, epoch, output_dir):
    """验证和调试函数"""
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    total_perceptual = 0.0
    num_samples = 0
    
    # Initialize VGG for perceptual loss in validation
    vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:16].eval()
    vgg = vgg.to(device)
    for param in vgg.parameters():
        param.requires_grad = False # Freeze VGG parameters
    l1_loss_val = nn.L1Loss()

    # Data normalization for perceptual loss
    normalize_perceptual = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])

    # 创建调试输出目录
    debug_dir = os.path.join(output_dir, f"debug_epoch_{epoch}")
    os.makedirs(debug_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            try:
                blur_images = batch['blur'].to(device)
                clear_images = batch['clear'].to(device)
                
                # 生成去模糊图像
                deblurred, motion_map = model(blur_images)
                
                # 计算指标
                deblurred_norm = (deblurred + 1) / 2
                clear_norm = (clear_images + 1) / 2
                
                # 计算PSNR and SSIM
                psnr = calculate_psnr(clear_norm, deblurred_norm)
                ssim_val = calculate_ssim(clear_norm, deblurred_norm)
                
                # Calculate perceptual loss for validation
                generated_perceptual = (deblurred + 1) / 2
                target_perceptual = (clear_images + 1) / 2

                generated_perceptual = normalize_perceptual(generated_perceptual)
                target_perceptual = normalize_perceptual(target_perceptual)

                gen_features = vgg(generated_perceptual)
                target_features = vgg(target_perceptual)
                perceptual = l1_loss_val(gen_features, target_features)
                
                if np.isfinite(psnr) and 0 <= ssim_val <= 1:
                    total_psnr += psnr
                    total_ssim += ssim_val
                    total_perceptual += perceptual.item()
                    num_samples += 1
                
                # 保存调试图像
                if i < 5:  # 只保存前5个样本
                    # 保存对比图
                    comparison = torch.cat([
                        (blur_images[0] + 1) / 2,
                        deblurred_norm[0],
                        clear_norm[0]
                    ], dim=2)
                    save_image(comparison, 
                             os.path.join(debug_dir, f"comparison_sample_{i}.png"))
                    
                    # 保存运动图
                    for ch in range(motion_map.shape[1]):
                        mm = motion_map[0, ch].detach().cpu()
                        mm = (mm - mm.min()) / (mm.max() - mm.min() + 1e-8)
                        mm = mm.unsqueeze(0).repeat(3, 1, 1)
                        save_image(mm, 
                                 os.path.join(debug_dir, f"motion_map_ch{ch}_sample_{i}.png"))
                
            except Exception as e:
                print(f"验证样本 {i} 时出错: {str(e)}")
                continue
    
    if num_samples == 0:
        print("警告：没有有效的验证样本")
        return 0.0, 0.0, 0.0
    
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples
    avg_perceptual = total_perceptual / num_samples
    
    # 打印详细的验证结果
    print(f"\n验证结果 (Epoch {epoch}):")
    print(f"PSNR: {avg_psnr:.4f}")
    print(f"SSIM: {avg_ssim:.4f}")
    print(f"Perceptual Loss: {avg_perceptual:.4f}")
    
    # 保存验证结果到文件
    with open(os.path.join(output_dir, "validation_log.txt"), "a") as f:
        f.write(f"Epoch {epoch}: PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}, "
                f"Perceptual={avg_perceptual:.4f}\n")
    
    return avg_psnr, avg_ssim, avg_perceptual

def train_innovative_model():
    # 设置参数
    batch_size = 8  # 减小batch_size
    num_epochs = 100
    learning_rate = 0.0002
    
    # 添加验证频率参数
    validation_frequency = 5 # 每5轮验证一次
    
    # GPU设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        # 设置GPU内存分配器
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True
        
        # 设置GPU内存分配策略 - 使用70%的显存
        torch.cuda.set_per_process_memory_fraction(0.7)
        torch.cuda.memory.set_per_process_memory_fraction(0.7)
        
        print(f"\nGPU信息:")
        print(f"设备名称: {torch.cuda.get_device_name(0)}")
        print(f"总内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"当前内存使用: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"当前内存缓存: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    else:
        print("警告: 未检测到GPU，将使用CPU训练（训练速度会很慢）")
    
    output_dir = "training_output_new"
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_path = os.path.join(output_dir, "checkpoint_latest.pth")
    
    # 数据预处理 - 减小图像尺寸
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # 减小图像尺寸
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 加载数据集
    dataset = DeblurDataset(
        blur_dir=r'C:\Users\91144\Desktop\dongtaimohu\dataset\processed\blur',
        clear_dir=r'C:\Users\91144\Desktop\dongtaimohu\dataset\processed\sharp',
        transform=transform
    )
    
    # 划分训练集和验证集
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # 优化数据加载器
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,  # 减少worker数量
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,  # 减少预加载因子
        drop_last=True
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True
    )
    
    # 初始化模型
    model = InnovativeDeblurGAN().to(device)
    # 初始化DeblurLoss并设置损失权重
    criterion = DeblurLoss(recon_weight=1.0, motion_weight=20.0, temporal_weight=0.01, perceptual_weight=5.0)
    
    # 使用混合精度训练
    scaler = torch.cuda.amp.GradScaler()
    
    # 优化器
    g_params = list(model.generator.parameters()) + list(model.motion_estimator.parameters())
    g_optimizer = optim.Adam(g_params, lr=learning_rate, betas=(0.5, 0.999))  # 使用原始学习率
    d_optimizer = optim.Adam(model.discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    
    # 使用更保守的学习率调度器
    g_scheduler = optim.lr_scheduler.OneCycleLR(
        g_optimizer,
        max_lr=0.001, # Increased max_lr
        epochs=num_epochs,
        steps_per_epoch=len(train_dataloader),
        pct_start=0.3,
        div_factor=5.0,  # Less conservative initial learning rate
        final_div_factor=10.0  # Less conservative final learning rate
    )
    d_scheduler = optim.lr_scheduler.OneCycleLR(
        d_optimizer,
        max_lr=0.001, # Increased max_lr
        epochs=num_epochs,
        steps_per_epoch=len(train_dataloader),
        pct_start=0.3,
        div_factor=5.0, # Less conservative initial learning rate
        final_div_factor=10.0 # Less conservative final learning rate
    )
    
    # Load checkpoint if exists
    start_epoch = load_checkpoint(checkpoint_path, model, g_optimizer, d_optimizer)
    
    # 添加验证参数
    min_psnr_threshold = 20.0
    min_ssim_threshold = 0.6
    max_perceptual_threshold = 0.5
    patience = 5
    best_metrics = {
        'psnr': 0,
        'ssim': 0,
        'perceptual': float('inf')
    }
    no_improve_count = 0
    
    # Run pre-training checks
    if pre_training_check(model, train_dataloader, val_dataloader, device):
        # 训练循环
        for epoch in range(start_epoch, num_epochs):
            model.train()
            total_g_loss = 0
            total_d_loss = 0
            
            for i, batch in enumerate(train_dataloader):
                # 获取数据并转移到GPU
                blur_images = batch['blur'].to(device, non_blocking=True)
                clear_images = batch['clear'].to(device, non_blocking=True)
                
                # 获取前一帧
                previous_frame = get_previous_frame(train_dataloader, i, batch_size)
                if previous_frame is not None:
                    previous_frame = previous_frame.to(device, non_blocking=True)
                
                # 训练判别器
                d_optimizer.zero_grad(set_to_none=True)
                
                with torch.cuda.amp.autocast():
                    # 真实图像
                    real_output = model.discriminator(clear_images)
                    d_real_loss = nn.BCEWithLogitsLoss()(real_output, torch.ones_like(real_output) * 0.9)
                    
                    # 生成图像
                    deblurred, motion_map = model(blur_images, previous_frame)
                    fake_output = model.discriminator(deblurred.detach())
                    d_fake_loss = nn.BCEWithLogitsLoss()(fake_output, torch.zeros_like(fake_output) * 0.1)
                    
                    d_loss = (d_real_loss + d_fake_loss) * 0.5
                
                scaler.scale(d_loss).backward()
                scaler.step(d_optimizer)
                
                # 训练生成器
                g_optimizer.zero_grad(set_to_none=True)
                
                with torch.cuda.amp.autocast():
                    # 生成图像
                    deblurred, motion_map = model(blur_images, previous_frame)
                    fake_output = model.discriminator(deblurred)
                    
                    # 计算损失
                    # GAN loss for generator
                    g_loss = nn.BCEWithLogitsLoss()(fake_output, torch.ones_like(fake_output) * 0.9)
                    
                    # Deblurring loss (Reconstruction, Motion, Temporal, Perceptual)
                    total_deblur_loss, recon_loss, motion_loss, temporal_loss, perceptual_loss = criterion(deblurred, clear_images, motion_map, previous_frame)
                    
                    # Combine GAN loss and deblurring loss
                    generator_total_loss = g_loss + total_deblur_loss
                
                scaler.scale(generator_total_loss).backward()
                
                # 梯度裁剪
                scaler.unscale_(g_optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(g_optimizer)
                scaler.update()
                
                # 更新学习率
                g_scheduler.step()
                d_scheduler.step()
                
                # 更新总损失统计
                total_g_loss += generator_total_loss.item()
                total_d_loss += d_loss.item()
                
                # 打印训练信息
                if i % 10 == 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_dataloader)}], '
                          f'D_loss: {d_loss.item():.4f}, G_loss: {generator_total_loss.item():.4f}, '
                          f'Recon_loss: {recon_loss.float().item():.4f}, '
                          f'Motion_loss: {motion_loss.float().item():.4f}, '
                          f'Temporal_loss: {temporal_loss.float().item():.4f}, '
                          f'Perceptual_loss: {perceptual_loss.float().item():.4f}, '
                          f'LR: {g_scheduler.get_last_lr()[0]:.6f}')
                
                # 定期清理GPU缓存
                if i % 50 == 0:
                    torch.cuda.empty_cache()
            
            # 每个epoch结束后清理GPU缓存
            torch.cuda.empty_cache()
            
            # Validation and checkpoint saving
            # 只在指定的验证频率或最后一轮进行验证
            if (epoch + 1) % validation_frequency == 0 or (epoch + 1) == num_epochs:
                print(f"\n--- 验证 Epoch {epoch+1} ---")
                avg_psnr, avg_ssim, avg_perceptual = validate_and_debug(model, val_dataloader, device, epoch+1, output_dir)
                
                # Update best metrics and check for early stopping
                current_metrics = {
                    'psnr': avg_psnr,
                    'ssim': avg_ssim,
                    'perceptual': avg_perceptual
                }
                
                # Check for improvement based on PSNR, SSIM, and Perceptual Loss
                # Prioritize SSIM and Perceptual Loss as they are currently low
                if avg_ssim > best_metrics['ssim'] or avg_perceptual < best_metrics['perceptual']:
                     best_metrics = current_metrics
                     no_improve_count = 0
                     print("模型性能改善，保存最佳检查点...")
                     save_checkpoint(epoch+1, model, g_optimizer, d_optimizer, total_g_loss / len(train_dataloader), total_d_loss / len(train_dataloader), best_metrics, output_dir)
                else:
                    no_improve_count += 1
                    print(f"模型性能未改善 ({no_improve_count}/{patience})")

                # Early stopping check
                if no_improve_count >= patience:
                    print(f"性能连续 {patience} 轮未改善，停止训练。")
                    break

            # Always save the latest checkpoint after each epoch - Moved inside validation block
            # This might need adjustment if you want to save checkpoint every epoch regardless of validation
            # For now, checkpoint saving is tied to validation.
            # If you need to save checkpoint every epoch, move save_checkpoint outside this if block.
            if (epoch + 1) % validation_frequency == 0 or (epoch + 1) == num_epochs:
                 # Save the latest checkpoint
                 save_checkpoint(epoch+1, model, g_optimizer, d_optimizer, total_g_loss / len(train_dataloader), total_d_loss / len(train_dataloader), current_metrics if 'current_metrics' in locals() else None, output_dir)

    print("训练完成！")

if __name__ == '__main__':
    train_innovative_model() 