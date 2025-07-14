# File: train_innovative_yuanshi.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from innovative_deblur import InnovativeDeblurGAN, DeblurLoss, PerceptualLoss # 确保导入 PerceptualLoss
from skimage.metrics import structural_similarity as ssim
from torchvision.utils import save_image
import traceback # 导入 traceback 模块
import torch.optim.lr_scheduler as lr_scheduler # 导入学习率调度器模块

class DeblurDataset(Dataset):
    """图片去模糊数据集"""
    def __init__(self, blur_dir, clear_dir, transform=None):
        self.blur_dir = blur_dir
        self.clear_dir = clear_dir
        self.transform = transform
        
        # 验证目录是否存在
        if not os.path.exists(blur_dir):
            raise ValueError(f"模糊图像目录不存在: {blur_dir}")
        if not os.path.exists(clear_dir):
            raise ValueError(f"清晰图像目录不存在: {clear_dir}")
        
        # 获取并验证文件列表
        self.blur_files = sorted([f for f in os.listdir(blur_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.clear_files = sorted([f for f in os.listdir(clear_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        if len(self.blur_files) == 0:
            raise ValueError(f"模糊图像目录为空: {blur_dir}")
        if len(self.clear_files) == 0:
            raise ValueError(f"清晰图像目录为空: {clear_dir}")
        
        # 验证文件数量是否匹配
        if len(self.blur_files) != len(self.clear_files):
            raise ValueError(f"模糊图像数量 ({len(self.blur_files)}) 与清晰图像数量 ({len(self.clear_files)}) 不匹配")
        
        # 验证文件名是否匹配
        for bf, cf in zip(self.blur_files, self.clear_files):
            if os.path.splitext(bf)[0] != os.path.splitext(cf)[0]:
                print(f"警告: 文件名不匹配 - 模糊: {bf}, 清晰: {cf}")
        
    def __len__(self):
        return len(self.blur_files)
    
    def __getitem__(self, idx):
        try:
            blur_path = os.path.join(self.blur_dir, self.blur_files[idx])
            clear_path = os.path.join(self.clear_dir, self.clear_files[idx])
            
            # 加载图像
            blur_image = Image.open(blur_path).convert('RGB')
            clear_image = Image.open(clear_path).convert('RGB')
            
            # 验证图像尺寸
            if blur_image.size != clear_image.size:
                raise ValueError(f"图像尺寸不匹配 - 模糊: {blur_image.size}, 清晰: {clear_image.size} - {blur_path} vs {clear_path}")
            
            # 应用变换
            if self.transform:
                blur_image = self.transform(blur_image)
                clear_image = self.transform(clear_image)
            
            # 验证变换后的图像
            if torch.isnan(blur_image).any() or torch.isnan(clear_image).any():
                raise ValueError("变换后的图像包含NaN值")
            
            return {
                'blur': blur_image,
                'clear': clear_image,
                'blur_path': blur_path,  # 保存路径用于调试
                'clear_path': clear_path
            }
            
        except Exception as e:
            print(f"加载图像时出错 (idx={idx}, file={self.blur_files[idx]}): {str(e)}")
            # 返回一个有效的替代样本（如果可能），或者抛出异常
            # 简单的处理是返回None，然后在dataloader的collate_fn中过滤掉None
            # 但这里为了简化，仍然尝试返回上一个有效样本，但这可能导致数据重复
            # 更好的做法是使用自定义的collate_fn来处理None
            # 为了避免无限递归，只尝试一次回退
            if idx > 0:
                 print(f"尝试加载前一个样本 (idx={idx-1})")
                 return self.__getitem__(idx - 1)
            else:
                # 如果是第一个样本就出错，则无法回退，抛出原始异常
                raise e


def calculate_psnr(original, generated):
    """计算PSNR"""
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy()
    if isinstance(generated, torch.Tensor):
        generated = generated.detach().cpu().numpy()
    
    # 确保数据类型和范围正确
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
    
    # 确保数据类型和范围正确
    original = np.clip(original * 255, 0, 255).astype(np.uint8)
    generated = np.clip(generated * 255, 0, 255).astype(np.uint8)

    # 确保输入是HWC格式以便 skimage.ssim 处理
    if len(original.shape) == 4: # Batch dimension
        original = original[0]
        generated = generated[0]

    if original.shape[0] == 3: # CHW format
        original = np.transpose(original, (1, 2, 0))
        generated = np.transpose(generated, (1, 2, 0))
    elif original.shape[0] == 1: # Grayscale CHW, remove channel dim
        original = np.squeeze(original, axis=0)
        generated = np.squeeze(generated, axis=0)


    h, w = original.shape[:2]
    # 调整 win_size
    win_size = min(11, min(h, w) - 1) # 使用更常用的 11x11 窗口，但不要超过图像尺寸
    if win_size % 2 == 0:
        win_size -= 1
    if win_size < 3:
        win_size = 3

    # 确保输入是灰度或RGB
    if original.ndim == 3 and original.shape[2] == 3: # RGB
        channel_axis = 2
        multichannel = True
    elif original.ndim == 2: # Grayscale
        channel_axis = None
        multichannel = False
    else:
         print(f"警告: SSIM输入图像形状非预期: {original.shape}")
         return 0.0 # 返回0或NaN表示计算失败

    try:
        ssim_val = ssim(original, generated,
                    channel_axis=channel_axis,
                    data_range=255,
                    win_size=win_size,
                    multichannel=multichannel if channel_axis is not None else False) # skimage 1.0+ 使用 multichannel
        return ssim_val
    except Exception as e:
        print(f"计算SSIM时出错: {str(e)}")
        return 0.0 # 返回0表示计算失败


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
                if h < 8 or w < 8: # 确保图像尺寸足够进行卷积等操作
                    print(f"Warning: Validation image size {h}x{w} is too small, skipping")
                    continue

                # 调用模型，不需要 previous_frame 进行验证（除非模型设计要求）
                # 如果 InnovativeDeblurGAN 在验证时需要 previous_frame，这里的调用需要修改
                deblurred, motion_map = model(blur_images) # Assuming motion_map is returned even in eval mode

                if torch.isnan(deblurred).any() or torch.isinf(deblurred).any():
                    print("Warning: NaN or Inf detected in generated validation images")
                    continue

                # 将图像从[-1, 1]归一化到[0, 1]范围，以计算PSNR/SSIM
                deblurred_norm = (deblurred + 1) / 2
                clear_images_norm = (clear_images + 1) / 2

                for i in range(deblurred_norm.size(0)):
                    try:
                        # 将单张图像从[0, 1]范围转换为[0, 255]并转换为uint8 numpy数组计算指标
                        deblurred_np = (deblurred_norm[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        clear_images_np = (clear_images_norm[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

                        psnr = calculate_psnr(clear_images_np, deblurred_np)
                        ssim_val = calculate_ssim(clear_images_np, deblurred_np)

                        if np.isfinite(psnr) and 0 <= ssim_val <= 1:
                            total_psnr += psnr
                            total_ssim += ssim_val
                            num_samples += 1
                        else:
                            print(f"Warning: Invalid metrics - PSNR: {psnr}, SSIM: {ssim_val}")
                    except Exception as e:
                        print(f"Error processing sample {i} in validation batch: {str(e)}")
                        continue

            except Exception as e:
                print(f"Error in validation batch: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

    if num_samples == 0:
        print("Warning: No valid samples were evaluated in validation")
        return 0.0, 0.0

    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples

    return avg_psnr, avg_ssim


def save_checkpoint(epoch, model, g_optimizer, d_optimizer, avg_g_loss_for_epoch, avg_d_loss_for_epoch, metrics, output_dir, train_dataloader, current_batch_idx=None, previous_frame=None, previous_motion_map=None, current_epoch_g_loss_cumulative=0.0, current_epoch_d_loss_cumulative=0.0, g_scheduler=None, d_scheduler=None):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': model.generator.state_dict(),
        'discriminator_state_dict': model.discriminator.state_dict(),
        # 安全地保存 motion_estimator 的状态字典，如果存在
        'motion_estimator_state_dict': model.motion_estimator.state_dict() if hasattr(model, 'motion_estimator') else None,
        'g_optimizer_state_dict': g_optimizer.state_dict(),
        'd_optimizer_state_dict': d_optimizer.state_dict(),
        'g_scheduler_state_dict': g_scheduler.state_dict() if g_scheduler is not None else None,
        'd_scheduler_state_dict': d_scheduler.state_dict() if d_scheduler is not None else None,
        'avg_g_loss_epoch': avg_g_loss_for_epoch, # epoch 的平均损失
        'avg_d_loss_epoch': avg_d_loss_for_epoch, # epoch 的平均损失
        'metrics': metrics,
        'current_batch': current_batch_idx, # 当前批次索引，如果保存的是批次内检查点
        'previous_frame': previous_frame.detach().cpu() if previous_frame is not None else None,
        'previous_motion_map': previous_motion_map.detach().cpu() if previous_motion_map is not None else None,
        'current_epoch_g_loss_cumulative': current_epoch_g_loss_cumulative, # 当前 epoch 累计 G 损失
        'current_epoch_d_loss_cumulative': current_epoch_d_loss_cumulative  # 当前 epoch 累计 D 损失
    }

    # 始终保存最新检查点
    checkpoint_path_latest = f"{output_dir}/checkpoint_latest.pth"
    try:
         torch.save(checkpoint, checkpoint_path_latest)
         print(f"检查点已保存: {os.path.basename(checkpoint_path_latest)}")
    except Exception as e:
         print(f"保存最新检查点失败: {str(e)}")
         traceback.print_exc()

    # 如果是 epoch 结束时保存的检查点 (current_batch_idx 为 None)
    if current_batch_idx is None:
        # 如果是最佳模型，也保存一份 (这里以 PSNR 作为最佳指标)
        if metrics and metrics.get('psnr', -float('inf')) > (getattr(save_checkpoint, 'best_psnr', -float('inf'))):
             try:
                 torch.save(checkpoint, f"{output_dir}/checkpoint_best.pth")
                 save_checkpoint.best_psnr = metrics['psnr'] # 更新最佳 PSNR
                 print(f"保存最佳检查点: {output_dir}/checkpoint_best.pth")
             except Exception as e:
                  print(f"保存最佳检查点失败: {str(e)}")
                  traceback.print_exc()

        # 每10个epoch保存一次
        if (epoch + 1) % 10 == 0:
            try:
                 torch.save(checkpoint, f"{output_dir}/checkpoint_epoch_{epoch+1}.pth")
                 print(f"保存定期检查点: {os.path.basename(f'checkpoint_epoch_{epoch+1}.pth')}")
            except Exception as e:
                 print(f"保存定期检查点失败: {str(e)}")
                 traceback.print_exc()


# 初始化 best_psnr 属性
save_checkpoint.best_psnr = -float('inf')
def load_checkpoint(checkpoint_path, model, g_optimizer, d_optimizer, g_scheduler=None, d_scheduler=None):
    """加载检查点"""
    if not os.path.exists(checkpoint_path):
        print(f"检查点文件不存在: {checkpoint_path}")
        return 0, None, None, 0.0, 0.0, None # 返回: start_epoch, previous_frame, previous_motion_map, current_epoch_g_loss_cumulative, current_epoch_d_loss_cumulative, current_batch_idx

    print(f"加载检查点: {checkpoint_path}")
    try:
         checkpoint = torch.load(checkpoint_path, map_location=model.generator.conv1.weight.device) # 确保加载到正确的设备
    except Exception as e:
         print(f"加载检查点文件失败: {str(e)}")
         traceback.print_exc()
         print("尝试从头开始训练。")
         return 0, None, None, 0.0, 0.0, None


    try:
        # 加载各个组件的状态
        model.generator.load_state_dict(checkpoint['generator_state_dict'], strict=False)
        model.discriminator.load_state_dict(checkpoint['discriminator_state_dict'], strict=False)

        # 安全加载 motion_estimator 状态字典
        if hasattr(model, 'motion_estimator') and 'motion_estimator_state_dict' in checkpoint and checkpoint['motion_estimator_state_dict'] is not None:
            try:
                 model.motion_estimator.load_state_dict(checkpoint['motion_estimator_state_dict'], strict=False)
            except RuntimeError as e:
                 print(f"警告: 加载 motion_estimator 状态字典时发生运行时错误: {e}")
                 print("请检查模型和检查点中 motion_estimator 的结构是否匹配。跳过加载 motion_estimator。")

        # 加载优化器状态
        g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])

        # 加载学习率调度器状态
        if g_scheduler is not None and 'g_scheduler_state_dict' in checkpoint and checkpoint['g_scheduler_state_dict'] is not None:
            g_scheduler.load_state_dict(checkpoint['g_scheduler_state_dict'])
        if d_scheduler is not None and 'd_scheduler_state_dict' in checkpoint and checkpoint['d_scheduler_state_dict'] is not None:
            d_scheduler.load_state_dict(checkpoint['d_scheduler_state_dict'])

        start_epoch = checkpoint.get('epoch', 0) 
        current_batch_idx = checkpoint.get('current_batch', None) # None 表示 epoch 结束时保存

        # 恢复 previous_frame 和 previous_motion_map
        previous_frame = checkpoint.get('previous_frame', None)
        previous_motion_map = checkpoint.get('previous_motion_map', None)

        # 恢复当前 epoch 累计损失
        current_epoch_g_loss_cumulative = checkpoint.get('current_epoch_g_loss_cumulative', 0.0)
        current_epoch_d_loss_cumulative = checkpoint.get('current_epoch_d_loss_cumulative', 0.0)

        print(f"已加载第 {start_epoch} 轮的检查点。")
        if current_batch_idx is not None: # 如果是批次内检查点，则从该批次继续
            print(f"将从批次 {current_batch_idx + 1} 开始继续训练（位于第 {start_epoch + 1} 轮）。")
            # 如果是批次内恢复，起始 epoch 保持不变，但需要从下一批次开始
        else: # 如果是 epoch 结束时保存的检查点，则从下一轮开始
            start_epoch = start_epoch + 1

        if 'metrics' in checkpoint:
            print(f"检查点指标: {checkpoint['metrics']}")
            # 恢复最佳 PSNR 状态以便正确保存 best 检查点
            if checkpoint['metrics'].get('psnr', -float('inf')) > save_checkpoint.best_psnr:
                save_checkpoint.best_psnr = checkpoint['metrics']['psnr']

        return start_epoch, previous_frame, previous_motion_map, current_epoch_g_loss_cumulative, current_epoch_d_loss_cumulative, current_batch_idx

    except Exception as e:
        print(f"解析检查点内容时出现错误: {str(e)}")
        traceback.print_exc()
        print("继续使用新的模型参数从头开始训练")
        return 0, None, None, 0.0, 0.0, None # 如果加载失败，从头开始训练


def pre_training_check(model, dataloader, val_dataloader, device):
    """预训练检查"""
    print("开始预训练检查...")

    # 1. 检查GPU
    if torch.cuda.is_available():
        print(f"GPU可用: {torch.cuda.get_device_name(0)}")
        try:
             gpu_properties = torch.cuda.get_device_properties(0)
             print(f"GPU内存: {gpu_properties.total_memory / 1024**3:.2f} GB")
             print(f"GPU算力: {gpu_properties.major}.{gpu_properties.minor}")
        except Exception as e:
             print(f"无法获取GPU属性: {str(e)}")
    else:
        print("警告: 未检测到GPU，将使用CPU训练（训练速度会很慢）")

    # 2. 检查数据集
    print("\n检查数据集...")
    try:
        # 尝试加载一个批次
        sample_batch = next(iter(dataloader))
        blur_images = sample_batch.get('blur')
        clear_images = sample_batch.get('clear')

        if blur_images is None or clear_images is None:
             raise ValueError("数据加载器未返回 'blur' 或 'clear' 键")

        print(f"批次大小: {blur_images.size(0)}")
        print(f"图像尺寸: {blur_images.size(2)}x{blur_images.size(3)}")
        print(f"通道数: {blur_images.size(1)}")

        print(f"模糊图像值范围: [{blur_images.min().item():.3f}, {blur_images.max().item():.3f}]")
        print(f"清晰图像值范围: [{clear_images.min().item():.3f}, {clear_images.max().item():.3f}]")

        # 检查图像尺寸是否过小 (例如小于卷积核或池化窗口)
        if blur_images.size(2) < 8 or blur_images.size(3) < 8:
            raise ValueError("图像尺寸太小，建议至少8x8或更大，取决于模型架构。")

        # 检查值范围是否合理（例如归一化后在[-1, 1]）
        if blur_images.min() < -1.05 or blur_images.max() > 1.05:
             print("警告: 模糊图像值范围超出[-1, 1]较多，请检查数据预处理。")
        if clear_images.min() < -1.05 or clear_images.max() > 1.05:
             print("警告: 清晰图像值范围超出[-1, 1]较多，请检查数据预处理。")


    except StopIteration:
        print("错误: 训练数据加载器为空，请检查数据集路径和文件。")
        return False
    except Exception as e:
        print(f"数据集检查失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    # 3. 检查模型
    print("\n检查模型...")
    try:
        model = model.to(device)
        model.eval() # 切换到评估模式进行检查

        # 尝试进行一次前向传播
        with torch.no_grad():
             # 使用一个示例输入（第一个batch的模糊图像）进行前向检查
             example_input = blur_images.to(device)
             # 在预检查时，previous_frame 可以设置为 None
             deblurred, motion_map = model(example_input, previous_frame=None)

        print(f"生成器输出尺寸: {deblurred.size()}")

        # 检查运动核输出
        if motion_map is not None:
             print(f"运动图输出尺寸: {motion_map.size()}")
             # 检查运动核的形状是否符合预期 (N, C, H, W)
             if motion_map.ndim != 4:
                  print(f"警告: 运动图输出维度非4，为{motion_map.ndim}。")
             # 如果预期是 (N, 6, 1, 1)
             elif motion_map.shape[2:] == (1, 1) and motion_map.shape[1] == 6:
                  print("运动图输出尺寸符合 (N, 6, 1, 1) 预期。")
             elif motion_map.shape[1] == 2 and motion_map.shape[2:] == deblurred.shape[2:]:
                  print("运动图输出尺寸符合 (N, 2, H, W) 运动场预期。")
             else:
                   print(f"警告: 运动图输出尺寸 {motion_map.shape} 非典型形状。")


        # 检查判别器
        d_output = model.discriminator(clear_images.to(device))
        print(f"判别器输出尺寸: {d_output.size()}")

        # 检查输出值范围
        print(f"生成器输出范围: [{deblurred.min().item():.3f}, {deblurred.max().item():.3f}]")
        print(f"判别器输出范围: [{d_output.min().item():.3f}, {d_output.max().item():.3f}]")

        # 检查模型参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"模型总参数数量: {total_params}")
        print(f"可训练参数数量: {trainable_params}")

    except Exception as e:
        print(f"模型检查失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    # 4. 检查评估指标
    print("\n检查评估指标...")
    try:
        # 使用示例数据计算一次指标
        deblurred_example_np = (deblurred[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        clear_images_example_np = (clear_images[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        psnr = calculate_psnr(clear_images_example_np, deblurred_example_np)
        print(f"PSNR测试值: {psnr:.2f}")

        ssim_val = calculate_ssim(clear_images_example_np, deblurred_example_np)
        print(f"SSIM测试值: {ssim_val:.4f}")

        if not (0 <= ssim_val <= 1 or np.isnan(ssim_val)): # 允许SSIM为NaN，因为可能计算失败
            print(f"警告: SSIM测试值超出范围或异常: {ssim_val}")


    except Exception as e:
        print(f"评估指标检查失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    # 5. 检查内存使用
    print("\n检查内存使用...")
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache() # 清理缓存
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            cached = torch.cuda.memory_reserved(0) / 1024**3
            print(f"GPU内存使用: {allocated:.2f} GB (已分配)")
            print(f"GPU内存缓存: {cached:.2f} GB (已保留)")

            # 简单判断内存是否可能不足 (如果已分配内存超过总内存的90%)
            gpu_properties = torch.cuda.get_device_properties(0)
            if allocated > 0.9 * gpu_properties.total_memory / 1024**3:
                 print("警告: GPU内存使用率过高，训练可能需要更小的批次或模型。")
                 # return False # 可以选择在这里阻止训练
    except Exception as e:
        print(f"内存检查失败: {str(e)}")
        import traceback
        traceback.print_exc()
        # return False # 可以选择在这里阻止训练

    print("\n所有检查通过！可以开始训练。")
    model.train() # 检查后切换回训练模式
    return True



def train_innovative_model():
    # 设置参数
    batch_size = 8
    num_epochs = 100
    learning_rate = 0.0002
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = "training_output"

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # 减小图像尺寸
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    try:
        # 加载数据集
        print("正在加载数据集...")
        dataset = DeblurDataset(
            blur_dir=r'C:\Users\91144\Desktop\dongtaimohu\dataset\processed\blur',
            clear_dir=r'C:\Users\91144\Desktop\dongtaimohu\dataset\processed\sharp',
            transform=transform
        )
        print(f"数据集加载成功，共 {len(dataset)} 个样本")

        # 划分训练集和验证集
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        # 确保训练集和验证集都有数据
        if train_size == 0 or val_size == 0:
             print("错误: 数据集大小不足以创建训练集和验证集。请确保数据集包含至少2个样本。")
             return
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        # 创建数据加载器
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4, # 可以根据系统资源调整 num_workers
            pin_memory=True if torch.cuda.is_available() else False,  # 只有在使用GPU时使用固定内存
            drop_last=True  # 丢弃不完整的最后一个batch
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4, # 可以根据系统资源调整 num_workers
            pin_memory=True if torch.cuda.is_available() else False # 只有在使用GPU时使用固定内存
        )

        print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")

        # 初始化模型
        print("正在初始化模型...")
        model = InnovativeDeblurGAN().to(device)
        # 实例化 DeblurLoss，可以调整权重
        criterion = DeblurLoss(recon_weight=1.0, # 重建损失权重
                               motion_smooth_weight=0.2, # 运动平滑损失权重
                               motion_consistency_weight=0.1, # 运动一致性损失权重
                               motion_reg_weight=0.05, # 运动正则化权重
                               perceptual_weight=0.5) # 感知损失权重，可以根据效果调整

        # 优化器
        g_optimizer = optim.Adam(model.generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        # 判别器学习率通常低于生成器
        d_optimizer = optim.Adam(model.discriminator.parameters(), lr=learning_rate * 0.5, betas=(0.5, 0.999)) # 将判别器学习率调整为生成器的一半

        # ======================================================================
        # 新增：学习率调度器
        # ======================================================================
        # Step decay scheduler: decay learning rate by gamma every step_size epochs
        step_size = 30 # Decay every 30 epochs
        gamma = 0.5 # Decay rate
        g_scheduler = lr_scheduler.StepLR(g_optimizer, step_size=step_size, gamma=gamma)
        d_scheduler = lr_scheduler.StepLR(d_optimizer, step_size=step_size, gamma=gamma)
        print(f"已设置学习率调度器: StepLR (step_size={step_size}, gamma={gamma})")
        # ======================================================================


        # 预训练检查
        if not pre_training_check(model, train_dataloader, val_dataloader, device):
            print("预训练检查失败，请解决上述问题后再开始训练。")
            return

        # 尝试加载最新检查点
        start_epoch, previous_frame, previous_motion_map, current_epoch_g_loss_cumulative, current_epoch_d_loss_cumulative, last_batch_idx = \
            load_checkpoint(f"{output_dir}/checkpoint_latest.pth",
                            model, g_optimizer, d_optimizer, g_scheduler, d_scheduler)

        # 将恢复的 previous_frame 和 previous_motion_map 移动到设备上
        if previous_frame is not None:
            previous_frame = previous_frame.to(device)
        if previous_motion_map is not None:
            previous_motion_map = previous_motion_map.to(device)

        # 训练循环
        print(f"开始训练，从第 {start_epoch} 轮开始...")
        for epoch in range(start_epoch, num_epochs):
            model.train()
            
            # 如果是新的一轮 (不是在上一轮中途恢复)，则重置本轮的损失累加器和 previous_frame/motion_map
            # 否则，如果是从上轮中途恢复，则继续使用从检查点加载的累计损失和 previous_frame/motion_map
            if last_batch_idx is None or epoch > start_epoch: # last_batch_idx 为 None 表示是从 epoch 结束时恢复，或者 epoch > start_epoch 表示进入了新的 epoch
                current_epoch_g_loss_cumulative = 0.0
                current_epoch_d_loss_cumulative = 0.0
                previous_frame = None # 在新 epoch 的第一个批次会重新初始化
                previous_motion_map = None # 在新 epoch 的第一个批次会重新初始化

            # 确定本轮训练的起始批次索引
            # 如果是从上一轮中途恢复，则从 last_batch_idx + 1 开始
            # 否则 (新的一轮或从头开始)，从 0 开始
            epoch_start_batch_idx = last_batch_idx + 1 if last_batch_idx is not None and epoch == start_epoch else 0
            # 重置 last_batch_idx，确保在新的 epoch 中不会跳过批次
            last_batch_idx = None 

            for i, batch in enumerate(train_dataloader):
                if i < epoch_start_batch_idx:
                    print(f"跳过批次 {i+1} (已在检查点中处理)")
                    continue

                try:
                    # 如果 previous_frame 在 epoch 开始时被重置为 None (即新 epoch 的第一个批次)
                    # 则使用当前批次的 clear_images 初始化 previous_frame 和 previous_motion_map
                    if previous_frame is None:
                        previous_frame = batch['clear'].to(device).detach().clone()
                        # 对于第一个批次，motion_map 会由 model(blur_images, None) 生成，
                        # initial previous_motion_map will be None, leading to motion_consistency_loss = 0.0
                        # which is expected for the very first frame.
                        # 确保 previous_motion_map 初始化
                        # 这里不直接初始化 previous_motion_map 为零，因为 motion_map 是由模型计算的
                        # 只有当模型输出 motion_map 后，previous_motion_map 才能被更新
                        # 在 DeblurLoss 中，如果 previous_motion_map 为 None，motion_consistency_loss 就会是 0，这是正确的逻辑

                    # pdb.set_trace() # 在这里设置断点 - 移除断点
                    blur_images = batch['blur'].to(device)
                    clear_images = batch['clear'].to(device)

                    # 训练判别器
                    d_optimizer.zero_grad()
                    real_output = model.discriminator(clear_images)
                    d_real_loss = nn.BCEWithLogitsLoss()(real_output, torch.ones_like(real_output))

                    # 生成去模糊图像和运动图
                    deblurred, motion_map = model(blur_images, previous_frame)

                    # 判别器训练
                    fake_output = model.discriminator(deblurred.detach()) # detach() 避免梯度回传到生成器
                    d_fake_loss = nn.BCEWithLogitsLoss()(fake_output, torch.zeros_like(fake_output))
                    d_loss = d_real_loss + d_fake_loss
                    d_loss.backward()
                    d_optimizer.step()

                    # 训练生成器
                    g_optimizer.zero_grad()
                    fake_output_gen = model.discriminator(deblurred)
                    g_gan_loss = nn.BCEWithLogitsLoss()(fake_output_gen, torch.ones_like(fake_output_gen))

                    # 计算去模糊损失，传入 previous_motion_map
                    # 确保 motion_map 和 previous_motion_map 在同一设备上
                    deblur_loss, loss_dict = criterion(deblurred, clear_images, motion_map, 
                                                     previous_frame, previous_motion_map)

                    # 生成器总损失
                    g_loss = g_gan_loss + deblur_loss
                    g_loss.backward()
                    g_optimizer.step()

                    # 更新 previous_frame 和 previous_motion_map
                    previous_frame = clear_images.detach().clone() # 使用清晰图像作为下一批次的前一帧
                    previous_motion_map = motion_map.detach().clone() # 使用当前运动图作为下一批次的前一运动图

                    # 累加损失
                    current_epoch_g_loss_cumulative += g_loss.item()
                    current_epoch_d_loss_cumulative += d_loss.item()

                    # 打印训练信息
                    if i % 10 == 0:
                         print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_dataloader)}], '\
                              f'D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}, '\
                              f'Recon_loss: {loss_dict["recon_loss"]:.4f}, '\
                              f'Perceptual_loss: {loss_dict["perceptual_loss"]:.4f}, '\
                              f'Motion_smooth_loss: {loss_dict["motion_smooth_loss"]:.4f}, '\
                              f'Motion_consistency_loss: {loss_dict["motion_consistency_loss"]:.4f}, '\
                              f'Motion_reg: {loss_dict["motion_reg"]:.4f}')

                    # 保存检查点 (每100个批次)
                    if i % 100 == 0:
                        save_checkpoint(epoch=epoch,
                                        model=model,
                                        g_optimizer=g_optimizer,
                                        d_optimizer=d_optimizer,
                                        # 注意: 这里传入的是当前 batch 的平均损失，不是 epoch 结束时的最终平均损失
                                        avg_g_loss_for_epoch=current_epoch_g_loss_cumulative / (i + 1),
                                        avg_d_loss_for_epoch=current_epoch_d_loss_cumulative / (i + 1),
                                        metrics={}, # 批次内不计算指标
                                        output_dir=output_dir,
                                        train_dataloader=train_dataloader,
                                        current_batch_idx=i, # 保存当前批次索引
                                        previous_frame=previous_frame, # 保存最近更新的 previous_frame
                                        previous_motion_map=previous_motion_map, # 保存最近更新的 previous_motion_map
                                        current_epoch_g_loss_cumulative=current_epoch_g_loss_cumulative, # 累计损失
                                        current_epoch_d_loss_cumulative=current_epoch_d_loss_cumulative, # 累计损失
                                        g_scheduler=g_scheduler,
                                        d_scheduler=d_scheduler)

                except Exception as e:
                    print(f"处理批次 {i} 时出错: {str(e)}")
                    traceback.print_exc()
                    continue

            # 更新学习率
            g_scheduler.step()
            d_scheduler.step()

            # 验证模型
            model.eval()
            avg_psnr, avg_ssim = validate_model(model, val_dataloader, device)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Validation PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")

            # 在 epoch 结束时保存检查点
            # 这里 avg_g_loss_for_epoch 和 avg_d_loss_for_epoch 是整个 epoch 的平均损失
            save_checkpoint(epoch=epoch,
                            model=model,
                            g_optimizer=g_optimizer,
                            d_optimizer=d_optimizer,
                            avg_g_loss_for_epoch=current_epoch_g_loss_cumulative / len(train_dataloader),
                            avg_d_loss_for_epoch=current_epoch_d_loss_cumulative / len(train_dataloader),
                            metrics={'psnr': avg_psnr, 'ssim': avg_ssim},
                            output_dir=output_dir,
                            train_dataloader=train_dataloader,
                            current_batch_idx=None, # 标志为 epoch 结束时保存
                            previous_frame=previous_frame, # epoch 最后一帧的 previous_frame
                            previous_motion_map=previous_motion_map, # epoch 最后一帧的 previous_motion_map
                            current_epoch_g_loss_cumulative=current_epoch_g_loss_cumulative, # epoch 累计损失
                            current_epoch_d_loss_cumulative=current_epoch_d_loss_cumulative, # epoch 累计损失
                            g_scheduler=g_scheduler,
                            d_scheduler=d_scheduler)

        # Save final model
        torch.save(model.state_dict(), f"{output_dir}/innovative_deblur_final.pth")
        print("\n训练完成！")

    except Exception as e:
        print(f"训练过程中出现顶层错误: {str(e)}")
        traceback.print_exc() # 打印详细错误信息
        raise e # Re-raise the exception

# Initialize best_psnr
train_innovative_model.best_psnr = -float('inf')

if __name__ == '__main__':
    # Load best_psnr from checkpoint_best.pth if it exists
    output_dir = "training_output" # Define output_dir here as well for loading best_psnr, ensure consistency with training
    best_checkpoint_path = f"{output_dir}/checkpoint_best.pth"
    if os.path.exists(best_checkpoint_path):
        try:
            # 加载检查点以恢复 best_psnr
            best_checkpoint = torch.load(best_checkpoint_path, map_location='cpu') # Load to CPU
            if 'metrics' in best_checkpoint and best_checkpoint['metrics'] is not None:
                 train_innovative_model.best_psnr = best_checkpoint['metrics'].get('psnr', -float('inf'))
                 print(f"从 {os.path.basename(best_checkpoint_path)} 加载最佳PSNR: {train_innovative_model.best_psnr:.4f}")
        except Exception as e:
            print(f"加载最佳检查点失败以恢复 best_psnr: {str(e)}")

    train_innovative_model()