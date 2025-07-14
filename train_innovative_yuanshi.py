import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from innovative_deblur import InnovativeDeblurGAN, DeblurLoss
from skimage.metrics import structural_similarity as ssim
from torchvision.utils import save_image
import traceback # 导入 traceback 模块

# 请确保 innovative_deblur.py 文件存在并包含 InnovativeDeblurGAN 和 DeblurLoss 的定义
# 为了代码的可执行性，这里假设了这些类的存在。
# 如果没有，请将它们的定义粘贴到这里或确保文件可导入。

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


def save_checkpoint(epoch, model, g_optimizer, d_optimizer, avg_g_loss, avg_d_loss, metrics, output_dir, train_dataloader, training_state=None):
    """保存检查点"""
    # 计算当前epoch的总批次数量
    total_batches = len(train_dataloader)

    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': model.generator.state_dict(),
        'discriminator_state_dict': model.discriminator.state_dict(),
        # 安全地保存 motion_estimator 的状态字典，如果存在
        'motion_estimator_state_dict': model.motion_estimator.state_dict() if hasattr(model, 'motion_estimator') else None,
        'g_optimizer_state_dict': g_optimizer.state_dict(),
        'd_optimizer_state_dict': d_optimizer.state_dict(),
        # 保存 epoch 的平均损失，而不是累积损失
        'avg_g_loss_epoch': avg_g_loss,
        'avg_d_loss_epoch': avg_d_loss,
        'metrics': metrics,
        # 保存传递进来的 training_state
        'training_state': training_state
    }

    # 如果是 batch 内部保存的检查点 (training_state 不为None)
    if training_state is not None:
         # 在文件名中包含 batch 索引以便区分
         checkpoint_path_latest = f"{output_dir}/checkpoint_latest_epoch_{epoch}_batch_{training_state.get('current_batch', 0)}.pth"
         checkpoint_path_epoch = f"{output_dir}/checkpoint_epoch_{epoch+1}_batch_{training_state.get('current_batch', 0)}.pth"
    else:
        # 如果是 epoch 结束时保存的检查点
        checkpoint_path_latest = f"{output_dir}/checkpoint_latest.pth"
        checkpoint_path_epoch = f"{output_dir}/checkpoint_epoch_{epoch+1}.pth"


    # 保存最新检查点
    try:
         torch.save(checkpoint, checkpoint_path_latest)
         print(f"检查点已保存: {os.path.basename(checkpoint_path_latest)}")
    except Exception as e:
         print(f"保存最新检查点失败: {str(e)}")
         import traceback
         traceback.print_exc()


    # 如果是最佳模型，也保存一份 (这里简单以 PSNR 作为最佳指标)
    # 最佳模型的判断应该基于 epoch 结束时的验证指标
    if metrics and metrics.get('psnr', -float('inf')) > (getattr(save_checkpoint, 'best_psnr', -float('inf'))):
        if training_state is None: # 只在 epoch 结束时保存最佳模型
             try:
                 torch.save(checkpoint, f"{output_dir}/checkpoint_best.pth")
                 save_checkpoint.best_psnr = metrics['psnr'] # 更新最佳 PSNR
                 print(f"保存最佳检查点: {output_dir}/checkpoint_best.pth")
             except Exception as e:
                  print(f"保存最佳检查点失败: {str(e)}")
                  import traceback
                  traceback.print_exc()


    # 每10个epoch保存一次 (只在 epoch 结束时)
    if training_state is None and (epoch + 1) % 10 == 0:
        try:
             torch.save(checkpoint, checkpoint_path_epoch)
             print(f"保存定期检查点: {os.path.basename(checkpoint_path_epoch)}")
        except Exception as e:
             print(f"保存定期检查点失败: {str(e)}")
             import traceback
             traceback.print_exc()

    # 移除旧的 batch 检查点，只保留最新的
    if training_state is not None:
        # 可以在这里添加逻辑来清理旧的 batch 检查点文件
        pass # 暂时不做清理，以免影响调试


# 初始化 best_psnr 属性
save_checkpoint.best_psnr = -float('inf')


def load_checkpoint(checkpoint_path, model, g_optimizer, d_optimizer):
    """加载检查点"""
    if not os.path.exists(checkpoint_path):
        print(f"检查点文件不存在: {checkpoint_path}")
        return 0, None # 返回0表示从头开始，返回None表示没有恢复训练状态

    print(f"加载检查点: {checkpoint_path}")
    try:
         checkpoint = torch.load(checkpoint_path, map_location=model.generator.initial[0].weight.device) # 确保加载到正确的设备
    except Exception as e:
         print(f"加载检查点文件失败: {str(e)}")
         import traceback
         traceback.print_exc()
         print("尝试从头开始训练。")
         return 0, None # 如果加载文件失败，从头开始


    try:
        # 加载各个组件的状态，使用strict=False来允许部分加载（例如motion_estimator可能在新旧模型中存在差异）
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

        start_epoch = checkpoint.get('epoch', 0) + 1 # 从下一个epoch开始
        print(f"已加载第 {start_epoch - 1} 轮的检查点")

        if 'metrics' in checkpoint:
            print(f"检查点指标: {checkpoint['metrics']}")
            # 恢复最佳 PSNR 状态以便正确保存 best 检查点
            if checkpoint['metrics'].get('psnr', -float('inf')) > save_checkpoint.best_psnr:
                save_checkpoint.best_psnr = checkpoint['metrics']['psnr']

        # 恢复 training_state
        training_state = checkpoint.get('training_state', None)
        if training_state is not None:
             print("恢复训练状态。")
             # 可以在这里根据 training_state 进一步设置训练状态，例如恢复 previous_frame
             # if 'previous_frame' in training_state and training_state['previous_frame'] is not None:
             #      # 需要将 previous_frame 移动到设备上
             #      training_state['previous_frame'] = training_state['previous_frame'].to(model.generator.conv1.weight.device)
             pass # training_state 会在 train_innovative_model 函数中根据需要使用


        return start_epoch, training_state # 返回起始epoch和恢复的训练状态

    except Exception as e:
        print(f"解析检查点内容时出现错误: {str(e)}")
        print("继续使用新的模型参数从头开始训练")
        import traceback
        traceback.print_exc()
        return 0, None # 如果加载失败，从头开始训练


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
        criterion = DeblurLoss()

        # 优化器
        g_optimizer = optim.Adam(model.generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        # 判别器学习率通常低于生成器
        d_optimizer = optim.Adam(model.discriminator.parameters(), lr=learning_rate * 0.5, betas=(0.5, 0.999)) # 将判别器学习率调整为生成器的一半

        # 预训练检查
        if not pre_training_check(model, train_dataloader, val_dataloader, device):
            print("预训练检查失败，请解决上述问题后再开始训练。")
            return

        # 尝试加载最新检查点
        # training_state 在 load_checkpoint 中不再包含累积损失，只可能包含 previous_frame
        start_epoch, training_state = load_checkpoint(r"C:\Users\91144\Desktop\dongtaimohu\training_output\checkpoint_latest_epoch_74_batch_0.pth",
                                                    model, g_optimizer, d_optimizer)

        # 训练循环
        print(f"开始训练，从第 {start_epoch} 轮开始...")
        for epoch in range(start_epoch, num_epochs):
            model.train()

            # 在每个epoch开始时，重新初始化累积损失
            current_epoch_g_loss = torch.tensor(0.0, device=device)
            current_epoch_d_loss = torch.tensor(0.0, device=device)

            # 处理从检查点恢复的 training_state
            if training_state is not None:
                 # 如果从检查点恢复了 previous_frame，则使用它
                 if 'previous_frame' in training_state and training_state['previous_frame'] is not None:
                      previous_frame = training_state['previous_frame'].to(device) # 将恢复的 previous_frame 移动到设备
                      print(f"从检查点恢复 previous_frame.")
                 else:
                      previous_frame = None # 检查点中没有 previous_frame 或为None
                      print(f"检查点中没有previous_frame或为None，将在第一个batch初始化。")

                 # 如果需要从检查点恢复累积损失（通常在 batch 断点时需要），可以在这里加载
                 # 例如： current_epoch_g_loss = torch.tensor(training_state.get('current_epoch_g_loss', 0.0), device=device)
                 #       current_epoch_d_loss = torch.tensor(training_state.get('current_epoch_d_loss', 0.0), device=device)

                 # 检查点状态只在加载后使用一次，避免影响后续epoch
                 training_state = None # 使用后将 training_state 设为 None

            else:
                # 如果没有从检查点加载，或者检查点中没有previous_frame
                previous_frame = None # 明确初始化 previous_frame 为 None，将在第一个batch初始化
                print(f"没有从检查点加载 previous_frame，将在第一个batch初始化。")


            for i, batch in enumerate(train_dataloader):
                try:
                    # 获取数据
                    blur_images = batch.get('blur')
                    clear_images = batch.get('clear')

                    # 再次检查数据加载是否成功
                    if blur_images is None or clear_images is None:
                         print(f"警告: 批次 {i} 数据加载失败，跳过。")
                         continue

                    blur_images = blur_images.to(device)
                    clear_images = clear_images.to(device)

                    # 在第一个批次初始化 previous_frame 如果它为None
                    if previous_frame is None:
                        # 使用当前批次的清晰图像作为第一个 previous_frame
                        previous_frame = clear_images.detach().clone()
                        print(f"在批次 {i} 初始化 previous_frame.")


                    # 验证输入数据是否有NaN/Inf
                    if torch.isnan(blur_images).any() or torch.isinf(blur_images).any() or \
                       torch.isnan(clear_images).any() or torch.isinf(clear_images).any():
                        print(f"警告: 批次 {i} 的输入包含NaN/Inf值，跳过")
                        continue

                    # 训练判别器
                    d_optimizer.zero_grad()

                    # 真实图像
                    real_output = model.discriminator(clear_images)
                    # 确保目标标签在[0,1]范围内
                    d_real_loss = nn.BCEWithLogitsLoss()(real_output, torch.ones_like(real_output)) # 使用BCEWithLogitsLoss更稳定，目标设为1.0
                    # d_real_loss = nn.BCELoss()(real_output, torch.ones_like(real_output) * 0.9) # 原来的平滑标签

                    # 生成图像 - 使用前一帧
                    deblurred, motion_map = model(blur_images, previous_frame)

                    # 验证模型输出
                    if torch.isnan(deblurred).any() or torch.isinf(deblurred).any():
                         print(f"警告: 批次 {i} 的去模糊图像输出包含无效值，跳过此批次。")
                         continue # 跳过当前批次的判别器和生成器训练

                    # 验证运动核 (现在是运动场 (N, 2, H, W))
                    # 检查运动场的尺寸是否符合预期
                    if motion_map is not None:
                        expected_shape = (blur_images.size(0), 2, blur_images.size(2), blur_images.size(3))
                        if motion_map.shape != expected_shape:
                            print(f"警告: 批次 {i} 的运动图形状非预期: {motion_map.shape} vs {expected_shape}")
                            motion_map = None # 如果形状不匹配，设置为None以避免后续错误
                            print("将运动图设置为None以避免后续错误。")
                        elif torch.isnan(motion_map).any() or torch.isinf(motion_map).any():
                             print(f"警告: 批次 {i} 的运动核包含无效值，可能影响运动损失。")
                             # 可以选择跳过此批次，或者仅打印警告并继续 (当前选择打印警告)
                    else:
                         print(f"警告: 批次 {i} 的运动图为None，可能模型输出有问题。")


                    # 判别器训练：使用 detach() 阻止梯度回传到生成器
                    fake_output = model.discriminator(deblurred.detach())
                    # 确保目标标签在[0,1]范围内
                    d_fake_loss = nn.BCEWithLogitsLoss()(fake_output, torch.zeros_like(fake_output)) # 使用BCEWithLogitsLoss更稳定，目标设为0.0
                    # d_fake_loss = nn.BCELoss()(fake_output, torch.zeros_like(fake_output) * 0.1) # 原来的平滑标签

                    # 判别器总损失
                    d_loss = d_real_loss + d_fake_loss
                    d_loss.backward()
                    d_optimizer.step()

                    # 训练生成器
                    g_optimizer.zero_grad()

                    # 生成器训练：重新进行前向传播以计算生成器的梯度
                    # 不需要再次验证 deblurred 和 motion_map，因为上面的检查已经覆盖
                    deblurred, motion_map = model(blur_images, previous_frame)

                     # 再次验证模型输出 (谨慎起见，可以保留，但通常第一次检查足够)
                    # if torch.isnan(deblurred).any() or torch.isinf(deblurred).any():
                    #      print(f"警告: 批次 {i} 的去模糊图像输出包含无效值（生成器训练后），跳过此批次。")
                    #      continue

                    # 再次验证运动核 (谨慎起见，可以保留)
                    # if motion_map is not None and (torch.isnan(motion_map).any() or torch.isinf(motion_map).any()):
                    #     print(f"警告: 批次 {i} 的运动核包含无效值或为None（生成器训练后），可能影响运动损失。")


                    # 生成器希望判别器认为是真的，不 detach() 以便梯度回传到生成器
                    fake_output_gen = model.discriminator(deblurred)
                    g_gan_loss = nn.BCEWithLogitsLoss()(fake_output_gen, torch.ones_like(fake_output_gen)) # 目标设为1.0

                    # 计算去模糊相关的损失 (重构损失、运动损失等)
                    # 确保 deblurred 和 motion_map 在计算损失时有效
                    # DeblurLoss 应该能处理 motion_map 为 None 的情况
                    deblur_loss, loss_dict = criterion(deblurred, clear_images, motion_map, previous_frame)
                    # 最终检查 deblur_loss
                    if torch.isnan(deblur_loss).any() or torch.isinf(deblur_loss).any():
                         print(f"警告: 批次 {i} 的去模糊损失最终结果包含无效值，设置为零。")
                         deblur_loss = torch.tensor(0.0, device=device)
                         loss_dict = {} # 重置 loss_dict


                    # 生成器总损失
                    # GAN 损失和去模糊损失的权重可能需要调整
                    g_loss = g_gan_loss + deblur_loss

                    # 确保 g_loss 有效
                    if torch.isnan(g_loss).any() or torch.isinf(g_loss).any():
                         print(f"警告: 批次 {i} 的生成器总损失包含无效值，跳过梯度更新。")
                         continue # 跳过当前批次的梯度更新

                    g_loss.backward()
                    g_optimizer.step()

                    # 累加损失（用于打印和保存检查点，使用item()避免梯度累积）
                    current_epoch_g_loss += g_loss.item()
                    current_epoch_d_loss += d_loss.item()

                    # 更新前一帧
                    previous_frame = clear_images.detach().clone() # 使用 detach().clone() 获取副本


                    # 打印训练信息
                    if i % 10 == 0:
                         # 计算当前平均批次损失用于打印
                         avg_batch_g_loss = current_epoch_g_loss.item() / (i + 1)
                         avg_batch_d_loss = current_epoch_d_loss.item() / (i + 1)

                         # 安全获取 loss_dict 中的值
                         # 确保 loss_dict 中的键存在
                         recon_loss_item = loss_dict.get("recon_loss", 0)
                         motion_smooth_loss_item = loss_dict.get("motion_smooth_loss", 0)
                         motion_consistency_loss_item = loss_dict.get("motion_consistency_loss", 0)
                         motion_reg_item = loss_dict.get("motion_reg", 0)

                         # 确保打印的值是有限的
                         if not np.isfinite(recon_loss_item): recon_loss_item = 0
                         if not np.isfinite(motion_smooth_loss_item): motion_smooth_loss_item = 0
                         if not np.isfinite(motion_consistency_loss_item): motion_consistency_loss_item = 0
                         if not np.isfinite(motion_reg_item): motion_reg_item = 0


                         print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_dataloader)}], '
                              f'D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}, '
                              f'Recon_loss: {recon_loss_item:.4f}, '
                              f'Motion_smooth_loss: {motion_smooth_loss_item:.4f}, '
                              f'Motion_consistency_loss: {motion_consistency_loss_item:.4f}, '
                              f'Motion_reg: {motion_reg_item:.4f}, '
                              f'Avg_D_loss: {avg_batch_d_loss:.4f}, Avg_G_loss: {avg_batch_g_loss:.4f}')

                    # 保存可视化结果
                    if i % 100 == 0:  # 每100个batch保存一次
                        try:
                            # 创建可视化目录
                            vis_dir = os.path.join(output_dir, 'visualizations', f'epoch_{epoch+1}')
                            os.makedirs(vis_dir, exist_ok=True)

                            # 保存运动图 (现在是运动场 (N, 2, H, W))
                            # 获取 DeblurLoss 中保存的 last_motion_map
                            motion_map_vis = criterion.last_motion_map
                            if motion_map_vis is not None:
                                motion_map_vis = motion_map_vis[0].detach().cpu()  # 取第一个样本 (2, H, W)

                                # === 修改运动场可视化逻辑 ===
                                # 将两个通道分别保存为灰度图
                                if motion_map_vis.ndim == 3 and motion_map_vis.shape[0] == 2:
                                     # 水平运动 (通道 0)
                                     motion_x = motion_map_vis[0, :, :].unsqueeze(0) # (1, H, W)
                                     # 垂直运动 (通道 1)
                                     motion_y = motion_map_vis[1, :, :].unsqueeze(0) # (1, H, W)

                                     # 归一化到 [0, 1] 范围以便保存为灰度图
                                     # 假设运动范围在 [-max_motion, +max_motion]，这里简单归一化
                                     # 如果需要更精确的可视化，可以找到全局最大运动值进行归一化
                                     max_val = torch.max(torch.abs(motion_map_vis))
                                     if max_val == 0: max_val = 1e-6 # 避免除以零
                                     motion_x_norm = (motion_x + max_val) / (2 * max_val)
                                     motion_y_norm = (motion_y + max_val) / (2 * max_val)


                                     # 检查有效性后保存
                                     if not torch.isnan(motion_x_norm).any() and not torch.isinf(motion_x_norm).any():
                                          save_image(motion_x_norm,
                                                 os.path.join(vis_dir, f'motion_map_batch_{i}_x.png'))
                                          print(f"成功保存运动图 (水平): {os.path.join(vis_dir, f'motion_map_batch_{i}_x.png')}")
                                     else:
                                          print(f"警告: 批次 {i} 的运动图 (水平) 处理后包含无效值，未保存。")

                                     if not torch.isnan(motion_y_norm).any() and not torch.isinf(motion_y_norm).any():
                                          save_image(motion_y_norm,
                                                 os.path.join(vis_dir, f'motion_map_batch_{i}_y.png'))
                                          print(f"成功保存运动图 (垂直): {os.path.join(vis_dir, f'motion_map_batch_{i}_y.png')}")
                                     else:
                                          print(f"警告: 批次 {i} 的运动图 (垂直) 处理后包含无效值，未保存。")

                                else:
                                    print(f"警告: 批次 {i} 的运动图样本形状{motion_map_vis.shape}非预期 (2, H, W)，无法处理为可视化图像，跳过保存。")

                                # === 结束修改运动场可视化逻辑 ===
                            else:
                                print(f"警告: 批次 {i} 的运动图为None，未保存。")


                            # 保存生成结果对比图（包括模糊输入，去模糊输出，清晰目标，前一帧）
                            # 确保所有图像都已分离梯度并移到CPU
                            blur_img_vis = blur_images[0].detach().cpu()
                            deblurred_img_vis = deblurred[0].detach().cpu()
                            clear_img_vis = clear_images[0].cpu() # clear_images 不需要detach()因为它不参与梯度计算
                            # previous_frame 可能在第一个batch初始化为None，或者在保存时为None
                            previous_frame_vis = previous_frame[0].detach().cpu() if previous_frame is not None and previous_frame.size(0) > 0 else torch.zeros_like(blur_img_vis) # 如果previous_frame为None或空，使用零填充


                            # 归一化图像以便保存，值范围通常在[-1, 1]经过Normalize，保存前需要归一化到[0, 1]
                            blur_img_vis = (blur_img_vis + 1) / 2
                            deblurred_img_vis = (deblurred_img_vis + 1) / 2
                            clear_img_vis = (clear_img_vis + 1) / 2
                            previous_frame_vis = (previous_frame_vis + 1) / 2

                            # 确保所有图像尺寸相同以便拼接
                            min_h = min(blur_img_vis.size(1), deblurred_img_vis.size(1), clear_img_vis.size(1), previous_frame_vis.size(1))
                            min_w = min(blur_img_vis.size(2), deblurred_img_vis.size(2), clear_img_vis.size(2), previous_frame_vis.size(2))

                            # 裁剪到最小尺寸
                            blur_img_vis = blur_img_vis[:, :min_h, :min_w]
                            deblurred_img_vis = deblurred_img_vis[:, :min_h, :min_w]
                            clear_img_vis = clear_img_vis[:, :min_h, :min_w]
                            previous_frame_vis = previous_frame_vis[:, :min_h, :min_w]


                            comparison = torch.cat([
                                blur_img_vis,
                                deblurred_img_vis,
                                clear_img_vis,
                                previous_frame_vis
                            ], dim=2) # 在宽度方向拼接
                            save_image(comparison,
                                     os.path.join(vis_dir, f'comparison_batch_{i}.png'))
                            print(f"成功保存对比图: {os.path.join(vis_dir, f'comparison_batch_{i}.png')}")


                        except Exception as e:
                            print(f"保存可视化结果时出错: {str(e)}")
                            # 继续训练，不要因为保存失败而中断
                            import traceback
                            traceback.print_exc() # 打印详细错误信息
                            continue

                    # 每100个batch保存一次断点
                    if i % 100 == 0:
                        try:
                            # 保存当前训练状态，包括 current_batch 和 previous_frame
                            training_state_checkpoint = {
                                'epoch': epoch,
                                'current_batch': i, # 保存当前批次索引
                                # 保存当前epoch到目前为止的累积损失
                                'current_epoch_g_loss': current_epoch_g_loss.item(),
                                'current_epoch_d_loss': current_epoch_d_loss.item(),
                                # 保存当前 batch 更新后的 previous_frame
                                # previous_frame 是图像张量 [N, C, H, W]，保存到CPU
                                'previous_frame': previous_frame.detach().cpu() if previous_frame is not None else None
                            }

                            # 计算保存检查点时的平均损失
                            avg_g_loss_checkpoint = current_epoch_g_loss.item() / (i + 1) if (i + 1) > 0 else 0.0
                            avg_d_loss_checkpoint = current_epoch_d_loss.item() / (i + 1) if (i + 1) > 0 else 0.0


                            # 调用 save_checkpoint 并传递 training_state
                            # save_checkpoint 函数内部需要处理 training_state 的保存
                            save_checkpoint(epoch, model, g_optimizer, d_optimizer,
                                         avg_g_loss_checkpoint, # 保存平均损失到检查点信息
                                         avg_d_loss_checkpoint,
                                         None, # metrics 在 epoch 结束时计算和保存
                                         output_dir,
                                         train_dataloader,
                                         training_state=training_state_checkpoint # 传递 training_state
                                         )
                        except Exception as e:
                            print(f"保存检查点时出错: {str(e)}")
                            import traceback
                            traceback.print_exc() # 打印详细错误信息
                            continue


                except Exception as e:
                    print(f"处理批次 {i} 时出现未捕获错误: {str(e)}")
                    import traceback
                    traceback.print_exc() # 打印详细错误信息
                    # 可以在这里添加保存临时断点的逻辑，以便从错误发生的地方附近恢复
                    # 例如，保存当前的 epoch 和 batch 索引
                    # 可以在这里考虑保存一个临时的检查点
                    try:
                        print(f"尝试在批次 {i} 发生错误时保存临时检查点...")
                        training_state_error = {
                            'epoch': epoch,
                            'current_batch': i,
                            'current_epoch_g_loss': current_epoch_g_loss.item(),
                            'current_epoch_d_loss': current_epoch_d_loss.item(),
                            'previous_frame': previous_frame.detach().cpu() if previous_frame is not None else None
                        }
                        # 保存到特定的临时文件
                        temp_checkpoint_path = f"{output_dir}/checkpoint_error_epoch_{epoch}_batch_{i}.pth"
                        torch.save({
                            'epoch': epoch,
                            'generator_state_dict': model.generator.state_dict(),
                            'discriminator_state_dict': model.discriminator.state_dict(),
                            'motion_estimator_state_dict': model.motion_estimator.state_dict() if hasattr(model, 'motion_estimator') else None,
                            'g_optimizer_state_dict': g_optimizer.state_dict(),
                            'd_optimizer_state_dict': d_optimizer.state_dict(),
                            'training_state': training_state_error
                        }, temp_checkpoint_path)
                        print(f"临时检查点已保存: {temp_checkpoint_path}")
                    except Exception as e_save_error:
                        print(f"保存临时检查点失败: {str(e_save_error)}")
                        traceback.print_exc()

                    continue # 继续下一个批次，或者可以选择 break 终止训练

            # 每个epoch进行验证
            try:
                # 在验证前，将 previous_frame 设置为 None，因为验证集通常是独立的图像对
                # 如果您的 validate_model 函数需要 previous_frame，请修改其逻辑
                avg_psnr, avg_ssim = validate_model(model, val_dataloader, device)
                print(f"Epoch [{epoch + 1}/{num_epochs}], Validation PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")

                # 计算平均损失 (使用epoch结束时的总累积损失)
                # 确保 train_dataloader 不为空，避免除以零
                avg_g_loss_epoch = current_epoch_g_loss.item() / len(train_dataloader) if len(train_dataloader) > 0 else 0.0
                avg_d_loss_epoch = current_epoch_d_loss.item() / len(train_dataloader) if len(train_dataloader) > 0 else 0.0


                print(f"Epoch [{epoch + 1}/{num_epochs}], Train G_Loss: {avg_g_loss_epoch:.4f}, Train D_Loss: {avg_d_loss_epoch:.4f}")

                # 保存检查点 (使用epoch结束时的平均损失和验证指标)
                metrics = {
                    'psnr': avg_psnr,
                    'ssim': avg_ssim,
                    'g_loss_train_epoch': avg_g_loss_epoch, # 记录训练集的epoch平均损失
                    'd_loss_train_epoch': avg_d_loss_epoch
                }

                # 在 epoch 结束保存检查点时，previous_frame 设置为 None
                training_state_epoch_end = {'previous_frame': None}


                save_checkpoint(epoch + 1, model, g_optimizer, d_optimizer,
                              avg_g_loss_epoch,
                              avg_d_loss_epoch,
                              metrics,
                              output_dir,
                              train_dataloader,
                              training_state=training_state_epoch_end # epoch 结束时 previous_frame 不跨 epoch 保留
                             )

                # 将结果记录到文件
                with open(f"{output_dir}/training_log.txt", "a") as f:
                    f.write(f"Epoch {epoch + 1}: PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}, "
                           f"G_Loss_train={avg_g_loss_epoch:.4f}, D_Loss_train={avg_d_loss_epoch:.4f}\n")

            except Exception as e:
                print(f"验证或epoch结束处理时出错: {str(e)}")
                import traceback
                traceback.print_exc() # 打印详细错误信息
                # 即使验证失败，也尝试保存检查点，但注意metrics可能不准确
                try:
                     # 保存一个不带metrics的检查点
                     training_state_epoch_end = {'previous_frame': None}
                     avg_g_loss_epoch = current_epoch_g_loss.item() / len(train_dataloader) if len(train_dataloader) > 0 else 0.0
                     avg_d_loss_epoch = current_epoch_d_loss.item() / len(train_dataloader) if len(train_dataloader) > 0 else 0.0
                     save_checkpoint(epoch + 1, model, g_optimizer, d_optimizer,
                                   avg_g_loss_epoch, avg_d_loss_epoch,
                                   None, output_dir, train_dataloader, training_state=training_state_epoch_end)
                except Exception as e_save:
                     print(f"错误处理中保存检查点失败: {str(e_save)}")
                     import traceback
                     traceback.print_exc()
                # continue # 确保训练不会因验证失败而中断，但这里应该 break 或 raise 以指示epoch失败
                # 如果验证失败是一个严重问题，可能需要终止训练
                print("验证失败，终止当前epoch。")
                break # 跳出当前epoch的循环，进入下一个epoch

        # 保存最终模型
        torch.save(model.state_dict(), f"{output_dir}/innovative_deblur_final.pth")
        print("\n训练完成！")

    except Exception as e:
        print(f"训练过程中出现顶层错误: {str(e)}")
        import traceback
        traceback.print_exc() # 打印详细错误信息
        # 可以在顶层异常处保存一个最终的检查点或错误信息
        # 顶层错误通常是致命的，直接raise
        raise e # 重新抛出异常，以便外部知道训练失败

if __name__ == '__main__':
    train_innovative_model() 