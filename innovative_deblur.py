import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image
from torchvision.models import VGG16_Weights

class MotionAwareAttention(nn.Module):
    """运动感知注意力模块 - 创新点1：动态运动轨迹感知"""
    def __init__(self, channels):
        super(MotionAwareAttention, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels//4, 1)
        self.conv2 = nn.Conv2d(channels//4, channels, 1)
        self.motion_conv = nn.Conv2d(2, channels//4, 3, padding=1)
        
        # 添加注意力可视化支持
        self.last_spatial_att = None
        self.last_motion_att = None
        self.last_fused_att = None
        
    def forward(self, x, motion_map):
        # 提取运动特征
        motion_features = self.motion_conv(motion_map)
        
        # 空间注意力
        spatial_att = torch.sigmoid(self.conv2(self.conv1(x)))
        
        # 运动注意力
        motion_att = torch.sigmoid(self.conv2(motion_features))
        
        # 融合注意力
        fused_att = spatial_att * motion_att
        
        # 保存注意力图用于可视化
        self.last_spatial_att = spatial_att.detach()
        self.last_motion_att = motion_att.detach()
        self.last_fused_att = fused_att.detach()
        
        return x * fused_att

class AdaptiveDeblurBlock(nn.Module):
    """自适应去模糊块 - 创新点2：动态模糊核估计"""
    def __init__(self, channels):
        super(AdaptiveDeblurBlock, self).__init__()
        
        # 动态卷积核估计器
        self.kernel_estimator = nn.Sequential(
            nn.Conv2d(channels, channels//2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//2, channels//4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//4, 9, 1)  # 输出9个通道，对应3x3卷积核
        )
        
        # 特征提取和融合
        self.feature_extract = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
        
    def forward(self, x):
        b, c, h, w = x.size()
        
        # 估计局部模糊核
        kernel = self.kernel_estimator(x)  # [B, 9, H, W]
        kernel = kernel.view(b, 1, 9, h, w)  # [B, 1, 9, H, W]
        kernel = F.softmax(kernel, dim=2)  # 在9个元素上做softmax
        
        # 使用unfold操作提取局部区域
        x_unfold = F.unfold(x, kernel_size=3, padding=1)  # [B, C*9, H*W]
        x_unfold = x_unfold.view(b, c, 9, h, w)  # [B, C, 9, H, W]
        
        # 应用动态卷积核
        out = torch.sum(x_unfold * kernel, dim=2)  # [B, C, H, W]
        
        # 特征提取和融合
        features = self.feature_extract(out)
        
        # 保存中间结果用于可视化
        self.last_kernel = kernel.detach()
        self.last_attention = None  # 将在MotionAwareAttention中设置
        
        return features

class TemporalCoherenceModule(nn.Module):
    """时序一致性模块 - 创新点3：多帧时序信息融合"""
    def __init__(self, channels):
        super(TemporalCoherenceModule, self).__init__()
        self.conv1 = nn.Conv2d(channels*2, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.attention = nn.Sequential(
            nn.Conv2d(channels*2, channels, 1),
            nn.Sigmoid()
        )
        
        # 添加时序一致性可视化支持
        self.last_temporal_att = None
        
    def forward(self, current, previous):
        # 确保previous的批次大小与current匹配
        if previous is not None:
            if previous.size(0) != current.size(0):
                previous = previous[:current.size(0)]
            
            # 确保previous与current在同一设备上
            if previous.device != current.device:
                previous = previous.to(current.device)
            
            # 特征融合
            combined = torch.cat([current, previous], dim=1)
            temporal_att = self.attention(combined)
            
            # 保存时序注意力用于可视化
            self.last_temporal_att = temporal_att.detach()
            
            # 时序一致性处理
            out = self.conv1(combined)
            out = F.relu(out, inplace=True)
            out = self.conv2(out)
            
            return out * temporal_att + current
        else:
            return current

class AdaptiveMotionPerception(nn.Module):
    """自适应运动感知模块 - 创新点6：动态运动强度感知"""
    def __init__(self, channels):
        super(AdaptiveMotionPerception, self).__init__()
        self.motion_conv = nn.Conv2d(2, channels, 3, padding=1)
        self.intensity_estimator = nn.Sequential(
            nn.Conv2d(channels, channels//2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//2, 1, 1),
            nn.Sigmoid()
        )
        self.feature_enhancer = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
        
    def forward(self, x, motion_map):
        # 提取运动特征
        motion_features = self.motion_conv(motion_map)
        
        # 估计运动强度
        intensity = self.intensity_estimator(motion_features)
        
        # 增强特征
        enhanced = self.feature_enhancer(x)
        
        # 自适应融合
        return x + enhanced * intensity

class DynamicFeatureFusion(nn.Module):
    """动态特征融合模块 - 创新点7：多尺度特征自适应融合"""
    def __init__(self, channels):
        super(DynamicFeatureFusion, self).__init__()
        self.scale_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//4, channels, 1),
            nn.Sigmoid()
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels*2, channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
        
    def forward(self, x1, x2):
        # 计算注意力权重
        att = self.scale_attention(x1)
        
        # 特征融合
        fused = torch.cat([x1 * att, x2 * (1-att)], dim=1)
        return self.fusion_conv(fused)

class InnovativeDeblurGenerator(nn.Module):
    """创新去模糊生成器"""
    def __init__(self):
        super(InnovativeDeblurGenerator, self).__init__()
        
        # 初始特征提取
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 运动感知注意力模块
        self.motion_attention = MotionAwareAttention(64)
        
        # 自适应运动感知模块
        self.motion_perception = AdaptiveMotionPerception(64)
        
        # 自适应去模糊块
        self.deblur_blocks = nn.ModuleList([
            AdaptiveDeblurBlock(64) for _ in range(4)
        ])
        
        # 时序一致性模块
        self.temporal_module = TemporalCoherenceModule(64)
        
        # 动态特征融合模块
        self.feature_fusion = DynamicFeatureFusion(64)
        
        # 输出层 - 修改为更稳定的结构
        self.output = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh()  # 输出范围[-1,1]
        )
        
    def forward(self, x, motion_map, previous_frame=None):
        # 特征提取
        features = self.initial(x)
        
        # 运动感知注意力
        features = self.motion_attention(features, motion_map)
        
        # 自适应运动感知
        features = self.motion_perception(features, motion_map)
        
        # 自适应去模糊
        deblurred_features = features
        for block in self.deblur_blocks:
            deblurred_features = block(deblurred_features)
        
        # 时序一致性处理
        if previous_frame is not None:
            prev_features = self.initial(previous_frame)
            temporal_features = self.temporal_module(deblurred_features, prev_features)
            # 动态特征融合
            features = self.feature_fusion(deblurred_features, temporal_features)
        else:
            features = deblurred_features
        
        # 输出清晰图像
        output = self.output(features)
        
        # 确保输出范围在[-1,1]之间
        output = torch.clamp(output, -1, 1)
        
        return output

class MotionEstimator(nn.Module):
    """运动估计器 - 创新点4：精确运动轨迹估计"""
    def __init__(self):
        super(MotionEstimator, self).__init__()
        
        # 输入特征提取
        self.input_conv = nn.Sequential(
            nn.Conv2d(6, 64, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),  # 使用LeakyReLU
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 下采样路径
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 中间特征处理
        self.middle = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 上采样路径
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 输出层
        self.output = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 2, 3, padding=1),
            nn.Tanh()  # 使用Tanh，输出范围[-1,1]
        )
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, current, previous):
        if previous is not None:
            if previous.size(0) != current.size(0):
                previous = previous[:current.size(0)]
            if previous.device != current.device:
                previous = previous.to(current.device)
            x = torch.cat([current, previous], dim=1)
        else:
            x = torch.cat([current, torch.zeros_like(current)], dim=1)
        
        # 特征提取
        x = self.input_conv(x)
        
        # 下采样路径
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        
        # 中间处理
        m = self.middle(d3)
        
        # 上采样路径（带跳跃连接）
        u1 = self.up1(m)
        u1 = u1 + d2  # 跳跃连接
        u2 = self.up2(u1)
        u2 = u2 + d1  # 跳跃连接
        
        # 输出
        motion_map = self.output(u2)
        
        return motion_map

class PerceptualLoss(nn.Module):
    """感知损失 - 使用预训练的VGG网络提取特征"""
    def __init__(self):
        super().__init__()
        # 加载预训练的VGG16模型 using weights
        vgg = models.vgg16(weights=VGG16_Weights.DEFAULT) # Use DEFAULT weights
        # 只使用前16层（到relu3_3）
        self.feature_extractor = nn.Sequential(*list(vgg.features)[:16])
        # 冻结VGG参数
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        # 将模型设置为评估模式
        self.feature_extractor.eval()
    
    def forward(self, x, y):
        # 确保输入在正确的设备上
        device = x.device
        # Move feature_extractor to the correct device
        self.feature_extractor = self.feature_extractor.to(device)
        
        # 提取特征
        with torch.no_grad():
            x_features = self.feature_extractor(x)
            y_features = self.feature_extractor(y)
        
        # 计算特征之间的L1损失
        loss = F.l1_loss(x_features, y_features)
        return loss

class DeblurLoss(nn.Module):
    """创新损失函数 - 创新点5：多目标联合优化"""
    def __init__(self, recon_weight=1.0, motion_smooth_weight=0.2, 
                 motion_consistency_weight=0.1, motion_reg_weight=0.05,
                 perceptual_weight=0.5):
        super().__init__()
        self.recon_weight = recon_weight
        self.motion_smooth_weight = motion_smooth_weight
        self.motion_consistency_weight = motion_consistency_weight
        self.motion_reg_weight = motion_reg_weight
        self.perceptual_weight = perceptual_weight
        
        # 初始化损失函数
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.perceptual_loss = PerceptualLoss()
        
        # 用于可视化的运动图
        self.last_motion_map = None

    def forward(self, deblurred, target, motion_map, previous_frame=None, previous_motion_map=None):
        # 保存运动图用于可视化
        self.last_motion_map = motion_map.detach().clone()
        
        # 重建损失
        recon_loss = self.l1_loss(deblurred, target)
        
        # 感知损失
        perceptual_loss = self.perceptual_loss(deblurred, target)
        
        # 运动平滑损失
        if motion_map is not None:
            # 计算水平和垂直方向的梯度
            motion_grad_x = torch.abs(motion_map[:, :, :, :-1] - motion_map[:, :, :, 1:])
            motion_grad_y = torch.abs(motion_map[:, :, :-1, :] - motion_map[:, :, 1:, :])
            
            # 添加中心差分项
            motion_grad_x_center = torch.abs(motion_map[:, :, :, 1:-1] - 
                                           (motion_map[:, :, :, :-2] + motion_map[:, :, :, 2:]) / 2)
            motion_grad_y_center = torch.abs(motion_map[:, :, 1:-1, :] - 
                                           (motion_map[:, :, :-2, :] + motion_map[:, :, 2:, :]) / 2)
            
            motion_smooth_loss = (torch.mean(motion_grad_x) + torch.mean(motion_grad_y) +
                                torch.mean(motion_grad_x_center) + torch.mean(motion_grad_y_center))
        else:
            motion_smooth_loss = torch.tensor(0.0, device=deblurred.device)
        
        # 运动一致性损失
        if motion_map is not None and previous_motion_map is not None:
            motion_consistency_loss = self.l1_loss(motion_map, previous_motion_map)
        else:
            motion_consistency_loss = torch.tensor(0.0, device=deblurred.device)
        
        # 运动正则化损失
        if motion_map is not None:
            # 鼓励运动图的均值为0
            motion_mean = torch.mean(motion_map)
            # 鼓励运动图的标准差在合理范围内
            motion_std = torch.std(motion_map)
            # 鼓励运动图的稀疏性
            motion_sparsity = torch.mean(torch.abs(motion_map))
            
            motion_reg = (torch.abs(motion_mean) + 
                         torch.abs(motion_std - 0.1) + 
                         motion_sparsity)
        else:
            motion_reg = torch.tensor(0.0, device=deblurred.device)
        
        # 总损失
        total_loss = (self.recon_weight * recon_loss +
                     self.perceptual_weight * perceptual_loss +
                     self.motion_smooth_weight * motion_smooth_loss +
                     self.motion_consistency_weight * motion_consistency_loss +
                     self.motion_reg_weight * motion_reg)
        
        # 返回损失字典
        loss_dict = {
            "recon_loss": recon_loss.item(),
            "perceptual_loss": perceptual_loss.item(),
            "motion_smooth_loss": motion_smooth_loss.item(),
            "motion_consistency_loss": motion_consistency_loss.item(),
            "motion_reg": motion_reg.item()
        }
        
        return total_loss, loss_dict 

class Discriminator(nn.Module):
    """判别器网络"""
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # 输入: 3 x 128 x 128
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64 x 64 x 64
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 128 x 32 x 32
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 256 x 16 x 16
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 512 x 8 x 8
            nn.Conv2d(512, 1, 4, stride=1, padding=0),
            # 输出: 1 x 5 x 5
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.main(x)

class InnovativeDeblurGAN(nn.Module):
    """创新去模糊GAN"""
    def __init__(self):
        super().__init__()
        self.motion_estimator = MotionEstimator()
        self.generator = InnovativeDeblurGenerator()
        self.discriminator = Discriminator()

    def forward(self, current_frame, previous_frame=None):
        # 确保输入在同一设备上
        if previous_frame is not None:
            previous_frame = previous_frame.to(current_frame.device)
        
        # 估计运动
        if previous_frame is not None:
            motion_map = self.motion_estimator(current_frame, previous_frame)
        else:
            # 如果没有前一帧，创建零运动图
            motion_map = torch.zeros((current_frame.size(0), 2, current_frame.size(2), current_frame.size(3)), 
                                   device=current_frame.device)
        
        # 生成去模糊图像
        deblurred = self.generator(current_frame, motion_map, previous_frame)
        
        return deblurred, motion_map 