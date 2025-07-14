import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image

class SingleScaleAttention(nn.Module):
    """单尺度注意力模块"""
    def __init__(self, channels):
        super(SingleScaleAttention, self).__init__()
        # 单一尺度的注意力计算
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels//8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//8, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 计算注意力权重
        attention = self.attention(x)
        # 应用注意力
        return x * attention

class SingleScaleDeblurBlock(nn.Module):
    """单尺度去模糊块"""
    def __init__(self, channels):
        super(SingleScaleDeblurBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.attention = SingleScaleAttention(channels)
        
    def forward(self, x):
        # 特征提取
        out = self.conv1(x)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        # 应用单尺度注意力
        out = self.attention(out)
        return out + x

class SingleScaleDeblurGenerator(nn.Module):
    """单尺度去模糊生成器"""
    def __init__(self):
        super(SingleScaleDeblurGenerator, self).__init__()
        
        # 初始特征提取
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 单尺度注意力模块
        self.attention = SingleScaleAttention(64)
        
        # 单尺度去模糊块
        self.deblur_blocks = nn.ModuleList([
            SingleScaleDeblurBlock(64) for _ in range(4)
        ])
        
        # 输出层
        self.output = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        # 特征提取
        features = self.initial(x)
        
        # 单尺度注意力
        features = self.attention(features)
        
        # 单尺度去模糊
        for block in self.deblur_blocks:
            features = block(features)
        
        # 输出清晰图像
        return self.output(features)

class SingleScaleDeblurGAN(nn.Module):
    """单尺度去模糊GAN"""
    def __init__(self):
        super(SingleScaleDeblurGAN, self).__init__()
        self.generator = SingleScaleDeblurGenerator()
        self.discriminator = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, stride=1, padding=0),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        deblurred = self.generator(x)
        return deblurred, None  # 返回None作为motion_map以保持接口一致

# 损失函数
class SingleScaleDeblurLoss(nn.Module):
    """单尺度损失函数"""
    def __init__(self):
        super(SingleScaleDeblurLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        
    def forward(self, deblurred, target):
        # 检查输入尺寸
        if deblurred.size() != target.size():
            raise ValueError(f"Size mismatch: deblurred {deblurred.size()} vs target {target.size()}")
            
        # 检查数值范围
        if torch.isnan(deblurred).any() or torch.isnan(target).any():
            print("Warning: NaN detected in loss calculation")
            return torch.tensor(0.0, device=deblurred.device), {
                'recon_loss': 0.0,
                'motion_smooth_loss': 0.0,
                'temporal_loss': 0.0
            }
        
        # 重建损失
        recon_loss = self.l1_loss(deblurred, target)
        
        return recon_loss, {
            'recon_loss': recon_loss.item(),
            'motion_smooth_loss': 0.0,
            'temporal_loss': 0.0
        } 