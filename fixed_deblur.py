import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image

class FixedAttention(nn.Module):
    """固定注意力模块"""
    def __init__(self, channels):
        super(FixedAttention, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels//8, 1)
        self.conv2 = nn.Conv2d(channels//8, channels, 1)
        
    def forward(self, x):
        # 空间注意力
        spatial_att = torch.sigmoid(self.conv2(self.conv1(x)))
        return x * spatial_att

class FixedDeblurBlock(nn.Module):
    """固定去模糊块"""
    def __init__(self, channels):
        super(FixedDeblurBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        
    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        return out + x

class FixedDeblurGenerator(nn.Module):
    """固定去模糊生成器"""
    def __init__(self):
        super(FixedDeblurGenerator, self).__init__()
        
        # 初始特征提取
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 固定注意力模块
        self.attention = FixedAttention(64)
        
        # 固定去模糊块
        self.deblur_blocks = nn.ModuleList([
            FixedDeblurBlock(64) for _ in range(4)
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
        
        # 固定注意力
        features = self.attention(features)
        
        # 固定去模糊
        for block in self.deblur_blocks:
            features = block(features)
        
        # 输出清晰图像
        return self.output(features)

class FixedDeblurGAN(nn.Module):
    """固定去模糊GAN"""
    def __init__(self):
        super(FixedDeblurGAN, self).__init__()
        self.generator = FixedDeblurGenerator()
        self.discriminator = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, stride=1, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        deblurred = self.generator(x)
        return deblurred, None  # 返回None作为motion_map以保持接口一致

# 损失函数
class FixedDeblurLoss(nn.Module):
    """固定损失函数"""
    def __init__(self):
        super(FixedDeblurLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        
    def forward(self, deblurred, target):
        # 重建损失
        recon_loss = self.l1_loss(deblurred, target)
        
        return recon_loss, {
            'recon_loss': recon_loss.item(),
            'motion_smooth_loss': 0.0,
            'temporal_loss': 0.0
        } 