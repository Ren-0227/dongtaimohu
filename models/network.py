import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads=8, window_size=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.window_size = window_size

    def forward(self, x):
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        
        # Window-based attention
        x = self.norm1(x)
        x = x + self.attn(x, x, x)[0]
        x = x + self.mlp(self.norm2(x))
        
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x

class SFTLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.scale = nn.Conv2d(channels, channels, 1)
        self.shift = nn.Conv2d(channels, channels, 1)

    def forward(self, x, kernel_params):
        scale = self.scale(kernel_params)
        shift = self.shift(kernel_params)
        return x * scale + shift

class KernelPredictor(nn.Module):
    def __init__(self, in_channels=3, num_kernel_types=12):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(64, num_kernel_types * 4)  # 4 parameters per kernel type
        
    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        kernel_params = self.fc(features)
        return kernel_params

class AdaptiveDeblurNet(nn.Module):
    def __init__(self, in_channels=3, num_kernel_types=12):
        super().__init__()
        self.kernel_predictor = KernelPredictor(in_channels, num_kernel_types)
        
        # Swin-CNN混合架构
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            SwinBlock(64),
            nn.Conv2d(64, 128, 3, padding=1),
            SwinBlock(128)
        )
        
        # SFT层
        self.sft1 = SFTLayer(64)
        self.sft2 = SFTLayer(128)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, in_channels, 3, padding=1)
        )
        
        # LSTM用于迭代修正
        self.lstm = nn.LSTM(input_size=num_kernel_types * 4,
                           hidden_size=128,
                           num_layers=2,
                           batch_first=True)
        
    def forward(self, x, num_iterations=3):
        # 初始模糊核预测
        kernel_params = self.kernel_predictor(x)
        
        # LSTM迭代修正
        kernel_params = kernel_params.unsqueeze(1).repeat(1, num_iterations, 1)
        kernel_params, _ = self.lstm(kernel_params)
        
        # 使用最终修正后的参数
        final_params = kernel_params[:, -1]
        
        # 特征提取和SFT
        feat1 = self.encoder[:2](x)
        feat1 = self.sft1(feat1, final_params.view(-1, 48, 1, 1))
        
        feat2 = self.encoder[2:](feat1)
        feat2 = self.sft2(feat2, final_params.view(-1, 48, 1, 1))
        
        # 解码得到清晰图像
        out = self.decoder(feat2)
        return out, final_params 