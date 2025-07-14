import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class DynamicKernelPredictor(nn.Module):
    """动态卷积核预测器"""
    def __init__(self, in_channels=3, kernel_size=5):
        super().__init__()
        self.kernel_size = kernel_size
        
        # 特征提取
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 动态卷积核生成
        self.kernel_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, kernel_size * kernel_size, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        features = self.feature_extractor(x)
        kernels = self.kernel_generator(features)
        return kernels.view(-1, 1, self.kernel_size, self.kernel_size)

class MultiScaleSpatialAttention(nn.Module):
    """多尺度空间注意力模块"""
    def __init__(self, channels):
        super().__init__()
        self.scales = [1, 2, 4]  # 多尺度因子
        
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels // 8, 1, 1),
            nn.Sigmoid()
        )
        
        # 多尺度特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * len(self.scales), channels, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
    def forward(self, x):
        multi_scale_features = []
        
        # 多尺度特征提取
        for scale in self.scales:
            if scale == 1:
                resized = x
            else:
                resized = F.interpolate(x, scale_factor=1/scale, mode='bilinear', align_corners=False)
            
            # 应用空间注意力
            attention = self.spatial_attention(resized)
            if scale != 1:
                attention = F.interpolate(attention, size=x.shape[2:], mode='bilinear', align_corners=False)
            
            multi_scale_features.append(attention * x)
        
        # 特征融合
        return self.fusion(torch.cat(multi_scale_features, dim=1))

class TemporalAttention(nn.Module):
    """时间注意力模块"""
    def __init__(self, channels):
        super().__init__()
        self.temporal_attention = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, current, previous):
        # 计算时间注意力
        attention = self.temporal_attention(torch.cat([current, previous], dim=1))
        return attention * current

class AdaptiveFeatureFusion(nn.Module):
    """自适应特征融合模块"""
    def __init__(self, channels):
        super().__init__()
        self.fusion_weights = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, 2, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x1, x2):
        # 计算融合权重
        weights = self.fusion_weights(torch.cat([x1, x2], dim=1))
        return weights[:, 0:1] * x1 + weights[:, 1:2] * x2

class InnovativeDeblurNet(nn.Module):
    """创新去模糊网络"""
    def __init__(self, in_channels=3):
        super().__init__()
        
        # 动态卷积核预测器
        self.kernel_predictor = DynamicKernelPredictor(in_channels)
        
        # 特征提取
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 多尺度空间注意力
        self.spatial_attention = MultiScaleSpatialAttention(256)
        
        # 时间注意力
        self.temporal_attention = TemporalAttention(256)
        
        # 自适应特征融合
        self.feature_fusion = AdaptiveFeatureFusion(256)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, in_channels, 3, padding=1)
        )
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, previous_state=None):
        # 预测动态卷积核
        kernels = self.kernel_predictor(x)
        
        # 应用动态卷积
        b, c, h, w = x.shape
        x = F.conv2d(x.view(1, -1, h, w), 
                    kernels.view(-1, 1, kernels.size(2), kernels.size(3)),
                    groups=b).view(b, c, h, w)
        
        # 特征提取
        features = self.encoder(x)
        
        # 多尺度空间注意力
        spatial_features = self.spatial_attention(features)
        
        # 时间注意力
        if previous_state is not None:
            temporal_features = self.temporal_attention(spatial_features, previous_state)
        else:
            temporal_features = spatial_features
        
        # 自适应特征融合
        fused_features = self.feature_fusion(spatial_features, temporal_features)
        
        # 解码得到清晰图像
        output = self.decoder(fused_features)
        
        return output, fused_features  # 返回当前状态用于下一帧

class InnovativeDeblurGAN(nn.Module):
    """创新去模糊GAN网络"""
    def __init__(self):
        super().__init__()
        self.generator = InnovativeDeblurNet()
        
        # 判别器
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
            nn.Conv2d(512, 1, 4, stride=1, padding=1)
        ) 