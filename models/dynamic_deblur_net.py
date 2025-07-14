import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.utils.checkpoint import checkpoint

class ProgressiveFeatureCompression(nn.Module):
    """渐进式特征压缩模块"""
    def __init__(self, in_channels, compression_ratio=0.125):  # 更激进的压缩比
        super().__init__()
        self.compression_ratio = compression_ratio
        self.conv = nn.Conv2d(in_channels, int(in_channels * compression_ratio), 1)
        
    def forward(self, x):
        return self.conv(x)

class EfficientMotionEncoder(nn.Module):
    """高效运动编码器"""
    def __init__(self, in_channels=3, base_channels=8):  # 进一步减少基础通道数
        super().__init__()
        # 使用深度可分离卷积减少参数量
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, base_channels, 1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1, groups=base_channels),
            nn.Conv2d(base_channels, base_channels*2, 1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*2, 3, padding=1, groups=base_channels*2),
            nn.Conv2d(base_channels*2, base_channels*4, 1)
        )
        
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(base_channels*4, base_channels*4 // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*4 // 16, base_channels*4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        
        # 应用通道注意力
        att = self.channel_attention(x3)
        return x3 * att

class MemoryEfficientAttention(nn.Module):
    """内存高效注意力模块"""
    def __init__(self, dim, num_heads=2):  # 减少注意力头数
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 使用线性投影减少计算量
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        B, T, C = x.shape
        
        # 分块处理以减少内存使用
        chunk_size = min(T, 1024)  # 限制每个块的大小
        num_chunks = (T + chunk_size - 1) // chunk_size
        output = []
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, T)
            
            # 获取当前块的查询
            x_chunk = x[:, start_idx:end_idx]
            
            # 线性投影
            q = self.q_proj(x_chunk).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x_chunk).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x_chunk).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
            
            # 计算注意力
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            
            # 应用注意力
            chunk_output = (attn @ v).transpose(1, 2).reshape(B, -1, C)
            chunk_output = self.out_proj(chunk_output)
            output.append(chunk_output)
            
            # 清理内存
            del q, k, v, attn, chunk_output
            torch.cuda.empty_cache()
        
        # 合并所有块的输出
        return torch.cat(output, dim=1)

class DynamicDeblurNet(nn.Module):
    """改进的动态去模糊网络"""
    def __init__(self, in_channels=3, num_frames=5):
        super().__init__()
        # 运动编码器
        self.motion_encoder = EfficientMotionEncoder(in_channels)
        
        # 特征压缩
        self.feature_compression = ProgressiveFeatureCompression(32, compression_ratio=0.125)  # 减少输入通道数
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(4, 4, 3, padding=1, groups=4),
            nn.Conv2d(4, 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 4, 3, padding=1, groups=4),
            nn.Conv2d(4, 4, 1),
            nn.ReLU(inplace=True)
        )
        
        # 时间注意力
        self.temporal_attention = MemoryEfficientAttention(4)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Conv2d(4, 4, 3, padding=1, groups=4),
            nn.Conv2d(4, 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, in_channels, 3, padding=1)
        )
        
    def forward(self, x, prev_states=None):
        # 提取运动特征
        motion_feat = checkpoint(self.motion_encoder, x, use_reentrant=False)
        
        # 特征压缩
        compressed_feat = self.feature_compression(motion_feat)
        
        # 特征融合
        fused_feat = checkpoint(self.fusion, compressed_feat, use_reentrant=False)
        
        # 时间注意力
        B, C, H, W = fused_feat.shape
        fused_feat = rearrange(fused_feat, 'b c h w -> b (h w) c')
        fused_feat = checkpoint(self.temporal_attention, fused_feat, use_reentrant=False)
        fused_feat = rearrange(fused_feat, 'b (h w) c -> b c h w', h=H, w=W)
        
        # 解码得到清晰图像
        out = checkpoint(self.decoder, fused_feat, use_reentrant=False)
        
        return out, None, None  # 简化版本，不返回kernel_params和states 