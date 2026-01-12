# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
额外的自定义模块：SPDConv 和 EMA
用于小目标检测的创新算子
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("SPDConv", "EMA")


class SPDConv(nn.Module):
    """
    SPD-Conv: Space-to-Depth Convolution
    实现空间转深度的无损下采样，将 2x2 空间区域转换为 4 倍深度通道。
    
    适用于小目标检测，避免传统下采样导致的信息丢失。
    
    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        kernel_size (int): 卷积核大小，默认 3
        stride (int): 步长，默认 1
        padding (int): 填充，默认 1
        
    Examples:
        >>> import torch
        >>> m = SPDConv(64, 128)
        >>> x = torch.randn(1, 64, 64, 64)
        >>> y = m(x)
        >>> print(y.shape)  # torch.Size([1, 128, 32, 32])
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels * 4,  # SPD 后通道数变为 4 倍
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x):
        """
        前向传播：执行空间到深度的转换，然后进行卷积。
        
        Args:
            x (torch.Tensor): 形状为 (B, C, H, W) 的输入张量
            
        Returns:
            (torch.Tensor): 形状为 (B, out_channels, H//2, W//2) 的输出张量
        """
        # Space-to-Depth: (B, C, H, W) -> (B, 4C, H/2, W/2)
        x = self.space_to_depth(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x
    
    @staticmethod
    def space_to_depth(x, block_size=2):
        """
        将空间维度转换为深度维度。
        
        Args:
            x (torch.Tensor): 输入张量，形状 (B, C, H, W)
            block_size (int): 块大小，默认 2
            
        Returns:
            (torch.Tensor): 形状为 (B, C*block_size^2, H//block_size, W//block_size)
        """
        B, C, H, W = x.shape
        # 确保尺寸可被 block_size 整除
        assert H % block_size == 0 and W % block_size == 0, \
            f"Height ({H}) and Width ({W}) must be divisible by block_size ({block_size})"
        
        # 重塑并重新排列
        x = x.view(B, C, H // block_size, block_size, W // block_size, block_size)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(B, C * (block_size ** 2), H // block_size, W // block_size)
        return x


class EMA(nn.Module):
    """
    EMA: Efficient Multi-Scale Attention
    实现轻量化的跨空间维度注意力机制，用于增强特征表达能力。
    
    通过分组卷积和多尺度池化实现高效的注意力计算。
    
    Args:
        channels (int): 输入/输出通道数
        num_groups (int): 分组数，默认 8
        spatial_kernel (int): 空间注意力的卷积核大小，默认 7
        
    Examples:
        >>> import torch
        >>> m = EMA(256)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> y = m(x)
        >>> print(y.shape)  # torch.Size([1, 256, 32, 32])
    """
    
    def __init__(self, channels, num_groups=8, spatial_kernel=7):
        super().__init__()
        self.channels = channels
        self.num_groups = num_groups
        assert channels % num_groups == 0, f"channels ({channels}) must be divisible by num_groups ({num_groups})"
        
        self.group_channels = channels // num_groups
        
        # 1x1 卷积用于通道注意力（使用 GroupNorm 避免批次大小限制）
        reduced_channels = max(channels // 4, 8)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced_channels, 1, bias=False),
            nn.GroupNorm(num_groups=min(reduced_channels, 4), num_channels=reduced_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # 分组卷积用于空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=spatial_kernel,
                padding=spatial_kernel // 2,
                groups=num_groups,
                bias=False
            ),
            nn.GroupNorm(num_groups=num_groups, num_channels=channels),
            nn.Sigmoid()
        )
        
        # 多尺度池化路径
        self.pool_sizes = [1, 3, 5]
        self.pool_convs = nn.ModuleList([
            nn.Sequential(
                nn.AvgPool2d(kernel_size=k, stride=1, padding=k // 2),
                nn.Conv2d(channels, channels, 1, bias=False),
                nn.GroupNorm(num_groups=num_groups, num_channels=channels)
            ) for k in self.pool_sizes
        ])
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * (len(self.pool_sizes) + 1), channels, 1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=channels),
            nn.SiLU(inplace=True)
        )
    
    def forward(self, x):
        """
        前向传播：计算多尺度注意力并应用到输入特征。
        
        Args:
            x (torch.Tensor): 形状为 (B, C, H, W) 的输入张量
            
        Returns:
            (torch.Tensor): 形状为 (B, C, H, W) 的输出张量
        """
        B, C, H, W = x.shape
        identity = x
        
        # 通道注意力
        ca = self.channel_attention(x)
        x_ca = x * ca
        
        # 空间注意力
        sa = self.spatial_attention(x)
        x_sa = x * sa
        
        # 多尺度池化
        pool_feats = [x_ca]
        for pool_conv in self.pool_convs:
            pool_feats.append(pool_conv(x_sa))
        
        # 融合所有尺度
        x_fused = torch.cat(pool_feats, dim=1)
        x_fused = self.fusion(x_fused)
        
        # 残差连接
        out = x_fused + identity
        return out
