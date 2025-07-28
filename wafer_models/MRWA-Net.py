import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np
from einops import rearrange

# todo 复现  Waveletintegrated attention network withmulti-resolution frequency learning for mixed-type waferdefect recognition
# ==================== 小波变换工具函数 ====================
def get_wavelet_kernels(wavelet='sym2'):
    """获取Symlet2小波基的卷积核"""
    wavelet = pywt.Wavelet(wavelet)
    dec_lo = torch.tensor(wavelet.dec_lo).float()
    dec_hi = torch.tensor(wavelet.dec_hi).float()
    return dec_lo, dec_hi


def dwt_2d(x, wavelet='sym2'):
    """2D离散小波变换(保留LL, LH, HL)"""
    # 获取滤波器
    dec_lo, dec_hi = get_wavelet_kernels(wavelet)

    # 扩展维度 (batch, channel, height, width)
    B, C, H, W = x.shape

    # 水平方向滤波
    x = x.reshape(-1, 1, H, W)
    x = F.pad(x, (0, 1, 0, 1), mode='reflect')  # 边界填充

    # 低通滤波 (水平)
    low_h = F.conv2d(x, dec_lo.view(1, 1, 1, -1), stride=(1, 2))
    # 高通滤波 (水平)
    high_h = F.conv2d(x, dec_hi.view(1, 1, 1, -1), stride=(1, 2))

    # 垂直方向滤波
    low_h = F.pad(low_h, (0, 0, 0, 1), mode='reflect')
    high_h = F.pad(high_h, (0, 0, 0, 1), mode='reflect')

    # 低通+低通 (LL)
    ll = F.conv2d(low_h, dec_lo.view(1, 1, -1, 1), stride=(2, 1))
    # 低通+高通 (LH)
    lh = F.conv2d(low_h, dec_hi.view(1, 1, -1, 1), stride=(2, 1))
    # 高通+低通 (HL)
    hl = F.conv2d(high_h, dec_lo.view(1, 1, -1, 1), stride=(2, 1))

    # 丢弃HH子带
    ll = ll.reshape(B, C, H // 2, W // 2)
    lh = lh.reshape(B, C, H // 2, W // 2)
    hl = hl.reshape(B, C, H // 2, W // 2)

    return torch.cat([ll, lh, hl], dim=1)  # 通道维度拼接


# ==================== 网络核心模块 ====================
class ResBlock(nn.Module):
    """残差块"""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)


class FLA(nn.Module):
    """频率位置注意力模块"""

    def __init__(self, channels):
        super().__init__()
        # 1D自编码器分支
        self.fc1 = nn.Linear(channels, channels // 2)
        self.fc2 = nn.Linear(channels // 2, channels)

        # 3D自编码器分支
        self.conv1 = nn.Conv2d(channels, channels // 2, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(channels // 2)
        self.conv2 = nn.Conv2d(channels // 2, channels // 2, kernel_size=3,
                               dilation=4, padding=4)
        self.bn2 = nn.BatchNorm2d(channels // 2)
        self.conv3 = nn.Conv2d(channels // 2, channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(channels)

        # HSigmoid激活函数
        self.hsigmoid = nn.Hardsigmoid()

    def forward(self, x):
        # 1D分支 (通道注意力)
        channel_att = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        channel_att = F.relu(self.fc1(channel_att))
        channel_att = torch.sigmoid(self.fc2(channel_att))

        # 3D分支 (空间注意力)
        spatial_att = F.relu(self.bn1(self.conv1(x)))
        spatial_att = F.relu(self.bn2(self.conv2(spatial_att)))
        spatial_att = self.hsigmoid(self.bn3(self.conv3(spatial_att)))

        # 组合注意力机制
        x = x * channel_att.view(-1, x.size(1), 1, 1)
        x = x * spatial_att
        return x


class DWT_Layer(nn.Module):
    """离散小波变换层"""

    def __init__(self, in_channels, out_channels, levels=1):
        super().__init__()
        self.levels = levels
        self.wavelet = 'sym2'

        # 小波域学习器 (WDL)
        self.res1 = ResBlock(out_channels)
        self.res2 = ResBlock(out_channels)
        self.fla = FLA(out_channels)

        # 通道调整卷积
        self.conv_adjust = nn.Conv2d(in_channels * 3, out_channels, kernel_size=1)

    def forward(self, x):
        # 多级小波变换
        for _ in range(self.levels):
            x = dwt_2d(x, self.wavelet)

        # 通道调整
        x = self.conv_adjust(x)

        # 小波域学习
        x = self.res1(x)
        x = self.res2(x)
        x = self.fla(x)
        return x


# ==================== 多分辨率网络架构 ====================
class MRWA_Net(nn.Module):
    """多分辨率小波注意力网络"""

    def __init__(self, in_channels=1, num_classes=8):
        super().__init__()

        # 分支B1: 三个1级DWT层 (逐步下采样)
        self.branch_b1 = nn.Sequential(
            DWT_Layer(in_channels, 32, levels=1),
            DWT_Layer(32, 64, levels=1),
            DWT_Layer(64, 128, levels=1)
        )

        # 分支B2: 一个2级DWT层 + 一个1级DWT层
        self.branch_b2 = nn.Sequential(
            DWT_Layer(in_channels, 64, levels=2),
            DWT_Layer(64, 128, levels=1)
        )

        # 分支B3: 一个3级DWT层
        self.branch_b3 = DWT_Layer(in_channels, 128, levels=3)

        # 分类头
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(128 * 3, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
            nn.Sigmoid()  # 多标签分类
        )

    def forward(self, x):
        # 原始输入尺寸: (B, 1, 52, 52)

        # 三路分支处理
        b1_out = self.branch_b1(x)  # 输出: (B, 128, 6, 6)
        b2_out = self.branch_b2(x)  # 输出: (B, 128, 6, 6)
        b3_out = self.branch_b3(x)  # 输出: (B, 128, 6, 6)

        # 全局平均池化
        b1_out = self.gap(b1_out).squeeze(-1).squeeze(-1)
        b2_out = self.gap(b2_out).squeeze(-1).squeeze(-1)
        b3_out = self.gap(b3_out).squeeze(-1).squeeze(-1)

        # 拼接特征
        features = torch.cat([b1_out, b2_out, b3_out], dim=1)

        # 分类
        logits = self.fc(features)
        return logits


# ==================== 损失函数与训练配置 ====================
class MultiLabelLoss(nn.Module):
    """多标签分类损失 (带阈值处理)"""

    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold
        self.bce_loss = nn.BCELoss()

    def forward(self, outputs, targets):
        # 应用阈值确定标签存在性
        preds = (outputs > self.threshold).float()
        return self.bce_loss(outputs, targets)


# 训练配置参数
config = {
    'input_size': (52, 52),
    'num_classes': 8,
    'batch_size': 128,
    'lr': 0.001,
    'epochs': 200,
    'threshold': 0.5,  # 多标签分类阈值
    'wavelet': 'sym2'
}

# ==================== 模型初始化与测试 ====================
if __name__ == "__main__":
    # 创建模型实例
    model = MRWA_Net(in_channels=1, num_classes=config['num_classes'])

    # 打印模型结构
    print(f"MRWA-Net 参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 测试输入
    test_input = torch.randn(2, 1, 52, 52)
    output = model(test_input)
    print(f"输入尺寸: {test_input.shape} 输出尺寸: {output.shape}")