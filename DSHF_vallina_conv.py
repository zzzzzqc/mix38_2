import torch
import torch.nn as nn
import torch.fft as fft
from torch.nn import functional as F
from oct_conv import *
from fea_fusion import Cross_Encoder_Fusion, Cross_Layer_Fusion
from einops.layers.torch import Rearrange


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()

        if mid_channels is None:
            mid_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        residual = self.conv3(x)
        out = out + residual
        return self.relu(out)


# class FrequencyFilter(nn.Module):
#     def __init__(self, in_channels, out_channels, first=False):
#         super().__init__()
#         self.first = first
#
#         if self.first:
#             self.first_conv = nn.Conv2d(1, 8, kernel_size=(1, 1),  stride=1, padding=0)
#
#         self.amp_mask = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
#         )
#         self.pha_mask = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
#         )
#
#         self.channel_adjust = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#
#     def forward(self, x):
#
#         if self.first: x = self.first_conv(x)
#
#         b, c, h, w = x.shape
#
#         msF = fft.rfft2(x + 1e-8, norm='backward')
#
#         msF_amp = torch.abs(msF)
#         msF_pha = torch.angle(msF)
#
#         amp_fuse = self.amp_mask(msF_amp) + msF_amp
#         pha_fuse = self.pha_mask(msF_pha) + msF_pha
#
#         real = amp_fuse * torch.cos(pha_fuse) + 1e-8
#         imag = amp_fuse * torch.sin(pha_fuse) + 1e-8
#
#         out = torch.complex(real, imag) + 1e-8
#         out = torch.abs(torch.fft.irfft2(out, s=(h, w), norm='backward'))
#         out = out + x
#         out = self.channel_adjust(out)
#         out = torch.nan_to_num(out, nan=1e-5, posinf=1e-5, neginf=1e-5)
#
#         return out


class FrequencyFilter(nn.Module):
    """
    频域滤波模块，支持五种模式：
    - 'original'      : 直接处理相位角（原始方式）
    - 'sincos'        : 在正弦-余弦空间处理相位（连续、稳定）
    - 'magnitude_only': 仅利用幅度谱，相位置零
    - 'phase_only'    : 仅利用相位谱，幅度归一化为1（正弦-余弦空间）
    - 'angle_only'    : 仅利用相位谱，直接处理原始角度（幅度固定为1）
    """

    def __init__(self, in_channels, out_channels, first=False, mode='sincos'):
        super().__init__()
        self.first = first
        self.mode = mode
        self.in_channels = in_channels
        self.out_channels = out_channels

        if self.first:
            self.first_conv = nn.Conv2d(1, in_channels, kernel_size=1)

        # 创建幅值掩码（除 phase_only 和 angle_only 外均使用）
        if mode != 'phase_only' and mode != 'angle_only':
            self.amp_mask = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(in_channels, in_channels, 3, padding=1)
            )

        # 创建相位掩码（根据模式选择通道数）
        if mode == 'original':
            self.pha_mask = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(in_channels, in_channels, 3, padding=1)
            )
        elif mode in ['sincos', 'phase_only']:
            self.pha_mask = nn.Sequential(
                nn.Conv2d(2 * in_channels, 2 * in_channels, 3, padding=1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(2 * in_channels, 2 * in_channels, 3, padding=1)
            )
        elif mode == 'angle_only':   # 新增
            self.pha_mask = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(in_channels, in_channels, 3, padding=1)
            )
        # magnitude_only 模式无需相位掩码

        # 通道调整层
        self.channel_adjust = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # 短路连接
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        if self.first:
            x = self.first_conv(x)

        identity = self.shortcut(x)
        b, c, h, w = x.shape

        # 频域变换
        msF = fft.rfft2(x + 1e-8, norm='backward')
        msF_amp = torch.abs(msF)
        msF_pha = torch.angle(msF)

        # ---------- 根据模式处理 ----------
        if self.mode == 'original':
            # 原始相位角模式
            amp_fuse = self.amp_mask(msF_amp) + msF_amp
            pha_fuse = self.pha_mask(msF_pha) + msF_pha
            real = amp_fuse * torch.cos(pha_fuse)
            imag = amp_fuse * torch.sin(pha_fuse)

        elif self.mode == 'sincos':
            # 正弦-余弦空间模式（同时学习幅值和相位）
            cos_pha, sin_pha = torch.cos(msF_pha), torch.sin(msF_pha)
            pha_feat = torch.cat([cos_pha, sin_pha], dim=1)
            pha_residual = self.pha_mask(pha_feat)
            delta_cos = pha_residual[:, :c, :, :]
            delta_sin = pha_residual[:, c:, :, :]
            cos_fuse = cos_pha + delta_cos
            sin_fuse = sin_pha + delta_sin
            norm = torch.sqrt(cos_fuse ** 2 + sin_fuse ** 2 + 1e-8)
            cos_fuse = cos_fuse / norm
            sin_fuse = sin_fuse / norm
            amp_fuse = self.amp_mask(msF_amp) + msF_amp
            real = amp_fuse * cos_fuse
            imag = amp_fuse * sin_fuse

        elif self.mode == 'magnitude_only':
            # 仅幅度谱模式
            amp_fuse = self.amp_mask(msF_amp) + msF_amp
            real, imag = amp_fuse, torch.zeros_like(amp_fuse)

        elif self.mode == 'phase_only':
            # 仅相位谱模式（正弦-余弦空间，幅度固定为1）
            cos_pha, sin_pha = torch.cos(msF_pha), torch.sin(msF_pha)
            pha_feat = torch.cat([cos_pha, sin_pha], dim=1)
            pha_residual = self.pha_mask(pha_feat)
            delta_cos = pha_residual[:, :c, :, :]
            delta_sin = pha_residual[:, c:, :, :]
            cos_fuse = cos_pha + delta_cos
            sin_fuse = sin_pha + delta_sin
            norm = torch.sqrt(cos_fuse ** 2 + sin_fuse ** 2 + 1e-8)
            cos_fuse = cos_fuse / norm
            sin_fuse = sin_fuse / norm
            real = cos_fuse
            imag = sin_fuse

        elif self.mode == 'angle_only':   # 新增
            # 仅相位角度模式：直接处理原始角度，幅度固定为1
            pha_fuse = self.pha_mask(msF_pha) + msF_pha
            real = torch.cos(pha_fuse)
            imag = torch.sin(pha_fuse)

        # 逆变换
        out = fft.irfft2(torch.complex(real, imag), s=(h, w), norm='backward')

        # 残差 + 通道调整
        out = self.channel_adjust(out) + identity
        out = torch.nan_to_num(out, nan=1e-5, posinf=1e-5, neginf=1e-5)

        return out


class SpatialEnc(nn.Module):
    def __init__(self, in_channels):
        super(SpatialEnc, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.enc_blk1 = DoubleConv(in_channels, out_channels=64)
        self.enc_blk2 = DoubleConv(in_channels=64, out_channels=128)
        self.enc_blk3 = DoubleConv(in_channels=128, out_channels=256)
        self.enc_blk4 = DoubleConv(in_channels=256, out_channels=512)

    def forward(self, x):
        x0 = self.enc_blk1(x)  # [64, H, W]

        x1 = self.maxpool(x0)  # [64, H/2, W/2]

        x2 = self.enc_blk2(x1)  # [128, H/2, W/2]

        x2 = self.maxpool(x2)  # [128, H/4, W/4]

        x3 = self.enc_blk3(x2)  # [256, H/4, W/4]

        x3 = self.maxpool(x3)  # [256, H/8, W/8]

        x4 = self.enc_blk4(x3)  # [512, H/8, W/8]

        x4 = self.maxpool(x4)  # [512, H/16, W/16]

        return x0, x1, x2, x3, x4


class FrequencyEnc(nn.Module):
    def __init__(self, in_channels, mode='original'):  # original,sincos,magnitude_only,phase_only
        super(FrequencyEnc, self).__init__()
        self.dounsample = nn.MaxPool2d(2)
        self.enc_blk1 = FrequencyFilter(in_channels * 8, out_channels=64, mode=mode, first=True)
        self.enc_blk2 = FrequencyFilter(in_channels=64, out_channels=128, mode=mode)
        self.enc_blk3 = FrequencyFilter(in_channels=128, out_channels=256, mode=mode)
        self.enc_blk4 = FrequencyFilter(in_channels=256, out_channels=512, mode=mode)

    def forward(self, x):
        x0 = self.enc_blk1(x)  # [64, H, W]

        x1 = self.dounsample(x0)  # [64, H/2, W/2]

        x2 = self.enc_blk2(x1)  # [128, H/2, W/2]

        x2 = self.dounsample(x2)  # [128, H/4, W/4]

        x3 = self.enc_blk3(x2)  # [256, H/4, W/4]

        x3 = self.dounsample(x3)  # [256, H/8, W/8]

        x4 = self.enc_blk4(x3)  # [512, H/8, W/8]

        x4 = self.dounsample(x4)  # [512, H/16, W/16]

        return x0, x1, x2, x3, x4


class DSHF_Integrated(nn.Module):
    def __init__(self, in_channels, dim, size, num_classes=38, mode='mixed', freq_mode='original', norm_layer=None):
        # # original,sincos,magnitude_only
        """
        - mode='mixed' : 同时使用空间和频率编码 (默认)
        - mode='spa'   : 仅使用空间编码
        - mode='freq'  : 仅使用频率编码
        """
        super().__init__()
        self.mode = mode
        self.relu = nn.ReLU(inplace=True)

        if self.mode in ['mixed', 'spa']:
            self.spatial_enc = SpatialEnc(in_channels)
        if self.mode in ['mixed', 'freq']:  # original,sincos,magnitude_only,phase_only
            self.frequency_enc = FrequencyEnc(in_channels, mode=freq_mode)

        self.cross_layer_fusion = Cross_Layer_Fusion(dim, size)

        if self.mode == 'mixed':
            self.cross_encoder_fusion2 = Cross_Encoder_Fusion(128)
            self.cross_encoder_fusion3 = Cross_Encoder_Fusion(256)
            self.cross_encoder_fusion4 = Cross_Encoder_Fusion(512)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        spa_features, freq_features = [], []
        final_out = None
        if self.mode in ['mixed', 'spa']:
            *spa_features, = self.spatial_enc(x)  # 解包所有空间特征
        if self.mode in ['mixed', 'freq']:
            *freq_features, = self.frequency_enc(x)  # 解包所有频率特征

        if self.mode == 'mixed':

            out2 = self.cross_encoder_fusion2(spa_features[2], freq_features[2])
            out3 = self.cross_encoder_fusion3(spa_features[3], freq_features[3])
            out4 = self.cross_encoder_fusion4(spa_features[4], freq_features[4])
            final_out = self.cross_layer_fusion(out2, out3, out4)
        elif self.mode == 'spa':
            final_out = self.cross_layer_fusion(spa_features[2], spa_features[3], spa_features[4])
        elif self.mode == 'freq':
            final_out = self.cross_layer_fusion(freq_features[2], freq_features[3], freq_features[4])

        # 分类处理
        final_out = self.avgpool(final_out)
        return self.fc(final_out.view(final_out.size(0), -1))


def build_integrated_model(mode='mixed', freq_mode='original', **kwargs):
    dim, size = [128, 256, 512], 196
    return DSHF_Integrated(
        in_channels=1,
        dim=dim,
        size=size,
        mode=mode,
        freq_mode=freq_mode,
        **kwargs
    )


def model_18_vallina(**kwargs):
    mixed_model = build_integrated_model(mode='mixed', **kwargs)
    return mixed_model


def model18_only_spa_vallina(**kwargs):
    spa_model = build_integrated_model(mode='spa', **kwargs)
    return spa_model


# original,sincos,magnitude_only,phase_only
def model18_only_freq_vallina_ori(pretrained=False, **kwargs):
    freq_model_or = build_integrated_model(mode='freq', freq_mode='original', **kwargs)
    return freq_model_or


def model18_only_freq_vallina_sincos(pretrained=False, **kwargs):
    freq_model_sc = build_integrated_model(mode='freq', freq_mode='sincos', **kwargs)
    return freq_model_sc


def model18_only_freq_vallina_amp(pretrained=False, **kwargs):
    freq_model_amp = build_integrated_model(mode='freq', freq_mode='magnitude_only', **kwargs)
    return freq_model_amp


def model18_only_freq_vallina_pha(pretrained=False, **kwargs):
    freq_model_pha = build_integrated_model(mode='freq', freq_mode='phase_only', **kwargs)
    return freq_model_pha

def model18_only_freq_vallina_ang(pretrained=False, **kwargs):
    freq_model_pha = build_integrated_model(mode='freq', freq_mode='angle_only', **kwargs)
    return freq_model_pha

from thop import profile
from thop import clever_format


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_flops(model, input):
    flops = profile(model, inputs=(input,))[0] * 2
    flops = clever_format(flops, "%.2f")
    return flops


if __name__ == '__main__':
    # 创建不同模式模型
    # mixed_model = model_18_vallina(num_classes=9)
    spa_model = model18_only_spa_vallina(num_classes=10)
    freq_model1 = model18_only_freq_vallina_ori(num_classes=11)
    freq_model2 = model18_only_freq_vallina_amp(num_classes=11)
    freq_model3 = model18_only_freq_vallina_pha(num_classes=11)
    freq_model4 = model18_only_freq_vallina_sincos(num_classes=11)
    mixed_model = freq_model4

    x = torch.randn(1, 1, 64, 64)
    print("Mixed模型输出尺寸:", mixed_model(x).shape)
    print("模型输出尺寸:", mixed_model(x).shape)
    total = count_parameters(mixed_model)
    flops = count_flops(mixed_model, x)
    print("Number of parameter: %.2fM" % (total / 1e6))
    print(flops)

    # # 参数统计
    # from torchinfo import summary
    #
    # summary(mixed_model, input_size=(1, 1, 64, 64))
