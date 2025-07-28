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


class FrequencyFilter(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.amp_mask = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        )
        self.pha_mask = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        )

        self.channel_adjust = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        msF = fft.rfft2(x + 1e-8, norm='backward')

        msF_amp = torch.abs(msF)
        msF_pha = torch.angle(msF)

        amp_fuse = self.amp_mask(msF_amp) + msF_amp
        pha_fuse = self.pha_mask(msF_pha) + msF_pha

        real = amp_fuse * torch.cos(pha_fuse) + 1e-8
        imag = amp_fuse * torch.sin(pha_fuse) + 1e-8

        out = torch.complex(real, imag) + 1e-8
        out = torch.abs(torch.fft.irfft2(out, s=(h, w), norm='backward'))
        out = out + x
        out = self.channel_adjust(out)
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
    def __init__(self, in_channels):
        super(FrequencyEnc, self).__init__()
        self.dounsample = nn.MaxPool2d(2)
        self.enc_blk1 = FrequencyFilter(in_channels, out_channels=64)
        self.enc_blk2 = FrequencyFilter(in_channels=64, out_channels=128)
        self.enc_blk3 = FrequencyFilter(in_channels=128, out_channels=256)
        self.enc_blk4 = FrequencyFilter(in_channels=256, out_channels=512)

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
    def __init__(self, in_channels, dim, size, num_classes=38, mode='mixed', norm_layer=None):
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
        if self.mode in ['mixed', 'freq']:
            self.frequency_enc = FrequencyEnc(in_channels)

        self.cross_layer_fusion = Cross_Layer_Fusion(dim, size)

        if self.mode == 'mixed':
            self.cross_encoder_fusion2 = Cross_Encoder_Fusion(128)
            self.cross_encoder_fusion3 = Cross_Encoder_Fusion(256)
            self.cross_encoder_fusion4 = Cross_Encoder_Fusion(512)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        spa_features, freq_features = [], []
        final_out =None
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


def build_integrated_model(mode='mixed', **kwargs):
    dim, size = [128, 256, 512], 196
    return DSHF_Integrated(
        in_channels=1,
        dim=dim,
        size=size,
        mode=mode,
        **kwargs
    )


def model_18_vallina(**kwargs):
    mixed_model = build_integrated_model(mode='mixed', **kwargs)
    return mixed_model


def model18_only_spa_vallina(**kwargs):
    spa_model = build_integrated_model(mode='spa', **kwargs)
    return spa_model


def model18_only_freq_vallina(pretrained=False, **kwargs):
    freq_model = build_integrated_model(mode='freq', **kwargs)
    return freq_model
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
    mixed_model = model_18_vallina(num_classes=9)
    spa_model = model18_only_spa_vallina(num_classes=10)
    freq_model = model18_only_freq_vallina(num_classes=11)

    x = torch.randn(1, 1, 64, 64)
    print("Mixed模型输出尺寸:", mixed_model(x).shape)
    print("模型输出尺寸:", mixed_model(x).shape)
    total = count_parameters(mixed_model)
    flops = count_flops(mixed_model, x)
    print("Number of parameter: %.2fM" % (total/1e6))
    print(flops)
    print("Spa模型输出尺寸:", spa_model(x).shape)
    print("Freq模型输出尺寸:", freq_model(x).shape)

    # # 参数统计
    # from torchinfo import summary
    #
    # summary(mixed_model, input_size=(1, 1, 64, 64))

