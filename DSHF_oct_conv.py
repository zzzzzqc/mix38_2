import torch
import torch.nn as nn
import torch.fft as fft
from torch.nn import functional as F
from graph_module import channel_graph_interaction
from oct_conv import *
from fea_fusion import Cross_Encoder_Fusion, Cross_Layer_Fusion
from einops.layers.torch import Rearrange


class DoubleOCTConv(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, alpha, stride=1, groups=1,
                 base_width=64, norm_layer=None, First=False):
        super(DoubleOCTConv, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(out_channels * (base_width / 64.)) * groups
        self.first = First
        if self.first:
            self.ocb1 = FirstOctaveCBR(in_channels, width, kernel_size=(3, 3), alpha=alpha, padding=1)
        else:
            self.ocb1 = OctaveCBR(in_channels, width, kernel_size=(3, 3), alpha=alpha, padding=1)

        self.ocb2 = OctaveCB(width, out_channels, kernel_size=(3, 3), alpha=alpha, padding=1)

        if self.first:
            self.ocb3 = FirstOctaveConv(in_channels, width, kernel_size=(3, 3), alpha=alpha, padding=1)
        else:
            self.ocb3 = OctaveConv(in_channels, out_channels, kernel_size=(3, 3), alpha=alpha)

        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        # try:
        #     print('input',x[0].shape,x[1].shape)
        # except:
        #     print('input',x.shape)

        if self.first:
            x_h_0, x_l_0 = self.ocb1(x)
            # print('first_1',x_h_res.shape, x_l_res.shape)
            x_h_1, x_l_1 = self.ocb2((x_h_0, x_l_0))
            # print('first_2',x_h.shape, x_l.shape)
            x_h_res, x_l_res = self.ocb3(x)
            x_h = x_h_1 + x_h_res
            x_l = x_l_1 + x_l_res
        else:
            x_h_0, x_l_0 = x
            # print('input', x_h_0.shape, x_l_0.shape)
            x_h_1, x_l_1 = self.ocb1((x_h_0, x_l_0))
            # print('conn1', x_h_1.shape, x_l_1.shape)
            x_h_2, x_l_2 = self.ocb2((x_h_1, x_l_1))
            # print('conv2', x_h_2.shape, x_l_2.shape)
            x_h_res, x_l_res = self.ocb3((x_h_0, x_l_0))
            # print('res', x_h_res.shape, x_l_res.shape)
            x_h = x_h_2 + x_h_res
            x_l = x_l_2 + x_l_res

        x_h = self.relu(x_h)
        x_l = self.relu(x_l)
        # print('output',x_h.shape, x_l.shape)
        # print('double block down')
        return x_h, x_l


class FrequencyOCTFilter(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, alpha, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None, First=False):
        super().__init__()
        self.first = First
        self.alpha = alpha
        if self.first:
            self.first_conv = FirstOctaveCBR(1, 8, kernel_size=(1, 1), alpha=self.alpha, stride=1, padding=0)

        self.amp_mask = nn.Sequential(
            FreqOctaveCR(in_channels, in_channels, kernel_size=(3, 3), alpha=self.alpha, stride=1, padding=1),
            OctaveConv(in_channels, in_channels, kernel_size=(3, 3), alpha=self.alpha, stride=1, padding=1)
        )
        self.pha_mask = nn.Sequential(
            FreqOctaveCR(in_channels, in_channels, kernel_size=(3, 3), alpha=self.alpha, stride=1, padding=1),
            OctaveConv(in_channels, in_channels, kernel_size=(3, 3), alpha=self.alpha, stride=1, padding=1)
        )

        self.channel_adjust_l = nn.Conv2d(int(self.alpha * in_channels), int(self.alpha * out_channels), kernel_size=1)
        self.channel_adjust_h = nn.Conv2d(int((1 - self.alpha) * in_channels), int((1 - self.alpha) * out_channels),
                                          kernel_size=1)

    def forward(self, x):
        # print(len(x))
        if isinstance(x, tuple):
            x_h, x_l = x[0], x[1]
        else:
            b, c, h, w = x.shape
            if c == 1 or c == 3:
                x_h, x_l = self.first_conv(x)

        b_h, c_h, h_h, w_h = x_h.shape
        b_l, c_l, h_l, w_l = x_l.shape

        msF_h = fft.rfft2(x_h + 1e-8, norm='backward')
        msF_l = fft.rfft2(x_l + 1e-8, norm='backward')

        msF_amp_h = torch.abs(msF_h)
        msF_amp_l = torch.abs(msF_l)
        msF_pha_h = torch.angle(msF_h)
        msF_pha_l = torch.angle(msF_l)
        msF_amp = (msF_amp_h, msF_amp_l)
        msF_pha = (msF_pha_h, msF_pha_l)
        # print(msF_amp[0].shape, msF_amp[1].shape)
        amp_mask_h, amp_mask_l = self.amp_mask(msF_amp)
        amp_fuse_h = amp_mask_h + msF_amp_h
        amp_fuse_l = amp_mask_l + msF_amp_l
        pha_mask_h, pha_mask_l = self.pha_mask(msF_pha)
        pha_fuse_h = pha_mask_h + msF_pha_h
        pha_fuse_l = pha_mask_l + msF_pha_l

        real_h = amp_fuse_h * torch.cos(pha_fuse_h) + 1e-8
        real_l = amp_fuse_l * torch.cos(pha_fuse_l) + 1e-8

        imag_h = amp_fuse_h * torch.sin(pha_fuse_h) + 1e-8
        imag_l = amp_fuse_l * torch.sin(pha_fuse_l) + 1e-8
        out_h = torch.complex(real_h, imag_h) + 1e-8
        out_l = torch.complex(real_l, imag_l) + 1e-8
        out_h = torch.abs(torch.fft.irfft2(out_h, s=(h_h, w_h), norm='backward'))
        out_l = torch.abs(torch.fft.irfft2(out_l, s=(h_l, w_l), norm='backward'))
        out_h = out_h + x_h
        out_l = out_l + x_l
        out_h = self.channel_adjust_h(out_h)
        out_l = self.channel_adjust_l(out_l)
        out_h = torch.nan_to_num(out_h, nan=1e-5, posinf=1e-5, neginf=1e-5)
        out_l = torch.nan_to_num(out_l, nan=1e-5, posinf=1e-5, neginf=1e-5)

        return out_h, out_l


class SpatialOCTEnc(nn.Module):
    def __init__(self, in_channels, alpha):
        super(SpatialOCTEnc, self).__init__()
        self.dounsample = nn.MaxPool2d(2)
        self.enc_blk1 = DoubleOCTConv(in_channels, out_channels=64, alpha=alpha, First=True)

        self.enc_blk2 = DoubleOCTConv(in_channels=64, out_channels=128, alpha=alpha)

        self.enc_blk3 = DoubleOCTConv(in_channels=128, out_channels=256, alpha=alpha)

        self.enc_blk4 = DoubleOCTConv(in_channels=256, out_channels=512, alpha=alpha)

    def forward(self, x):
        x0 = self.enc_blk1(x)  # [64, H, W]

        x1 = (self.dounsample(x0[0]), self.dounsample(x0[1]))  # [64, H/2, W/2]

        x2 = self.enc_blk2(x1)  # [128, H/2, W/2]

        x2 = (self.dounsample(x2[0]), self.dounsample(x2[1]))  # [128, H/4, W/4]

        x3 = self.enc_blk3(x2)  # [256, H/4, W/4]

        x3 = (self.dounsample(x3[0]), self.dounsample(x3[1]))  # [256, H/8, W/8]

        x4 = self.enc_blk4(x3)  # [512, H/8, W/8]

        x4 = (self.dounsample(x4[0]), self.dounsample(x4[1]))  # [512, H/16, W/16]

        return x0, x1, x2, x3, x4


class FrequencyOCTEnc(nn.Module):
    def __init__(self, in_channels, alpha):
        super(FrequencyOCTEnc, self).__init__()
        self.dounsample = nn.MaxPool2d(2)
        self.enc_blk1 = FrequencyOCTFilter(in_channels * 8, out_channels=64, alpha=alpha, First=True)
        self.enc_blk2 = FrequencyOCTFilter(in_channels=64, out_channels=128, alpha=alpha)
        self.enc_blk3 = FrequencyOCTFilter(in_channels=128, out_channels=256, alpha=alpha)
        self.enc_blk4 = FrequencyOCTFilter(in_channels=256, out_channels=512, alpha=alpha)

    def forward(self, x):
        x0 = self.enc_blk1(x)  # [64, H, W]

        x1 = (self.dounsample(x0[0]), self.dounsample(x0[1]))  # [64, H/2, W/2]

        x2 = self.enc_blk2(x1)  # [128, H/2, W/2]

        x2 = (self.dounsample(x2[0]), self.dounsample(x2[1]))  # [128, H/4, W/4]

        x3 = self.enc_blk3(x2)  # [256, H/4, W/4]

        x3 = (self.dounsample(x3[0]), self.dounsample(x3[1]))  # [256, H/8, W/8]

        x4 = self.enc_blk4(x3)  # [512, H/8, W/8]

        x4 = (self.dounsample(x4[0]), self.dounsample(x4[1]))  # [512, H/16, W/16]
        return x0, x1, x2, x3, x4


class DSHF_OCT_Integrated(nn.Module):
    def __init__(self, in_channels, dim, size, num_classes=38, mode='mixed', alpha=0.5, norm_layer=None):
        """
        - mode='mixed' : 同时使用空间和频率编码 (默认)
        - mode='spa'   : 仅使用空间编码
        - mode='freq'  : 仅使用频率编码
        """
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.mode = mode
        # print(self.mode)
        self.relu = nn.ReLU(inplace=True)
        self.conv_out_layer1 = LastOctaveConv(64, 64, kernel_size=(1, 1), alpha=alpha, stride=1, padding=0, groups=1)
        self.conv_out_layer2 = LastOctaveConv(128, 128, kernel_size=(1, 1), alpha=alpha, stride=1, padding=0, groups=1)
        self.conv_out_layer3 = LastOctaveConv(256, 256, kernel_size=(1, 1), alpha=alpha, stride=1, padding=0, groups=1)
        self.conv_out_layer4 = LastOctaveConv(512, 512, kernel_size=(1, 1), alpha=alpha, stride=1, padding=0, groups=1)
        if self.mode in ['mixed', 'spa', 'mixed_wo_cross_encoder', 'mixed_wo_cross_layer']:
            # print(f'in mode {self.mode} get spa_enc')
            self.spatial_enc = SpatialOCTEnc(in_channels, alpha=alpha)
        if self.mode in ['mixed', 'freq', 'mixed_wo_cross_encoder', 'mixed_wo_cross_layer']:
            # print(f'in mode {self.mode} get freq_enc')
            self.frequency_enc = FrequencyOCTEnc(in_channels, alpha=alpha)

        self.cross_layer_fusion = Cross_Layer_Fusion(dim, size)

        if self.mode in ['mixed', 'spa', 'freq', 'mixed_wo_cross_encoder', 'mixed_wo_cross_layer']:
            self.cross_encoder_fusion2 = Cross_Encoder_Fusion(128)
            self.cross_encoder_fusion3 = Cross_Encoder_Fusion(256)
            self.cross_encoder_fusion4 = Cross_Encoder_Fusion(512)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        if self.mode == 'spa_single_out' or self.mode == 'freq_single_out' or self.mode == 'mixed_wo_cross_layer':
            self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        spa_layer0, spa_layer1, spa_layer2, spa_layer3, spa_layer4 = None, None, None, None, None,
        freq_layer0, freq_layer1, freq_layer2, freq_layer3, freq_layer4 = None, None, None, None, None,
        final_out = None
        if self.mode in ['mixed', 'spa', 'mixed_wo_cross_encoder', 'mixed_wo_cross_layer']:
            spa_layer0, spa_layer1, spa_layer2, spa_layer3, spa_layer4 = self.spatial_enc(x)  # 解包所有空间特征

        if self.mode in ['mixed', 'freq', 'mixed_wo_cross_encoder', 'mixed_wo_cross_layer']:
            freq_layer0, freq_layer1, freq_layer2, freq_layer3, freq_layer4 = self.frequency_enc(x)  # 解包所有频率特征
        if self.mode == 'mixed':
            spa_features1, freq_features1 = self.conv_out_layer1(spa_layer1), self.conv_out_layer1(freq_layer1)
            spa_features2, freq_features2 = self.conv_out_layer2(spa_layer2), self.conv_out_layer2(freq_layer2)
            spa_features3, freq_features3 = self.conv_out_layer3(spa_layer3), self.conv_out_layer3(freq_layer3)
            spa_features4, freq_features4 = self.conv_out_layer4(spa_layer4), self.conv_out_layer4(freq_layer4)

            out2 = self.cross_encoder_fusion2(spa_features2, freq_features2)
            out3 = self.cross_encoder_fusion3(spa_features3, freq_features3)
            out4 = self.cross_encoder_fusion4(spa_features4, freq_features4)
            final_out = self.cross_layer_fusion(out2, out3, out4)
        if self.mode == 'mixed_wo_cross_encoder':
            spa_features1, freq_features1 = self.conv_out_layer1(spa_layer1), self.conv_out_layer1(freq_layer1)
            spa_features2, freq_features2 = self.conv_out_layer2(spa_layer2), self.conv_out_layer2(freq_layer2)
            spa_features3, freq_features3 = self.conv_out_layer3(spa_layer3), self.conv_out_layer3(freq_layer3)
            spa_features4, freq_features4 = self.conv_out_layer4(spa_layer4), self.conv_out_layer4(freq_layer4)

            out2 = spa_features2 + freq_features2
            out3 = spa_features3 + freq_features3
            out4 = spa_features4 + freq_features4
            final_out = self.cross_layer_fusion(out2, out3, out4)

        if self.mode == 'mixed_wo_cross_layer':
            spa_features1, freq_features1 = self.conv_out_layer1(spa_layer1), self.conv_out_layer1(freq_layer1)
            spa_features2, freq_features2 = self.conv_out_layer2(spa_layer2), self.conv_out_layer2(freq_layer2)
            spa_features3, freq_features3 = self.conv_out_layer3(spa_layer3), self.conv_out_layer3(freq_layer3)
            spa_features4, freq_features4 = self.conv_out_layer4(spa_layer4), self.conv_out_layer4(freq_layer4)

            out2 = self.cross_encoder_fusion2(spa_features2, freq_features2)
            out3 = self.cross_encoder_fusion3(spa_features3, freq_features3)
            out4 = self.cross_encoder_fusion4(spa_features4, freq_features4)
            final_out = out4

        elif self.mode == 'spa':
            spa_features1 = self.conv_out_layer1(spa_layer1)
            spa_features2 = self.conv_out_layer2(spa_layer2)
            spa_features3 = self.conv_out_layer3(spa_layer3)
            spa_features4 = self.conv_out_layer4(spa_layer4)
            final_out = self.cross_layer_fusion(spa_features2, spa_features3, spa_features4)
        elif self.mode == 'freq':
            freq_features1 = self.conv_out_layer1(freq_layer1)
            freq_features2 = self.conv_out_layer2(freq_layer2)
            freq_features3 = self.conv_out_layer3(freq_layer3)
            freq_features4 = self.conv_out_layer4(freq_layer4)
            final_out = self.cross_layer_fusion(freq_features2, freq_features3, freq_features4)
        elif self.mode == 'spa_single_out':
            spa_features1 = self.conv_out_layer1(spa_layer1)
            spa_features2 = self.conv_out_layer2(spa_layer2)
            spa_features3 = self.conv_out_layer3(spa_layer3)
            spa_features4 = self.conv_out_layer4(spa_layer4)
            final_out = spa_features4
        elif self.mode == 'freq_single_out':
            freq_features1 = self.conv_out_layer1(freq_layer1)
            freq_features2 = self.conv_out_layer2(freq_layer2)
            freq_features3 = self.conv_out_layer3(freq_layer3)
            freq_features4 = self.conv_out_layer4(freq_layer4)
            final_out = freq_features4

        # 分类处理
        final_out = self.avgpool(final_out)
        return self.fc(final_out.view(final_out.size(0), -1))


def build_integrated_model(mode='mixed', **kwargs):
    dim, size = [128, 256, 512], 196
    return DSHF_OCT_Integrated(
        in_channels=1,
        dim=dim,
        size=size,
        mode=mode,
        **kwargs
    )


def model_18(**kwargs):
    mixed_model = build_integrated_model(mode='mixed', **kwargs)
    return mixed_model


def model_18_wo_cross_encoder(**kwargs):
    mixed_model = build_integrated_model(mode='mixed_wo_cross_encoder', **kwargs)
    return mixed_model


def model_18_wo_cross_layer(**kwargs):
    mixed_model = build_integrated_model(mode='mixed_wo_cross_layer', **kwargs)
    return mixed_model


def model18_only_spa(**kwargs):
    spa_model = build_integrated_model(mode='spa', **kwargs)
    return spa_model


def model18_only_freq(pretrained=False, **kwargs):
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
    # 测试前向传播
    x = torch.randn(1, 1, 64, 64)

    # 创建不同模式模型
    mixed_model = model_18(num_classes=9)
    print("Mixed模型输出尺寸:", mixed_model(x).shape)
    print("模型输出尺寸:", mixed_model(x).shape)
    total = count_parameters(mixed_model)
    flops = count_flops(mixed_model, x)
    print("Number of parameter: %.2fM" % (total/1e6))
    print(flops)
    mixed_model_2 = model_18_wo_cross_encoder(num_classes=9)
    print("Mixed模型输出尺寸:", mixed_model_2(x).shape)
    mixed_model_3 = model_18_wo_cross_layer(num_classes=9)
    print("Mixed模型输出尺寸:", mixed_model_3(x).shape)
    # spa_model = model18_only_spa(num_classes=15)
    # freq_model = model18_only_freq(num_classes=11)
    x = torch.randn(1, 1, 64, 64)

    pass




    # print("Spa模型输出尺寸:", spa_model(x).shape)
    # print("Freq模型输出尺寸:", freq_model(x).shape)

    # # 参数统计
    # from torchinfo import summary
    # summary(mixed_model, input_size=(1, 1, 64, 64))
