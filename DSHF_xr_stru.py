import torch
import torch.nn as nn
import torch.fft as fft
from torch.nn import functional as F
from graph_module import channel_graph_interaction
from oct_conv import *
from fea_fusion import Cross_Encoder_Fusion, Cross_Layer_Fusion
from fea_fusion import (
    Cross_Encoder_Fusion_abla,
    MultiScaleFPNd2u,
    MultiScaleFPNu2d,
    madd,
    multi
)
from einops.layers.torch import Rearrange

from DSHF_vallina_conv import SpatialEnc, FrequencyEnc


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
                 base_width=64, norm_layer=None, First=False, freq_mode='original'):
        """
        Args:
            in_channels: 输入通道数（高低频之和）
            out_channels: 输出通道数（高低频之和）
            alpha: 八度卷积中低频通道比例
            First: 是否为第一个模块（将1通道输入转换为八度表示）
            mode: 处理模式，可选 'original', 'sincos', 'magnitude_only', 'phase_only'
        """
        super().__init__()
        self.first = First
        self.alpha = alpha
        self.mode = freq_mode

        # 计算高低频通道数
        self.c_h = int((1 - alpha) * in_channels)
        self.c_l = int(alpha * in_channels)
        self.out_c_h = int((1 - alpha) * out_channels)
        self.out_c_l = int(alpha * out_channels)

        if self.first:
            self.first_conv = FirstOctaveCBR(1, 8, kernel_size=(1, 1), alpha=self.alpha, stride=1, padding=0)

        # 幅值掩码（所有模式除 phase_only 和 angle_only 外均使用）
        if freq_mode != 'phase_only' and freq_mode != 'angle_only':
            self.amp_mask = nn.Sequential(
                FreqOctaveCR(in_channels, in_channels, kernel_size=(3, 3), alpha=self.alpha, stride=1, padding=1),
                OctaveConv(in_channels, in_channels, kernel_size=(3, 3), alpha=self.alpha, stride=1, padding=1)
            )

        # 相位掩码（根据模式选择通道数和是否创建）
        if freq_mode == 'original':
            # 直接处理相位角，通道数不变
            self.pha_mask = nn.Sequential(
                FreqOctaveCR(in_channels, in_channels, kernel_size=(3, 3), alpha=self.alpha, stride=1, padding=1),
                OctaveConv(in_channels, in_channels, kernel_size=(3, 3), alpha=self.alpha, stride=1, padding=1)
            )
        elif freq_mode in ['sincos', 'phase_only']:
            # 正弦-余弦空间处理，输入输出通道翻倍
            pha_in_channels = 2 * in_channels
            self.pha_mask = nn.Sequential(
                FreqOctaveCR(pha_in_channels, pha_in_channels, kernel_size=(3, 3), alpha=self.alpha, stride=1,
                             padding=1),
                OctaveConv(pha_in_channels, pha_in_channels, kernel_size=(3, 3), alpha=self.alpha, stride=1, padding=1)
            )
        # magnitude_only 模式无需相位掩码
        elif freq_mode == 'angle_only':
            pha_in_channels = in_channels
            self.pha_mask = nn.Sequential(
                FreqOctaveCR(pha_in_channels, pha_in_channels, kernel_size=(3, 3), alpha=self.alpha, stride=1,
                             padding=1),
                OctaveConv(pha_in_channels, pha_in_channels, kernel_size=(3, 3), alpha=self.alpha, stride=1, padding=1)
            )

        # 通道调整层（高低频分别）
        self.channel_adjust_h = nn.Conv2d(self.c_h, self.out_c_h, kernel_size=1)
        self.channel_adjust_l = nn.Conv2d(self.c_l, self.out_c_l, kernel_size=1)

        # 短路连接（处理残差，确保通道匹配）
        if self.c_h != self.out_c_h:
            self.shortcut_h = nn.Conv2d(self.c_h, self.out_c_h, kernel_size=1)
        else:
            self.shortcut_h = nn.Identity()
        if self.c_l != self.out_c_l:
            self.shortcut_l = nn.Conv2d(self.c_l, self.out_c_l, kernel_size=1)
        else:
            self.shortcut_l = nn.Identity()

    def forward(self, x):
        # ---------- 输入处理，得到高低频分支 ----------
        if isinstance(x, tuple):
            x_h, x_l = x[0], x[1]
        else:
            # 若为单张量，则通过 first_conv 转换为八度表示
            b, c, h, w = x.shape
            if c == 1 or c == 3:
                x_h, x_l = self.first_conv(x)

        # 保存残差连接的原始值（经过通道调整）
        identity_h = self.shortcut_h(x_h)
        identity_l = self.shortcut_l(x_l)

        b_h, c_h, h_h, w_h = x_h.shape
        b_l, c_l, h_l, w_l = x_l.shape

        # ---------- 频域变换 ----------
        msF_h = fft.rfft2(x_h + 1e-8, norm='backward')
        msF_l = fft.rfft2(x_l + 1e-8, norm='backward')

        msF_amp_h = torch.abs(msF_h)
        msF_amp_l = torch.abs(msF_l)
        msF_pha_h = torch.angle(msF_h)
        msF_pha_l = torch.angle(msF_l)

        # ---------- 根据模式处理 ----------
        if self.mode == 'original':
            # 原始模式：直接处理相位角
            # 幅值增强
            amp_fuse_h, amp_fuse_l = self.amp_mask((msF_amp_h, msF_amp_l))
            amp_fuse_h = amp_fuse_h + msF_amp_h
            amp_fuse_l = amp_fuse_l + msF_amp_l
            # 相位增强
            pha_fuse_h, pha_fuse_l = self.pha_mask((msF_pha_h, msF_pha_l))
            pha_fuse_h = pha_fuse_h + msF_pha_h
            pha_fuse_l = pha_fuse_l + msF_pha_l
            # 重建复数
            real_h = amp_fuse_h * torch.cos(pha_fuse_h)
            imag_h = amp_fuse_h * torch.sin(pha_fuse_h)
            real_l = amp_fuse_l * torch.cos(pha_fuse_l)
            imag_l = amp_fuse_l * torch.sin(pha_fuse_l)

        elif self.mode == 'sincos':
            # 正弦-余弦空间模式：相位用连续表示
            cos_h, sin_h = torch.cos(msF_pha_h), torch.sin(msF_pha_h)
            cos_l, sin_l = torch.cos(msF_pha_l), torch.sin(msF_pha_l)

            # 拼接成通道翻倍的张量
            pha_feat_h = torch.cat([cos_h, sin_h], dim=1)
            pha_feat_l = torch.cat([cos_l, sin_l], dim=1)

            # 通过相位掩码预测残差
            delta_h, delta_l = self.pha_mask((pha_feat_h, pha_feat_l))

            # 分离 Δcos 和 Δsin（前后各一半通道）
            c_h_half = c_h
            c_l_half = c_l
            delta_cos_h = delta_h[:, :c_h_half, :, :]
            delta_sin_h = delta_h[:, c_h_half:, :, :]
            delta_cos_l = delta_l[:, :c_l_half, :, :]
            delta_sin_l = delta_l[:, c_l_half:, :, :]

            # 融合并归一化到单位圆
            cos_fuse_h = cos_h + delta_cos_h
            sin_fuse_h = sin_h + delta_sin_h
            norm_h = torch.sqrt(cos_fuse_h ** 2 + sin_fuse_h ** 2 + 1e-8)
            cos_fuse_h = cos_fuse_h / norm_h
            sin_fuse_h = sin_fuse_h / norm_h

            cos_fuse_l = cos_l + delta_cos_l
            sin_fuse_l = sin_l + delta_sin_l
            norm_l = torch.sqrt(cos_fuse_l ** 2 + sin_fuse_l ** 2 + 1e-8)
            cos_fuse_l = cos_fuse_l / norm_l
            sin_fuse_l = sin_fuse_l / norm_l

            # 幅值增强
            amp_fuse_h, amp_fuse_l = self.amp_mask((msF_amp_h, msF_amp_l))
            amp_fuse_h = amp_fuse_h + msF_amp_h
            amp_fuse_l = amp_fuse_l + msF_amp_l

            # 重建复数
            real_h = amp_fuse_h * cos_fuse_h
            imag_h = amp_fuse_h * sin_fuse_h
            real_l = amp_fuse_l * cos_fuse_l
            imag_l = amp_fuse_l * sin_fuse_l

        elif self.mode == 'magnitude_only':
            # 仅幅度谱模式：相位固定为0
            amp_fuse_h, amp_fuse_l = self.amp_mask((msF_amp_h, msF_amp_l))
            amp_fuse_h = amp_fuse_h + msF_amp_h
            amp_fuse_l = amp_fuse_l + msF_amp_l
            real_h, imag_h = amp_fuse_h, torch.zeros_like(amp_fuse_h)
            real_l, imag_l = amp_fuse_l, torch.zeros_like(amp_fuse_l)

        elif self.mode == 'phase_only':
            # 仅相位谱模式：幅度固定为1
            cos_h, sin_h = torch.cos(msF_pha_h), torch.sin(msF_pha_h)
            cos_l, sin_l = torch.cos(msF_pha_l), torch.sin(msF_pha_l)
            pha_feat_h = torch.cat([cos_h, sin_h], dim=1)
            pha_feat_l = torch.cat([cos_l, sin_l], dim=1)
            delta_h, delta_l = self.pha_mask((pha_feat_h, pha_feat_l))

            delta_cos_h = delta_h[:, :c_h, :, :]
            delta_sin_h = delta_h[:, c_h:, :, :]
            delta_cos_l = delta_l[:, :c_l, :, :]
            delta_sin_l = delta_l[:, c_l:, :, :]

            cos_fuse_h = cos_h + delta_cos_h
            sin_fuse_h = sin_h + delta_sin_h
            norm_h = torch.sqrt(cos_fuse_h ** 2 + sin_fuse_h ** 2 + 1e-8)
            cos_fuse_h = cos_fuse_h / norm_h
            sin_fuse_h = sin_fuse_h / norm_h

            cos_fuse_l = cos_l + delta_cos_l
            sin_fuse_l = sin_l + delta_sin_l
            norm_l = torch.sqrt(cos_fuse_l ** 2 + sin_fuse_l ** 2 + 1e-8)
            cos_fuse_l = cos_fuse_l / norm_l
            sin_fuse_l = sin_fuse_l / norm_l

            # 幅度固定为1
            real_h, imag_h = cos_fuse_h, sin_fuse_h
            real_l, imag_l = cos_fuse_l, sin_fuse_l
        elif self.mode == 'angle_only':
            # 仅相位角度模式
            pha_fuse_h, pha_fuse_l = self.pha_mask((msF_pha_h, msF_pha_l))
            pha_fuse_h = pha_fuse_h + msF_pha_h
            pha_fuse_l = pha_fuse_l + msF_pha_l

            real_h = torch.cos(pha_fuse_h)
            imag_h = torch.sin(pha_fuse_h)
            real_l = torch.cos(pha_fuse_l)
            imag_l = torch.sin(pha_fuse_l)

        # ---------- 逆变换 ----------
        out_h = fft.irfft2(torch.complex(real_h, imag_h), s=(h_h, w_h), norm='backward').real
        out_l = fft.irfft2(torch.complex(real_l, imag_l), s=(h_l, w_l), norm='backward').real

        # 残差连接 + 通道调整
        out_h = self.channel_adjust_h(out_h) + identity_h
        out_l = self.channel_adjust_l(out_l) + identity_l

        # 数值稳定
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


# original,sincos,magnitude_only,phase_only
class FrequencyOCTEnc(nn.Module):
    def __init__(self, in_channels, alpha, freq_mode='', ):
        super(FrequencyOCTEnc, self).__init__()
        self.dounsample = nn.MaxPool2d(2)
        self.enc_blk1 = FrequencyOCTFilter(in_channels * 8, out_channels=64, alpha=alpha, freq_mode=freq_mode,
                                           First=True)
        self.enc_blk2 = FrequencyOCTFilter(in_channels=64, out_channels=128, alpha=alpha, freq_mode=freq_mode, )
        self.enc_blk3 = FrequencyOCTFilter(in_channels=128, out_channels=256, alpha=alpha, freq_mode=freq_mode, )
        self.enc_blk4 = FrequencyOCTFilter(in_channels=256, out_channels=512, alpha=alpha, freq_mode=freq_mode, )

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


# original,sincos,magnitude_only,phase_only
class DSHF_OCT_Integrated(nn.Module):
    def __init__(self, in_channels, dim, size, num_classes=38, mode='mixed',
                 freq_mode='original', alpha=0.5, norm_layer=None,
                 sffm_mode='base', clfm_type='default', attenmode='base', ):
        """
        Args:
            sffm_mode: SFFM 模式，可选 'base', 'mixed_eadd', 'mixed_emul',
                      'mixed_eatten', 'mixed_add' 等（对应 Cross_Encoder_Fusion_abla）
            clfm_type: CLFM 类型，可选 'default', 'add', 'up', 'down', 'fpn_u2d', 'fpn_d2u', 'madd', 'multi'
        """
        super().__init__()
        """
        - mode='mixed' : 同时使用空间和频率编码 (默认)
        - mode='spa'   : 仅使用空间编码
        - mode='freq'  : 仅使用频率编码
        """
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.mode = mode
        self.sffm_mode = sffm_mode
        self.clfm_type = clfm_type
        self.attenmode = attenmode
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
            self.frequency_enc = FrequencyOCTEnc(in_channels, alpha=alpha, freq_mode=freq_mode, )

        self.cross_layer_fusion = Cross_Layer_Fusion(dim, size)

        # 根据模式创建 SFFM 模块（Cross_Encoder_Fusion_abla）
        if self.mode not in ['spa', 'freq', 'mixed_wo_cross_encoder', ]:
            self.cross_encoder_fusion2 = Cross_Encoder_Fusion_abla(128, attenmode=self.attenmode,
                                                                   addmode=self.sffm_mode)
            self.cross_encoder_fusion3 = Cross_Encoder_Fusion_abla(256, attenmode=self.attenmode,
                                                                   addmode=self.sffm_mode)
            self.cross_encoder_fusion4 = Cross_Encoder_Fusion_abla(512, attenmode=self.attenmode,
                                                                   addmode=self.sffm_mode)
        self.fc = nn.Linear(256, num_classes)
        if clfm_type == 'base':
            self.cross_layer_fusion = Cross_Layer_Fusion(dim, size)

        elif clfm_type == 'fpn_u2d':  # 低分辨率分支输出（256通道）
            self.cross_layer_fusion = MultiScaleFPNu2d()
            self.fc = nn.Linear(512, num_classes)
        elif clfm_type == 'fpn_d2u':  # 高分辨率分支输出（64通道）
            self.cross_layer_fusion = MultiScaleFPNd2u()
            self.fc = nn.Linear(128, num_classes)
        elif clfm_type == 'madd':
            # 需要为 madd 指定三个卷积层，这里简单用 1x1 卷积对齐通道
            conv1 = nn.Conv2d(128, 512, 1)
            conv2 = nn.Conv2d(256, 512, 1)
            conv3 = nn.Conv2d(512, 512, 1)
            self.cross_layer_fusion = madd(conv1, conv2, conv3)
            self.fc = nn.Linear(512, num_classes)
        elif clfm_type == 'multi':
            conv1 = nn.Conv2d(128, 512, 1)
            conv2 = nn.Conv2d(256, 512, 1)
            conv3 = nn.Conv2d(512, 512, 1)
            self.cross_layer_fusion = multi(conv1, conv2, conv3)
            self.fc = nn.Linear(512, num_classes)
        else:
            raise ValueError(f"Unknown clfm_type: {clfm_type}")

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

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


# original,sincos,magnitude_only,phase_only
def build_integrated_model(mode='mixed', freq_mode='magnitude_only', sffm_mode='base', clfm_type='base',
                           attenmode='base', **kwargs):
    dim, size = [128, 256, 512], 196
    return DSHF_OCT_Integrated(
        in_channels=1,
        dim=dim,
        size=size,
        mode=mode,
        freq_mode=freq_mode,
        sffm_mode=sffm_mode,
        clfm_type=clfm_type,
        attenmode=attenmode,
        **kwargs
    )


class DSHF_Mixed_OCT_SPA_Conv_FREQ(nn.Module):
    def __init__(self, in_channels, dim, size, num_classes=38, freq_mode='original', alpha=0.5):
        super().__init__()
        self.spatial_enc = SpatialOCTEnc(in_channels, alpha=alpha)  # 八度空间编码
        self.frequency_enc = FrequencyEnc(in_channels, mode=freq_mode)  # 普通频率编码
        self.conv_out_layer1 = LastOctaveConv(64, 64, kernel_size=(1, 1), alpha=alpha, stride=1, padding=0, groups=1)
        self.conv_out_layer2 = LastOctaveConv(128, 128, kernel_size=(1, 1), alpha=alpha, stride=1, padding=0, groups=1)
        self.conv_out_layer3 = LastOctaveConv(256, 256, kernel_size=(1, 1), alpha=alpha, stride=1, padding=0, groups=1)
        self.conv_out_layer4 = LastOctaveConv(512, 512, kernel_size=(1, 1), alpha=alpha, stride=1, padding=0, groups=1)
        self.cross_layer_fusion = Cross_Layer_Fusion(dim, size)
        self.cross_encoder_fusion2 = Cross_Encoder_Fusion(128)
        self.cross_encoder_fusion3 = Cross_Encoder_Fusion(256)
        self.cross_encoder_fusion4 = Cross_Encoder_Fusion(512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # 空间编码返回 (x0,x1,x2,x3,x4)，取后三个（通道128,256,512）
        spa0, spa1, spa2, spa3, spa4 = self.spatial_enc(x)
        # spa1 = self.conv_out_layer1(spa1)
        spa2 = self.conv_out_layer2(spa2)
        spa3 = self.conv_out_layer3(spa3)
        spa4 = self.conv_out_layer4(spa4)
        # 频率编码返回类似
        freq0, freq1, freq2, freq3, freq4 = self.frequency_enc(x)
        out2 = self.cross_encoder_fusion2(spa2, freq2)
        out3 = self.cross_encoder_fusion3(spa3, freq3)
        out4 = self.cross_encoder_fusion4(spa4, freq4)
        final_out = self.cross_layer_fusion(out2, out3, out4)
        final_out = self.avgpool(final_out)
        return self.fc(final_out.view(final_out.size(0), -1))


class DSHF_Mixed_Conv_SPA_OCT_FREQ(nn.Module):
    def __init__(self, in_channels, dim, size, num_classes=38, freq_mode='original', alpha=0.5):
        super().__init__()
        self.spatial_enc = SpatialEnc(in_channels)  # 普通空间编码
        self.frequency_enc = FrequencyOCTEnc(in_channels, alpha=alpha, freq_mode=freq_mode)  # 八度频率编码
        self.cross_layer_fusion = Cross_Layer_Fusion(dim, size)
        self.conv_out_layer1 = LastOctaveConv(64, 64, kernel_size=(1, 1), alpha=alpha, stride=1, padding=0, groups=1)
        self.conv_out_layer2 = LastOctaveConv(128, 128, kernel_size=(1, 1), alpha=alpha, stride=1, padding=0, groups=1)
        self.conv_out_layer3 = LastOctaveConv(256, 256, kernel_size=(1, 1), alpha=alpha, stride=1, padding=0, groups=1)
        self.conv_out_layer4 = LastOctaveConv(512, 512, kernel_size=(1, 1), alpha=alpha, stride=1, padding=0, groups=1)
        self.cross_encoder_fusion2 = Cross_Encoder_Fusion(128)
        self.cross_encoder_fusion3 = Cross_Encoder_Fusion(256)
        self.cross_encoder_fusion4 = Cross_Encoder_Fusion(512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        spa0, spa1, spa2, spa3, spa4 = self.spatial_enc(x)
        freq0, freq1, freq2, freq3, freq4 = self.frequency_enc(x)
        freq2 = self.conv_out_layer2(freq2)
        freq3 = self.conv_out_layer3(freq3)
        freq4 = self.conv_out_layer4(freq4)
        out2 = self.cross_encoder_fusion2(spa2, freq2)
        out3 = self.cross_encoder_fusion3(spa3, freq3)
        out4 = self.cross_encoder_fusion4(spa4, freq4)
        final_out = self.cross_layer_fusion(out2, out3, out4)
        final_out = self.avgpool(final_out)
        return self.fc(final_out.view(final_out.size(0), -1))


class DSHF_Mixed__VLI_Conv(nn.Module):
    def __init__(self, in_channels, dim, size, num_classes=38, freq_mode='original', alpha=0.5):
        super().__init__()
        self.spatial_enc = SpatialEnc(in_channels)  # 普通空间编码
        self.frequency_enc = FrequencyEnc(in_channels, mode=freq_mode)  # 八度频率编码
        self.cross_layer_fusion = Cross_Layer_Fusion(dim, size)

        self.cross_encoder_fusion2 = Cross_Encoder_Fusion(128)
        self.cross_encoder_fusion3 = Cross_Encoder_Fusion(256)
        self.cross_encoder_fusion4 = Cross_Encoder_Fusion(512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        spa0, spa1, spa2, spa3, spa4 = self.spatial_enc(x)
        freq0, freq1, freq2, freq3, freq4 = self.frequency_enc(x)

        out2 = self.cross_encoder_fusion2(spa2, freq2)
        out3 = self.cross_encoder_fusion3(spa3, freq3)
        out4 = self.cross_encoder_fusion4(spa4, freq4)
        final_out = self.cross_layer_fusion(out2, out3, out4)
        final_out = self.avgpool(final_out)
        return self.fc(final_out.view(final_out.size(0), -1))


def SFFNet_wo_cross_encoder(**kwargs):
    mixed_model = build_integrated_model(mode='mixed_wo_cross_encoder', **kwargs)
    return mixed_model


def SFFNet_wo_cross_layer(**kwargs):
    mixed_model = build_integrated_model(mode='mixed_wo_cross_layer', **kwargs)
    return mixed_model


def SFFNet_conv_spa_oct_freq(num_classes=38, alpha=0.5, freq_mode='magnitude_only'):
    return DSHF_Mixed_Conv_SPA_OCT_FREQ(
        in_channels=1,
        dim=[128, 256, 512],
        size=196,
        num_classes=num_classes,
        freq_mode=freq_mode,
        alpha=alpha
    )


def SFFNet_vil_conv(num_classes=38, alpha=0.5, freq_mode='magnitude_only'):
    return DSHF_Mixed__VLI_Conv(
        in_channels=1,
        dim=[128, 256, 512],
        size=196,
        num_classes=num_classes,
        freq_mode=freq_mode,
        alpha=alpha
    )


def SFFNet_oct_spa_conv_freq(num_classes=38, alpha=0.5, freq_mode='magnitude_only'):
    return DSHF_Mixed_OCT_SPA_Conv_FREQ(
        in_channels=1,
        dim=[128, 256, 512],
        size=196,
        num_classes=num_classes,
        freq_mode=freq_mode,
        alpha=alpha
    )


def SFFNet(**kwargs):
    mixed_model = build_integrated_model(mode='mixed', **kwargs)
    return mixed_model


# todo sffm有效性
def SFFNet_sffm_add(**kwargs):
    mixed_model = build_integrated_model(sffm_mode='mixed_add', **kwargs)
    return mixed_model


def SFFNet_sffm_mul(**kwargs):
    mixed_model = build_integrated_model(sffm_mode='mixed_emul', **kwargs)
    return mixed_model


def SFFNet_sffm_crossatten(**kwargs):
    mixed_model = build_integrated_model(sffm_mode='mixed_eatten', attenmode='mixed_eatten', **kwargs)
    return mixed_model


# todo sffm注意力设置结构有效性


def SFFNet_sffmatten_cs(**kwargs):
    mixed_model = build_integrated_model(attenmode='mixed_casa', **kwargs)
    return mixed_model


def SFFNet_sffmatten_cc(**kwargs):
    mixed_model = build_integrated_model(attenmode='mixed_caca', **kwargs)
    return mixed_model


def SFFNet_sffmatten_ss(**kwargs):
    mixed_model = build_integrated_model(attenmode='mixed_sasa', **kwargs)
    return mixed_model


# todo 跨层结构有效性

def SFFNet_clfm_add(**kwargs):
    mixed_model = build_integrated_model(clfm_type='madd', **kwargs)
    return mixed_model


def SFFNet_clfm_mul(**kwargs):
    mixed_model = build_integrated_model(clfm_type='multi', **kwargs)
    return mixed_model


def SFFNet_clfm_fpnd2u(**kwargs):
    mixed_model = build_integrated_model(clfm_type='fpn_d2u', **kwargs)
    return mixed_model


def SFFNet_clfm_fpnu2d(**kwargs):
    mixed_model = build_integrated_model(clfm_type='fpn_u2d', **kwargs)
    return mixed_model


# original,sincos,magnitude_only,phase_only

def model18_only_freq_or(pretrained=False, **kwargs):
    freq_model = build_integrated_model(mode='freq', freq_mode='original', **kwargs)
    return freq_model


def model18_only_freq_am(pretrained=False, **kwargs):
    freq_model = build_integrated_model(mode='freq', freq_mode='magnitude_only', **kwargs)
    return freq_model


def model18_only_freq_ph(pretrained=False, **kwargs):
    freq_model = build_integrated_model(mode='freq', freq_mode='phase_only', **kwargs)
    return freq_model


def model18_only_freq_an(pretrained=False, **kwargs):
    freq_model = build_integrated_model(mode='freq', freq_mode='angle_only', **kwargs)
    return freq_model


def model18_only_freq_sc(pretrained=False, **kwargs):
    freq_model = build_integrated_model(mode='freq', freq_mode='sincos', **kwargs)
    return freq_model


def SFFNet_freq_ori(pretrained=False, **kwargs):
    model = build_integrated_model(mode='mixed', freq_mode='original', **kwargs)
    return model


def SFFNet_freq_amp(pretrained=False, **kwargs):
    model = build_integrated_model(mode='mixed', freq_mode='magnitude_only', **kwargs)
    return model


def SFFNet_freq_pha(pretrained=False, **kwargs):
    model = build_integrated_model(mode='mixed', freq_mode='phase_only', **kwargs)
    return model


def SFFNet_freq_ang(pretrained=False, **kwargs):
    model = build_integrated_model(mode='mixed', freq_mode='angle_only', **kwargs)
    return model


def SFFNet_freq_sc(pretrained=False, **kwargs):
    model = build_integrated_model(mode='mixed', freq_mode='sincos', **kwargs)
    return model


from thop import profile
from thop import clever_format


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_flops(model, input):
    flops = profile(model, inputs=(input,), verbose=False)[0] * 2
    flops = clever_format(flops, "%.10f")
    return flops


if __name__ == '__main__':
    # 测试前向传播
    x = torch.randn(1, 1, 64, 64)

    # 创建不同模式模型
    # mixed_model = model_18(num_classes=9)
    freq_model1 = model18_only_freq_or(num_classes=9)
    freq_model2 = model18_only_freq_am(num_classes=9)
    freq_model3 = model18_only_freq_ph(num_classes=9)
    freq_model4 = model18_only_freq_sc(num_classes=9)
    model_factories = [

        # todo 跨编码器融合结构
        # lambda: SFFNet_sffm_add(num_classes=38, alpha=0.5),
        # lambda: SFFNet_sffm_mul(num_classes=38, alpha=0.5),
        # lambda: SFFNet_sffm_crossatten(num_classes=38, alpha=0.5),

        # todo SFFM注意力设置
        lambda: SFFNet_sffmatten_cs(num_classes=38, alpha=0.5),
        # lambda: SFFNet_sffmatten_cc(num_classes=38, alpha=0.5),
        # lambda: SFFNet_sffmatten_ss(num_classes=38, alpha=0.5),
        #
        # # todo 跨层融合结构
        # lambda: SFFNet_clfm_add(num_classes=38, alpha=0.5),
        # lambda: SFFNet_clfm_mul(num_classes=38, alpha=0.5),
        # lambda: SFFNet_clfm_fpnd2u(num_classes=38, alpha=0.5),
        # lambda: SFFNet_clfm_fpnu2d(num_classes=38, alpha=0.5),

    ]
    for model_factory in model_factories:
        x = torch.randn(1, 1, 64, 64)
        mixed_model = model_factory()
        print("Mixed模型输出尺寸:", mixed_model(x).shape)
        print("模型输出尺寸:", mixed_model(x).shape)
        total = count_parameters(mixed_model)
        flops = count_flops(mixed_model, x)
        print("Number of parameter: %.2fM" % (total / 1e6))
        print(flops)

        print("模型输出尺寸:", mixed_model(x).shape)
