import torch
import torch.nn as nn
import torch.fft as fft
from torch.nn import functional as F
from graph_module import channel_graph_interaction
from oct_conv import *
from fea_fusion import Cross_Encoder_Fusion, Cross_Layer_Fusion, AttentionPool2d, Cross_Atten_Encoder_Fusion
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


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 conv with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(3, 3), stride=stride,
                     padding=1, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 conv"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1, 1), stride=stride, bias=False, padding=0)


class DoubleOCTConv(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None, First=False):
        super(DoubleOCTConv, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.first = First
        if self.first:
            self.ocb1 = FirstOctaveCBR(inplanes, width, kernel_size=(3, 3), norm_layer=norm_layer, padding=1)
        else:
            self.ocb1 = OctaveCBR(inplanes, width, kernel_size=(3, 3), norm_layer=norm_layer, padding=1)

        self.ocb2 = OctaveCB(width, planes * self.expansion, kernel_size=(3, 3), stride=stride, groups=groups,
                             norm_layer=norm_layer, padding=1)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # try:
        #     print('input',x[0].shape,x[1].shape)
        # except:
        #     print('input',x.shape)

        if self.first:
            x_h_res, x_l_res = self.ocb1(x)
            # print('first_1',x_h_res.shape, x_l_res.shape)
            x_h, x_l = self.ocb2((x_h_res, x_l_res))
            # print('first_2',x_h.shape, x_l.shape)
        else:
            x_h_res, x_l_res = x
            x_h, x_l = self.ocb1((x_h_res, x_l_res))
            # print('conn1', x_h.shape, x_l.shape)
            x_h, x_l = self.ocb2((x_h, x_l))
            # print('conv2', x_h.shape, x_l.shape)

        if self.downsample is not None:
            # print('before downsample', x_h_res.shape, x_l_res.shape)
            x_h_res, x_l_res = self.downsample((x_h_res, x_l_res))
            # print('downsample', x_h_res.shape, x_l_res.shape)
        x_h += x_h_res
        x_l += x_l_res

        x_h = self.relu(x_h)
        x_l = self.relu(x_l)
        # print('output',x_h.shape, x_l.shape)
        # print('double block down')
        return x_h, x_l


class FrequencyOCTFilter(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None, First=False):
        super().__init__()
        self.first = First
        self.alpha = 0.5
        self.first_conv = FirstOctaveCBR(1, 64, kernel_size=(1, 1), alpha=self.alpha, stride=1, padding=0)

        self.amp_mask = nn.Sequential(
            FreqOctaveCBR(in_channels, in_channels, kernel_size=(3, 3), alpha=self.alpha, stride=1, padding=1),
            FreqOctaveCBR(in_channels, in_channels, kernel_size=(3, 3), alpha=self.alpha, stride=1, padding=1)
        )
        self.pha_mask = nn.Sequential(
            FreqOctaveCBR(in_channels, in_channels, kernel_size=(3, 3), alpha=self.alpha, stride=1, padding=1),
            FreqOctaveCBR(in_channels, in_channels, kernel_size=(3, 3), alpha=self.alpha, stride=1, padding=1)
        )

        self.channel_adjust_h = nn.Conv2d(int(self.alpha * in_channels), int(self.alpha * out_channels), kernel_size=1)
        self.channel_adjust_l = nn.Conv2d(int((1 - self.alpha) * in_channels), int((1 - self.alpha) * out_channels),
                                          kernel_size=1)

    def forward(self, x):
        # print(len(x))
        if isinstance(x, tuple):
            # 如果 x 是一个 tuple，则直接解包
            x_h, x_l = x
        else:
            b, c, h, w = x.shape
            # print(b,c,h,w)
            if c == 1 or c == 3:
                x_h, x_l = self.first_conv(x)
                # print('first', x_h.shape, x_l.shape)
        # else:
        #     x_h, x_l = x

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


class BottleneckLast(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None):
        super(BottleneckLast, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Last means the end of two branch
        self.ocb1 = OctaveCBR(inplanes, width, kernel_size=(3, 3), padding=1)
        self.ocb2 = OctaveCB(width, planes * self.expansion, kernel_size=(3, 3), padding=1, stride=stride,
                             groups=groups, norm_layer=norm_layer)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.stride = stride

    def forward(self, x):

        x_h_res, x_l_res = x
        x_h, x_l = self.ocb1((x_h_res, x_l_res))

        x_h, x_l = self.ocb2((x_h, x_l))

        if self.downsample is not None:
            x_h_res = self.downsample((x_h_res, x_l_res))
        x_l = self.upsample(x_l)
        x_h = torch.cat((x_h, x_l), dim=1)
        x_h += x_h_res
        x_h = self.relu(x_h)

        return x_h


class BottleneckOrigin(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None):
        super(BottleneckOrigin, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups)
        self.bn2 = norm_layer(width)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class mymodel(nn.Module):

    def __init__(self, block1, block2, layers, dim, size, num_classes=38, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):
        super(mymodel, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.spa_inplanes = 1
        self.freq_inplanes = 64
        self.alpha = 0.5
        self.groups = groups
        self.base_width = width_per_group
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.cross_encoder_fusion1 = Cross_Encoder_Fusion(64)
        self.cross_encoder_fusion2 = Cross_Encoder_Fusion(128)
        self.cross_encoder_fusion3 = Cross_Encoder_Fusion(256)
        self.cross_encoder_fusion4 = Cross_Encoder_Fusion(512)
        self.cross_layer_fusion = Cross_Layer_Fusion(dim, size)
        self.spa_layer1 = self._make_spa_layer(block1, 64, layers[0], stride=2, norm_layer=norm_layer,
                                               First=True)
        self.freq_layer1 = self._make_freq_layer(block2, 64, layers[0], stride=1, norm_layer=norm_layer,
                                                 First=True)
        self.conv_out_layer1 = out_OCtaveConv(64, 64, kernel_size=(1, 1), stride=1, padding=0,  ##############
                                              groups=1, norm_layer=norm_layer)

        self.spa_layer2 = self._make_spa_layer(block1, 128, layers[1], stride=2, norm_layer=norm_layer)
        self.freq_layer2 = self._make_freq_layer(block2, 128, layers[1], stride=1, norm_layer=norm_layer)

        self.conv_out_layer2 = out_OCtaveConv(128, 128, kernel_size=(1, 1), alpha=self.alpha, stride=1, padding=0,
                                              ##############
                                              groups=1, norm_layer=norm_layer)

        self.spa_layer3 = self._make_spa_layer(block1, 256, layers[2], stride=2, norm_layer=norm_layer)
        self.freq_layer3 = self._make_freq_layer(block2, 256, layers[2], stride=1, norm_layer=norm_layer)
        # print(self.freq_layer3)

        self.conv_out_layer3 = out_OCtaveConv(256, 256, kernel_size=(1, 1), stride=1, padding=0,  ##############
                                              groups=1, norm_layer=norm_layer)
        self.spa_layer4 = self._make_spa_layer(block1, 512, layers[3], stride=2, norm_layer=norm_layer)
        self.freq_layer4 = self._make_freq_layer(block2, 512, layers[3], stride=1, norm_layer=norm_layer)
        self.conv_out_layer4 = out_OCtaveConv(512, 512, kernel_size=(1, 1), stride=1, padding=0,  ##############
                                              groups=1, norm_layer=norm_layer)


        self.avgpoolatten = nn.AdaptiveAvgPool2d(4)
        self.pool = nn.AvgPool2d(2, 1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(256, num_classes)
        # self.head = nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, DoubleOCTConv):
                    nn.init.constant_(m.bn2.weight, 0)
                elif isinstance(m, DoubleConv):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_spa_layer(self, block, planes, blocks, stride=2, norm_layer=None, First=False):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.spa_inplanes != planes * block.expansion:
            if self.spa_inplanes == 1 or self.spa_inplanes == 3:
                downsample = nn.Sequential(
                    OctaveCB(in_channels=planes, out_channels=planes * block.expansion, kernel_size=(1, 1),
                             stride=stride, padding=0)
                )  # todo in_channels=self.inplanes -->in_channels=planes
            else:
                downsample = nn.Sequential(
                    OctaveCB(in_channels=self.spa_inplanes, out_channels=planes * block.expansion, kernel_size=(1, 1),
                             stride=stride, padding=0)
                )

        layers = []
        layers.append(block(self.spa_inplanes, planes, stride, downsample, self.groups,
                            self.base_width, norm_layer, First))
        self.spa_inplanes = planes * block.expansion
        # print('input channel change to ',self.inplanes)
        for _ in range(1, blocks):
            layers.append(block(self.spa_inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _make_freq_layer(self, block, planes, blocks, stride=2, norm_layer=None, First=False):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.freq_inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                OctaveCB(in_channels=self.freq_inplanes, out_channels=planes * block.expansion, kernel_size=(1, 1),
                         stride=stride, padding=0)
            )  # todo in_channels=self.inplanes -->in_channels=planes

        layers = []
        layers.append(block(self.freq_inplanes, planes, stride, downsample, self.groups,
                            self.base_width, norm_layer, First))
        self.freq_inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.freq_inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):

        x_spa_h1, x_spa_l1 = self.spa_layer1(x)
        x_freq_h1, x_freq_l1 = self.freq_layer1(x)

        x_freq_h1_down, x_freq_l1_down = self.maxpool(x_freq_h1), self.maxpool(x_freq_l1)

        x_spa_h2, x_spa_l2 = self.spa_layer2((x_spa_h1, x_spa_l1))

        x_freq_h2, x_freq_l2 = self.freq_layer2((x_freq_h1_down, x_freq_l1_down))

        x_freq_h2_down, x_freq_l2_down = self.maxpool(x_freq_h2), self.maxpool(x_freq_l2)

        x_spa_h3, x_spa_l3 = self.spa_layer3((x_spa_h2, x_spa_l2))

        x_freq_h3, x_freq_l3 = self.freq_layer3((x_freq_h2_down, x_freq_l2_down))

        x_freq_h3_down, x_freq_l3_down = self.maxpool(x_freq_h3), self.maxpool(x_freq_l3)

        x_spa_h4, x_spa_l4 = self.spa_layer4((x_spa_h3, x_spa_l3))

        x_freq_h4, x_freq_l4 = self.freq_layer4((x_freq_h3_down, x_freq_l3_down))

        x_freq_h4_down, x_freq_l4_down = self.maxpool(x_freq_h4), self.maxpool(x_freq_l4)

        x_spa_2, x_spa_3, x_spa_4 = \
            self.conv_out_layer2((x_spa_h2, x_spa_l2)), \
            self.conv_out_layer3((x_spa_h3, x_spa_l3)), \
            self.conv_out_layer4((x_spa_h4, x_spa_l4))

        x_freq_2, x_freq_3, x_freq_4 = \
            self.conv_out_layer2((x_freq_h2_down, x_freq_l2_down)), \
            self.conv_out_layer3((x_freq_h3_down, x_freq_l3_down)), \
            self.conv_out_layer4((x_freq_h4_down, x_freq_l4_down))


        out_layer_2 = self.cross_encoder_fusion2(x_spa_2, x_freq_2)
        out_layer_3 = self.cross_encoder_fusion3(x_spa_3, x_freq_3)
        out_layer_4 = self.cross_encoder_fusion4(x_spa_4, x_freq_4)



        out_x = self.cross_layer_fusion(out_layer_2, out_layer_3, out_layer_4)
        # intermediates = {
        #     'spa': [x_spa_2, x_spa_3, x_spa_4],
        #     'freq': [x_freq_2, x_freq_3, x_freq_4]
        #     'out_layer': [out_layer_2, out_layer_3, out_layer_4]
        #       'final_result':[out_x]
        # }
        # print(out_x.shape)
        # out_x = Rearrange('b (h w) c -> b c h w', h=16, w=16)(out_x)
        out_x = self.avgpool(out_x)
        out_x = out_x.view(out_x.size(0), -1)
        out_x = self.fc(out_x)

        return out_x



class mymodel_add(nn.Module):

    def __init__(self, block1, block2, layers, dim, size, num_classes=38, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):
        super(mymodel_add, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.spa_inplanes = 1
        self.freq_inplanes = 64
        self.alpha = 0.5
        self.groups = groups
        self.base_width = width_per_group
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.cross_encoder_fusion1 = Cross_Encoder_Fusion(64)
        # self.cross_encoder_fusion2 = Cross_Encoder_Fusion(128)
        # self.cross_encoder_fusion3 = Cross_Encoder_Fusion(256)
        # self.cross_encoder_fusion4 = Cross_Encoder_Fusion(512)
        self.cross_layer_fusion = Cross_Layer_Fusion(dim, size)
        self.spa_layer1 = self._make_spa_layer(block1, 64, layers[0], stride=2, norm_layer=norm_layer,
                                               First=True)
        self.freq_layer1 = self._make_freq_layer(block2, 64, layers[0], stride=1, norm_layer=norm_layer,
                                                 First=True)
        self.conv_out_layer1 = out_OCtaveConv(64, 64, kernel_size=(1, 1), stride=1, padding=0,  ##############
                                              groups=1, norm_layer=norm_layer)

        self.spa_layer2 = self._make_spa_layer(block1, 128, layers[1], stride=2, norm_layer=norm_layer)
        self.freq_layer2 = self._make_freq_layer(block2, 128, layers[1], stride=1, norm_layer=norm_layer)

        self.conv_out_layer2 = out_OCtaveConv(128, 128, kernel_size=(1, 1), alpha=self.alpha, stride=1, padding=0,
                                              ##############
                                              groups=1, norm_layer=norm_layer)

        self.spa_layer3 = self._make_spa_layer(block1, 256, layers[2], stride=2, norm_layer=norm_layer)
        self.freq_layer3 = self._make_freq_layer(block2, 256, layers[2], stride=1, norm_layer=norm_layer)
        # print(self.freq_layer3)

        self.conv_out_layer3 = out_OCtaveConv(256, 256, kernel_size=(1, 1), stride=1, padding=0,  ##############
                                              groups=1, norm_layer=norm_layer)
        self.spa_layer4 = self._make_spa_layer(block1, 512, layers[3], stride=2, norm_layer=norm_layer)
        self.freq_layer4 = self._make_freq_layer(block2, 512, layers[3], stride=1, norm_layer=norm_layer)
        self.conv_out_layer4 = out_OCtaveConv(512, 512, kernel_size=(1, 1), stride=1, padding=0,  ##############
                                              groups=1, norm_layer=norm_layer)


        self.avgpoolatten = nn.AdaptiveAvgPool2d(4)
        self.pool = nn.AvgPool2d(2, 1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(256, num_classes)
        # self.head = nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, DoubleOCTConv):
                    nn.init.constant_(m.bn2.weight, 0)
                elif isinstance(m, DoubleConv):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_spa_layer(self, block, planes, blocks, stride=2, norm_layer=None, First=False):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.spa_inplanes != planes * block.expansion:
            if self.spa_inplanes == 1 or self.spa_inplanes == 3:
                downsample = nn.Sequential(
                    OctaveCB(in_channels=planes, out_channels=planes * block.expansion, kernel_size=(1, 1),
                             stride=stride, padding=0)
                )  # todo in_channels=self.inplanes -->in_channels=planes
            else:
                downsample = nn.Sequential(
                    OctaveCB(in_channels=self.spa_inplanes, out_channels=planes * block.expansion, kernel_size=(1, 1),
                             stride=stride, padding=0)
                )

        layers = []
        layers.append(block(self.spa_inplanes, planes, stride, downsample, self.groups,
                            self.base_width, norm_layer, First))
        self.spa_inplanes = planes * block.expansion
        # print('input channel change to ',self.inplanes)
        for _ in range(1, blocks):
            layers.append(block(self.spa_inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _make_freq_layer(self, block, planes, blocks, stride=2, norm_layer=None, First=False):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.freq_inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                OctaveCB(in_channels=self.freq_inplanes, out_channels=planes * block.expansion, kernel_size=(1, 1),
                         stride=stride, padding=0)
            )  # todo in_channels=self.inplanes -->in_channels=planes

        layers = []
        layers.append(block(self.freq_inplanes, planes, stride, downsample, self.groups,
                            self.base_width, norm_layer, First))
        self.freq_inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.freq_inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):

        x_spa_h1, x_spa_l1 = self.spa_layer1(x)
        x_freq_h1, x_freq_l1 = self.freq_layer1(x)

        x_freq_h1_down, x_freq_l1_down = self.maxpool(x_freq_h1), self.maxpool(x_freq_l1)

        x_spa_h2, x_spa_l2 = self.spa_layer2((x_spa_h1, x_spa_l1))

        x_freq_h2, x_freq_l2 = self.freq_layer2((x_freq_h1_down, x_freq_l1_down))

        x_freq_h2_down, x_freq_l2_down = self.maxpool(x_freq_h2), self.maxpool(x_freq_l2)

        x_spa_h3, x_spa_l3 = self.spa_layer3((x_spa_h2, x_spa_l2))

        x_freq_h3, x_freq_l3 = self.freq_layer3((x_freq_h2_down, x_freq_l2_down))

        x_freq_h3_down, x_freq_l3_down = self.maxpool(x_freq_h3), self.maxpool(x_freq_l3)

        x_spa_h4, x_spa_l4 = self.spa_layer4((x_spa_h3, x_spa_l3))

        x_freq_h4, x_freq_l4 = self.freq_layer4((x_freq_h3_down, x_freq_l3_down))

        x_freq_h4_down, x_freq_l4_down = self.maxpool(x_freq_h4), self.maxpool(x_freq_l4)

        x_spa_2, x_spa_3, x_spa_4 = \
            self.conv_out_layer2((x_spa_h2, x_spa_l2)), \
            self.conv_out_layer3((x_spa_h3, x_spa_l3)), \
            self.conv_out_layer4((x_spa_h4, x_spa_l4))

        x_freq_2, x_freq_3, x_freq_4 = \
            self.conv_out_layer2((x_freq_h2_down, x_freq_l2_down)), \
            self.conv_out_layer3((x_freq_h3_down, x_freq_l3_down)), \
            self.conv_out_layer4((x_freq_h4_down, x_freq_l4_down))

        out_layer_2 = x_spa_2 + x_freq_2
        out_layer_3 = x_spa_3 + x_freq_3
        out_layer_4 = x_spa_4 + x_freq_4
        # out_layer_2 = self.cross_encoder_fusion2(x_spa_2, x_freq_2)
        # out_layer_3 = self.cross_encoder_fusion3(x_spa_3, x_freq_3)
        # out_layer_4 = self.cross_encoder_fusion4(x_spa_4, x_freq_4)

        out_x = self.cross_layer_fusion(out_layer_2, out_layer_3, out_layer_4)
        # print(out_x.shape)
        # out_x = Rearrange('b (h w) c -> b c h w', h=16, w=16)(out_x)
        out_x = self.avgpool(out_x)
        out_x = out_x.view(out_x.size(0), -1)
        out_x = self.fc(out_x)

        return out_x

class mymodel_mul(nn.Module):

    def __init__(self, block1, block2, layers, dim, size, num_classes=38, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):
        super(mymodel_mul, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.spa_inplanes = 1
        self.freq_inplanes = 64
        self.alpha = 0.5
        self.groups = groups
        self.base_width = width_per_group
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.cross_layer_fusion = Cross_Layer_Fusion(dim, size)
        self.spa_layer1 = self._make_spa_layer(block1, 64, layers[0], stride=2, norm_layer=norm_layer,
                                               First=True)
        self.freq_layer1 = self._make_freq_layer(block2, 64, layers[0], stride=1, norm_layer=norm_layer,
                                                 First=True)
        self.conv_out_layer1 = out_OCtaveConv(64, 64, kernel_size=(1, 1), stride=1, padding=0,  ##############
                                              groups=1, norm_layer=norm_layer)

        self.spa_layer2 = self._make_spa_layer(block1, 128, layers[1], stride=2, norm_layer=norm_layer)
        self.freq_layer2 = self._make_freq_layer(block2, 128, layers[1], stride=1, norm_layer=norm_layer)

        self.conv_out_layer2 = out_OCtaveConv(128, 128, kernel_size=(1, 1), alpha=self.alpha, stride=1, padding=0,
                                              ##############
                                              groups=1, norm_layer=norm_layer)

        self.spa_layer3 = self._make_spa_layer(block1, 256, layers[2], stride=2, norm_layer=norm_layer)
        self.freq_layer3 = self._make_freq_layer(block2, 256, layers[2], stride=1, norm_layer=norm_layer)
        # print(self.freq_layer3)

        self.conv_out_layer3 = out_OCtaveConv(256, 256, kernel_size=(1, 1), stride=1, padding=0,  ##############
                                              groups=1, norm_layer=norm_layer)
        self.spa_layer4 = self._make_spa_layer(block1, 512, layers[3], stride=2, norm_layer=norm_layer)
        self.freq_layer4 = self._make_freq_layer(block2, 512, layers[3], stride=1, norm_layer=norm_layer)
        self.conv_out_layer4 = out_OCtaveConv(512, 512, kernel_size=(1, 1), stride=1, padding=0,  ##############
                                              groups=1, norm_layer=norm_layer)


        self.avgpoolatten = nn.AdaptiveAvgPool2d(4)
        self.pool = nn.AvgPool2d(2, 1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(256, num_classes)
        # self.head = nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, DoubleOCTConv):
                    nn.init.constant_(m.bn2.weight, 0)
                elif isinstance(m, DoubleConv):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_spa_layer(self, block, planes, blocks, stride=2, norm_layer=None, First=False):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.spa_inplanes != planes * block.expansion:
            if self.spa_inplanes == 1 or self.spa_inplanes == 3:
                downsample = nn.Sequential(
                    OctaveCB(in_channels=planes, out_channels=planes * block.expansion, kernel_size=(1, 1),
                             stride=stride, padding=0)
                )  # todo in_channels=self.inplanes -->in_channels=planes
            else:
                downsample = nn.Sequential(
                    OctaveCB(in_channels=self.spa_inplanes, out_channels=planes * block.expansion, kernel_size=(1, 1),
                             stride=stride, padding=0)
                )

        layers = []
        layers.append(block(self.spa_inplanes, planes, stride, downsample, self.groups,
                            self.base_width, norm_layer, First))
        self.spa_inplanes = planes * block.expansion
        # print('input channel change to ',self.inplanes)
        for _ in range(1, blocks):
            layers.append(block(self.spa_inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _make_freq_layer(self, block, planes, blocks, stride=2, norm_layer=None, First=False):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.freq_inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                OctaveCB(in_channels=self.freq_inplanes, out_channels=planes * block.expansion, kernel_size=(1, 1),
                         stride=stride, padding=0)
            )  # todo in_channels=self.inplanes -->in_channels=planes

        layers = []
        layers.append(block(self.freq_inplanes, planes, stride, downsample, self.groups,
                            self.base_width, norm_layer, First))
        self.freq_inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.freq_inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):

        x_spa_h1, x_spa_l1 = self.spa_layer1(x)
        x_freq_h1, x_freq_l1 = self.freq_layer1(x)

        x_freq_h1_down, x_freq_l1_down = self.maxpool(x_freq_h1), self.maxpool(x_freq_l1)

        x_spa_h2, x_spa_l2 = self.spa_layer2((x_spa_h1, x_spa_l1))

        x_freq_h2, x_freq_l2 = self.freq_layer2((x_freq_h1_down, x_freq_l1_down))

        x_freq_h2_down, x_freq_l2_down = self.maxpool(x_freq_h2), self.maxpool(x_freq_l2)

        x_spa_h3, x_spa_l3 = self.spa_layer3((x_spa_h2, x_spa_l2))

        x_freq_h3, x_freq_l3 = self.freq_layer3((x_freq_h2_down, x_freq_l2_down))

        x_freq_h3_down, x_freq_l3_down = self.maxpool(x_freq_h3), self.maxpool(x_freq_l3)

        x_spa_h4, x_spa_l4 = self.spa_layer4((x_spa_h3, x_spa_l3))

        x_freq_h4, x_freq_l4 = self.freq_layer4((x_freq_h3_down, x_freq_l3_down))

        x_freq_h4_down, x_freq_l4_down = self.maxpool(x_freq_h4), self.maxpool(x_freq_l4)

        x_spa_2, x_spa_3, x_spa_4 = \
            self.conv_out_layer2((x_spa_h2, x_spa_l2)), \
            self.conv_out_layer3((x_spa_h3, x_spa_l3)), \
            self.conv_out_layer4((x_spa_h4, x_spa_l4))

        x_freq_2, x_freq_3, x_freq_4 = \
            self.conv_out_layer2((x_freq_h2_down, x_freq_l2_down)), \
            self.conv_out_layer3((x_freq_h3_down, x_freq_l3_down)), \
            self.conv_out_layer4((x_freq_h4_down, x_freq_l4_down))

        out_layer_2 = x_spa_2 * x_freq_2
        out_layer_3 = x_spa_3 * x_freq_3
        out_layer_4 = x_spa_4 * x_freq_4
        # out_layer_2 = self.cross_encoder_fusion2(x_spa_2, x_freq_2)
        # out_layer_3 = self.cross_encoder_fusion3(x_spa_3, x_freq_3)
        # out_layer_4 = self.cross_encoder_fusion4(x_spa_4, x_freq_4)

        out_x = self.cross_layer_fusion(out_layer_2, out_layer_3, out_layer_4)
        # print(out_x.shape)
        # out_x = Rearrange('b (h w) c -> b c h w', h=16, w=16)(out_x)
        out_x = self.avgpool(out_x)
        out_x = out_x.view(out_x.size(0), -1)
        out_x = self.fc(out_x)

        return out_x

class mymodel_cross_atten(nn.Module):

    def __init__(self, block1, block2, layers, dim, size, num_classes=38, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):
        super(mymodel_cross_atten, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.spa_inplanes = 1
        self.freq_inplanes = 64
        self.alpha = 0.5
        self.groups = groups
        self.base_width = width_per_group
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = AttentionPool2d()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.cross_encoder_fusion1 = Cross_Encoder_Fusion(64)
        self.cross_encoder_fusion2 = Cross_Atten_Encoder_Fusion(128)
        self.cross_encoder_fusion3 = Cross_Atten_Encoder_Fusion(256)
        self.cross_encoder_fusion4 = Cross_Atten_Encoder_Fusion(512)
        self.cross_layer_fusion = Cross_Layer_Fusion(dim, size)
        self.spa_layer1 = self._make_spa_layer(block1, 64, layers[0], stride=2, norm_layer=norm_layer,
                                               First=True)
        self.freq_layer1 = self._make_freq_layer(block2, 64, layers[0], stride=1, norm_layer=norm_layer,
                                                 First=True)
        self.conv_out_layer1 = out_OCtaveConv(64, 64, kernel_size=(1, 1), stride=1, padding=0,  ##############
                                              groups=1, norm_layer=norm_layer)

        self.spa_layer2 = self._make_spa_layer(block1, 128, layers[1], stride=2, norm_layer=norm_layer)
        self.freq_layer2 = self._make_freq_layer(block2, 128, layers[1], stride=1, norm_layer=norm_layer)

        self.conv_out_layer2 = out_OCtaveConv(128, 128, kernel_size=(1, 1), alpha=self.alpha, stride=1, padding=0,
                                              ##############
                                              groups=1, norm_layer=norm_layer)

        self.spa_layer3 = self._make_spa_layer(block1, 256, layers[2], stride=2, norm_layer=norm_layer)
        self.freq_layer3 = self._make_freq_layer(block2, 256, layers[2], stride=1, norm_layer=norm_layer)
        # print(self.freq_layer3)

        self.conv_out_layer3 = out_OCtaveConv(256, 256, kernel_size=(1, 1), stride=1, padding=0,  ##############
                                              groups=1, norm_layer=norm_layer)
        self.spa_layer4 = self._make_spa_layer(block1, 512, layers[3], stride=2, norm_layer=norm_layer)
        self.freq_layer4 = self._make_freq_layer(block2, 512, layers[3], stride=1, norm_layer=norm_layer)
        self.conv_out_layer4 = out_OCtaveConv(512, 512, kernel_size=(1, 1), stride=1, padding=0,  ##############
                                              groups=1, norm_layer=norm_layer)




        # 通过 Pooling 层将高宽降低为 1x1,[b,128,1,1]

        self.avgpoolatten = nn.AdaptiveAvgPool2d(4)
        self.pool = nn.AvgPool2d(2, 1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # self.attnpool = AttentionPool2d(self.freq_inplanes // 32, embed_dim, heads, output_dim)

        self.fc = nn.Linear(256, num_classes)
        # self.head = nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, DoubleOCTConv):
                    nn.init.constant_(m.bn2.weight, 0)
                elif isinstance(m, DoubleConv):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_spa_layer(self, block, planes, blocks, stride=2, norm_layer=None, First=False):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.spa_inplanes != planes * block.expansion:
            if self.spa_inplanes == 1 or self.spa_inplanes == 3:
                downsample = nn.Sequential(
                    OctaveCB(in_channels=planes, out_channels=planes * block.expansion, kernel_size=(1, 1),
                             stride=stride, padding=0)
                )  # todo in_channels=self.inplanes -->in_channels=planes
            else:
                downsample = nn.Sequential(
                    OctaveCB(in_channels=self.spa_inplanes, out_channels=planes * block.expansion, kernel_size=(1, 1),
                             stride=stride, padding=0)
                )

        layers = []
        layers.append(block(self.spa_inplanes, planes, stride, downsample, self.groups,
                            self.base_width, norm_layer, First))
        self.spa_inplanes = planes * block.expansion
        # print('input channel change to ',self.inplanes)
        for _ in range(1, blocks):
            layers.append(block(self.spa_inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _make_freq_layer(self, block, planes, blocks, stride=2, norm_layer=None, First=False):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.freq_inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                OctaveCB(in_channels=self.freq_inplanes, out_channels=planes * block.expansion, kernel_size=(1, 1),
                         stride=stride, padding=0)
            )  # todo in_channels=self.inplanes -->in_channels=planes

        layers = []
        layers.append(block(self.freq_inplanes, planes, stride, downsample, self.groups,
                            self.base_width, norm_layer, First))
        self.freq_inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.freq_inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):

        x_spa_h1, x_spa_l1 = self.spa_layer1(x)
        x_freq_h1, x_freq_l1 = self.freq_layer1(x)

        x_freq_h1_down, x_freq_l1_down = self.maxpool(x_freq_h1), self.maxpool(x_freq_l1)

        x_spa_h2, x_spa_l2 = self.spa_layer2((x_spa_h1, x_spa_l1))

        x_freq_h2, x_freq_l2 = self.freq_layer2((x_freq_h1_down, x_freq_l1_down))

        x_freq_h2_down, x_freq_l2_down = self.maxpool(x_freq_h2), self.maxpool(x_freq_l2)

        x_spa_h3, x_spa_l3 = self.spa_layer3((x_spa_h2, x_spa_l2))

        x_freq_h3, x_freq_l3 = self.freq_layer3((x_freq_h2_down, x_freq_l2_down))

        x_freq_h3_down, x_freq_l3_down = self.maxpool(x_freq_h3), self.maxpool(x_freq_l3)

        x_spa_h4, x_spa_l4 = self.spa_layer4((x_spa_h3, x_spa_l3))

        x_freq_h4, x_freq_l4 = self.freq_layer4((x_freq_h3_down, x_freq_l3_down))

        x_freq_h4_down, x_freq_l4_down = self.maxpool(x_freq_h4), self.maxpool(x_freq_l4)

        x_spa_2, x_spa_3, x_spa_4 = \
            self.conv_out_layer2((x_spa_h2, x_spa_l2)), \
            self.conv_out_layer3((x_spa_h3, x_spa_l3)), \
            self.conv_out_layer4((x_spa_h4, x_spa_l4)) \

        x_freq_2, x_freq_3, x_freq_4 = \
            self.conv_out_layer2((x_freq_h2_down, x_freq_l2_down)), \
            self.conv_out_layer3((x_freq_h3_down, x_freq_l3_down)), \
            self.conv_out_layer4((x_freq_h4_down, x_freq_l4_down)) \

        out_layer_2 = self.cross_encoder_fusion2(x_spa_2, x_freq_2)
        out_layer_3 = self.cross_encoder_fusion3(x_spa_3, x_freq_3)
        out_layer_4 = self.cross_encoder_fusion4(x_spa_4, x_freq_4)

        out_x = self.cross_layer_fusion(out_layer_2, out_layer_3, out_layer_4)
        out_x = self.avgpool(out_x)
        out_x = out_x.view(out_x.size(0), -1)
        out_x = self.fc(out_x)

        return out_x

class mymodel_spa(nn.Module):

    def __init__(self, block1, block2, layers, dim, size, num_classes=38, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):
        super(mymodel_spa, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.spa_inplanes = 1

        self.alpha = 0.5
        self.groups = groups
        self.base_width = width_per_group
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.cross_layer_fusion = Cross_Layer_Fusion(dim, size)
        self.spa_layer1 = self._make_spa_layer(block1, 64, layers[0], stride=1, norm_layer=norm_layer,
                                               First=True)

        self.conv_out_layer1 = out_OCtaveConv(64, 64, kernel_size=(1, 1), stride=1, padding=0,  ##############
                                              groups=1, norm_layer=norm_layer)

        self.spa_layer2 = self._make_spa_layer(block1, 128, layers[1], stride=2, norm_layer=norm_layer)

        self.conv_out_layer2 = out_OCtaveConv(128, 128, kernel_size=(1, 1), alpha=self.alpha, stride=1, padding=0,
                                              ##############
                                              groups=1, norm_layer=norm_layer)

        self.spa_layer3 = self._make_spa_layer(block1, 256, layers[2], stride=2, norm_layer=norm_layer)

        # print(self.freq_layer3)

        self.conv_out_layer3 = out_OCtaveConv(256, 256, kernel_size=(1, 1), stride=1, padding=0,  ##############
                                              groups=1, norm_layer=norm_layer)
        self.spa_layer4 = self._make_spa_layer(block1, 512, layers[3], stride=2, norm_layer=norm_layer)

        self.conv_out_layer4 = out_OCtaveConv(512, 512, kernel_size=(1, 1), stride=1, padding=0,  ##############
                                              groups=1, norm_layer=norm_layer)



        # 通过 Pooling 层将高宽降低为 1x1,[b,128,1,1]
        self.avgpoolatten = nn.AdaptiveAvgPool2d(4)
        self.pool = nn.AvgPool2d(2, 1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(256, num_classes)
        # self.head = nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, DoubleOCTConv):
                    nn.init.constant_(m.bn2.weight, 0)
                elif isinstance(m, DoubleConv):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_spa_layer(self, block, planes, blocks, stride=2, norm_layer=None, First=False):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.spa_inplanes != planes * block.expansion:
            if self.spa_inplanes == 1 or self.spa_inplanes == 3:
                downsample = nn.Sequential(
                    OctaveCB(in_channels=planes, out_channels=planes * block.expansion, kernel_size=(1, 1),
                             stride=stride, padding=0)
                )  # todo in_channels=self.inplanes -->in_channels=planes
            else:
                downsample = nn.Sequential(
                    OctaveCB(in_channels=self.spa_inplanes, out_channels=planes * block.expansion, kernel_size=(1, 1),
                             stride=stride, padding=0)
                )

        layers = []
        layers.append(block(self.spa_inplanes, planes, stride, downsample, self.groups,
                            self.base_width, norm_layer, First))
        self.spa_inplanes = planes * block.expansion
        # print('input channel change to ',self.inplanes)
        for _ in range(1, blocks):
            layers.append(block(self.spa_inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):

        x_spa_h1, x_spa_l1 = self.spa_layer1(x)

        x_spa_h2, x_spa_l2 = self.spa_layer2((x_spa_h1, x_spa_l1))

        x_spa_h3, x_spa_l3 = self.spa_layer3((x_spa_h2, x_spa_l2))

        x_spa_h4, x_spa_l4 = self.spa_layer4((x_spa_h3, x_spa_l3))

        x_spa_2, x_spa_3, x_spa_4 = \
            self.conv_out_layer2((x_spa_h2, x_spa_l2)), \
                self.conv_out_layer3((x_spa_h3, x_spa_l3)), \
                self.conv_out_layer4((x_spa_h4, x_spa_l4))

        out_x = self.cross_layer_fusion(x_spa_2, x_spa_3, x_spa_4)
        # print(out_x.shape)
        # out_x = Rearrange('b (h w) c -> b c h w', h=16, w=16)(out_x)
        out_x = self.avgpool(out_x)
        out_x = out_x.view(out_x.size(0), -1)
        out_x = self.fc(out_x)

        return out_x


class mymodel_freq(nn.Module):

    def __init__(self, block1, block2, layers, dim, size, num_classes=38, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):
        super(mymodel_freq, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.freq_inplanes = 64
        self.alpha = 0.5
        self.groups = groups
        self.base_width = width_per_group
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.cross_encoder_fusion1 = Cross_Encoder_Fusion(64)
        self.cross_layer_fusion = Cross_Layer_Fusion(dim, size)

        self.freq_layer1 = self._make_freq_layer(block2, 64, layers[0], stride=1, norm_layer=norm_layer,
                                                 First=True)
        self.conv_out_layer1 = out_OCtaveConv(64, 64, kernel_size=(1, 1), stride=1, padding=0,  ##############
                                              groups=1, norm_layer=norm_layer)

        self.freq_layer2 = self._make_freq_layer(block2, 128, layers[1], stride=1, norm_layer=norm_layer)

        self.conv_out_layer2 = out_OCtaveConv(128, 128, kernel_size=(1, 1), alpha=self.alpha, stride=1, padding=0,
                                              ##############
                                              groups=1, norm_layer=norm_layer)

        self.freq_layer3 = self._make_freq_layer(block2, 256, layers[2], stride=1, norm_layer=norm_layer)
        # print(self.freq_layer3)

        self.conv_out_layer3 = out_OCtaveConv(256, 256, kernel_size=(1, 1), stride=1, padding=0,  ##############
                                              groups=1, norm_layer=norm_layer)

        self.freq_layer4 = self._make_freq_layer(block2, 512, layers[3], stride=1, norm_layer=norm_layer)
        self.conv_out_layer4 = out_OCtaveConv(512, 512, kernel_size=(1, 1), stride=1, padding=0,  ##############
                                              groups=1, norm_layer=norm_layer)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))



        # 通过 Pooling 层将高宽降低为 1x1,[b,128,1,1]
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.avgpoolatten = nn.AdaptiveAvgPool2d(4)
        self.pool = nn.AvgPool2d(2, 1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(256, num_classes)
        # self.head = nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, DoubleOCTConv):
                    nn.init.constant_(m.bn2.weight, 0)
                elif isinstance(m, DoubleConv):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_freq_layer(self, block, planes, blocks, stride=2, norm_layer=None, First=False):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.freq_inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                OctaveCB(in_channels=self.freq_inplanes, out_channels=planes * block.expansion, kernel_size=(1, 1),
                         stride=stride, padding=0)
            )  # todo in_channels=self.inplanes -->in_channels=planes

        layers = []
        layers.append(block(self.freq_inplanes, planes, stride, downsample, self.groups,
                            self.base_width, norm_layer, First))
        self.freq_inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.freq_inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):

        x_freq_h1, x_freq_l1 = self.freq_layer1(x)

        x_freq_h1_down, x_freq_l1_down = self.maxpool(x_freq_h1), self.maxpool(x_freq_l1)

        x_freq_h2, x_freq_l2 = self.freq_layer2((x_freq_h1_down, x_freq_l1_down))

        x_freq_h2_down, x_freq_l2_down = self.maxpool(x_freq_h2), self.maxpool(x_freq_l2)

        x_freq_h3, x_freq_l3 = self.freq_layer3((x_freq_h2_down, x_freq_l2_down))

        x_freq_h3_down, x_freq_l3_down = self.maxpool(x_freq_h3), self.maxpool(x_freq_l3)

        x_freq_h4, x_freq_l4 = self.freq_layer4((x_freq_h3_down, x_freq_l3_down))

        x_freq_h4_down, x_freq_l4_down = self.maxpool(x_freq_h4), self.maxpool(x_freq_l4)

        x_freq_1, x_freq_2, x_freq_3, x_freq_4 = \
            self.conv_out_layer1((x_freq_h1_down, x_freq_l1_down)), \
                self.conv_out_layer2((x_freq_h2_down, x_freq_l2_down)), \
                self.conv_out_layer3((x_freq_h3_down, x_freq_l3_down)), \
                self.conv_out_layer4((x_freq_h4_down, x_freq_l4_down))

        out_x = self.cross_layer_fusion(x_freq_2, x_freq_3, x_freq_4)
        # print(out_x.shape)
        # out_x = Rearrange('b (h w) c -> b c h w', h=8, w=8)(out_x)
        out_x = self.avgpool(out_x)
        out_x = out_x.view(out_x.size(0), -1)
        out_x = self.fc(out_x)

        return out_x


def model18(pretrained=False, **kwargs):
    dim, size = [128, 256, 512], 196
    model = mymodel(DoubleOCTConv, FrequencyOCTFilter, [2, 2, 2, 2], dim, size, **kwargs)
    return model


def model18_add(pretrained=False, **kwargs):
    dim, size = [128, 256, 512], 196
    model = mymodel_add(DoubleOCTConv, FrequencyOCTFilter, [2, 2, 2, 2], dim, size, **kwargs)
    return model

def model18_cross_atten(pretrained=False, **kwargs):
    dim, size = [128, 256, 512], 196
    model = mymodel_cross_atten(DoubleOCTConv, FrequencyOCTFilter, [2, 2, 2, 2], dim, size, **kwargs)
    return model


def model18_only_spa(pretrained=False, **kwargs):
    dim, size = [128, 256, 512], 196
    model = mymodel_spa(DoubleOCTConv, FrequencyOCTFilter, [2, 2, 2, 2], dim, size, **kwargs)
    return model


def model18_only_freq(pretrained=False, **kwargs):
    dim, size = [128, 256, 512], 196
    model = mymodel_freq(DoubleOCTConv, FrequencyOCTFilter, [2, 2, 2, 2], dim, size, **kwargs)
    return model

def model18_mul(pretrained=False, **kwargs):
    dim, size = [128, 256, 512], 196
    model = mymodel_mul(DoubleOCTConv, FrequencyOCTFilter, [2, 2, 2, 2], dim, size, **kwargs)
    return model


if __name__ == '__main__':
    x = torch.randn(1, 1, 64, 64)
    model = model18()
    model_mul = model18_mul()
    model_add = model18_add()
    model_cross_atten = model18_cross_atten()
    model_only_spa = model18_only_spa()
    model_only_frep = model18_only_freq()
    print(model_only_frep(x).shape)
    from torchinfo import summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        print('_________________________________model_mul_____________________________________________________')
        summary(model_mul, input_size=(1, 1, 64, 64), device=str(device))
        # print('_________________________________model_add_________________________________________________')
        # summary(model_add, input_size=(1, 1, 64, 64), device=str(device))
        # print('_________________________________model_cross_atten_________________________________________________')
        # summary(model_cross_atten, input_size=(1, 1, 64, 64), device=str(device))
        # print('_________________________________model_only_spa____________________________________________')
        # summary(model_only_spa, input_size=(1, 1, 64, 64), device=str(device))
        # print('_________________________________model_only_frep___________________________________________')
        # summary(model_only_frep, input_size=(1, 1, 64, 64), device=str(device))
    except Exception as e:
        print("Error in summary:", e)
