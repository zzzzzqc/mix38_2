# encoding: utf-8
from __future__ import print_function, division

import copy
import torch
import torch.nn as nn
import numpy as np
import time
import torch.nn.init as init
import warnings
import random

from thop import profile, clever_format

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
warnings.simplefilter(action='ignore')


def weight_init(m):
    """Kaiming 初始化 + BatchNorm 初始化"""
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


# ---------------------- 1. 计算参数量 ----------------------
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


# ---------------------- 2. 计算FLOPs ----------------------
def count_flops(model, input_size=(1, 1, 64, 64), device='cpu'):
    model.eval()
    dummy_input = torch.randn(input_size).to(device)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    flops = flops * 2
    flops_str = clever_format([flops], "%.3f")[0]
    return flops_str, flops


# ---------------------- 3. 计算推理时间 ----------------------
# ---------------------- 3. 计算推理时间 (优化稳定版) ----------------------
def measure_latency(model, input_size=(1, 1, 64, 64), device='cuda', warmup=50, repeats=300, discard_ratio=0.1):
    """
    discard_ratio: 剔除头尾异常值的比例 (例如 0.1 代表去掉最快10%和最慢10%)
    """
    model.eval()
    model.to(device)
    dummy_input = torch.randn(input_size).to(device)

    # 1. 预热
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    # 2. 正式测量：记录每一次的单独时间
    timings = []
    with torch.no_grad():
        for _ in range(repeats):
            if device == 'cuda':
                torch.cuda.synchronize()
            start_time = time.time()

            _ = model(dummy_input)

            if device == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()

            timings.append((end_time - start_time) * 1000)  # 转为 ms

    # 3. 关键步骤：剔除异常值 (Trimming)
    if discard_ratio > 0:
        timings.sort()
        trim_count = int(len(timings) * discard_ratio)
        # 切片：去掉头部 trim_count 个和尾部 trim_count 个
        if trim_count > 0:
            timings = timings[trim_count:-trim_count]

    # 4. 计算最终结果
    import numpy as np
    avg_latency = np.mean(timings)
    std_latency = np.std(timings)

    # 可以打印一下标准差看看波动情况，主循环不需要可以删掉
    # print(f"  (Stability: Std={std_latency:.4f}ms on {len(timings)} samples)")

    return avg_latency

if __name__ == '__main__':
    # 设备设置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 导入模型（保留你原有的模型导入）
    from wafer_models import cwdr, mernet, msftrans, WMPeleeNet, WSCN
    from DSHF_oct_conv import (model18_only_freq_or, model18_only_freq_am, model18_only_freq_ph,
                                model18_only_freq_sc, model18_only_freq_an)
    from DSHF_vallina_conv import (model18_only_freq_vallina_amp, model18_only_freq_vallina_ori,
                                    model18_only_freq_vallina_pha, model18_only_freq_vallina_sincos,
                                    model18_only_freq_vallina_ang)
    from DSHF_xr_stru import (SFFNet_freq_sc, SFFNet_freq_amp, SFFNet_freq_ang, SFFNet_freq_ori,
                               SFFNet_freq_pha)
    from DSHF_oct_conv_copy import DSHF_copy_only_spa, DSHF_copy_only_freq, DSHF_copy
    from DSHF_vallina_copy import DSHF_copy_only_freq_vallina, DSHF_copy_vallina, DSHF_copy_only_spa_vallina
    from DSHF_xr_stru import (SFFNet_wo_cross_layer, SFFNet_wo_cross_encoder, SFFNet_conv_spa_oct_freq,
                              SFFNet_oct_spa_conv_freq, SFFNet_sffm_add, SFFNet_sffm_mul, SFFNet_sffm_crossatten,
                              SFFNet_sffmatten_cc, SFFNet_sffmatten_cs, SFFNet_sffmatten_ss)
    from models import convnext, densenet, efficientnet, mobilvit, resnet, swintransformer, vgg, vit,peleenet,repvit
    from wafer_models import cwdr, mernet, msftrans, WMPeleeNet, WSCN,ACLDN,DPFEENet,FKMA
    from wafer_models import mlr_wm_vit,MRWANet,msmpdcnn
    from DSHF_oct_conv import model18_freq_am, model18_freq_or, model18_freq_ph, model18y_freq_an, model18_freq_sc
    # 定义待分析的模型列表
    model_factories = [

        #
        # # todo 对比试验模型
        lambda: peleenet.PeleeNet(),
        lambda: convnext.convnext_small(38, 1),
        lambda: repvit.repvit_m1_5(num_classes=38,in_channels=1),
        lambda: mlr_wm_vit.MLR_WM_ViT(1, 38),
        lambda: MRWANet.MRWANet(1, 38),
        lambda: msmpdcnn.Conv5_MSMP_DCNN(1, 38),
        # lambda: efficientnet.efficientnet_b0(38, 1),
        # lambda: vit.vit(image_size=64, patch_size=4, num_classes=38, channels=1),
        # lambda: swintransformer.swin_tiny_patch4_window7_224(38,1),
        # lambda: resnet.ResNet18(1, 38),
        # lambda: densenet.DenseNet121(38, 1),
        # lambda: WSCN.WaferSegClassNet(in_channels=1, num_class=38),
        # lambda: WMPeleeNet.WMPeleeNet(in_ch=1, num_classes=38),
        # lambda: cwdr.CWDR_model(num_of_classes=38),
        # lambda: mernet.MERNet(in_channels=1, num_classes=38),
        # lambda: mobilvit.mobilevit_xxs(38, 1),
        # lambda: msftrans.MSFTransformer(num_classes=38, image_size=64, embed_dim=64, num_heads=8, num_layers=6),
        # lambda: DPFEENet.DPFEE_Net(in_channels=1, num_classes=38),
        # lambda: FKMA.FMKANet(in_channels=1, num_classes=38, base_channels=64),


    ]

    model_names = [

        # # todo 对比实验模型
        'peleenet',
        'convnext',
        'repvit',
        'mlr_wm_vit',
        'MRWANet',
        'Conv5_MSMP_DCNN',

        # "efficientnet", "vit",
        # "swintransformer",
        # "ResNet18", "DenseNet121",
        # "WaferSegClassNet", "WMPeleeNet", "CWDR_model",
        # "MERNet",
        # "mobilevit_xxs", "MSFTransformer",
        # "DPEENet",
        # "FKMA",
        #

    ]

    print(f"\n===== 开始模型分析 =====")
    print(f"Total models to analyze: {len(model_names)}\n")

    #

    for i, (model_factory, model_name) in enumerate(zip(model_factories, model_names)):
        print(f"[{i+1}/{len(model_names)}] Analyzing: {model_name}")
        print("-" * 50)

        model = model_factory()
        weight_init(model)
        model = model.to(device)

        # 1. 参数量 (转换为 M，保留8位小数)
        total_params, trainable_params = count_parameters(model)
        total_params_m = total_params / 1e6
        trainable_params_m = trainable_params / 1e6
        print(f"  Total Params: {total_params_m:.8f} M ({total_params:,})")
        print(f"  Trainable Params: {trainable_params_m:.8f} M ({trainable_params:,})")

        # 2. FLOPs (转换为 G，保留8位小数)
        flops_str, flops_raw = count_flops(model, input_size=(1, 1, 64, 64), device=device)
        flops_g = flops_raw / 1e9
        print(f"  FLOPs: {flops_g:.8f} G")

        # 3. 推理时间 (直接调用封装好的 measure_latency)
        # 注意：这里传入 repeats=30 来匹配你原本想要的 30次平均的需求
        avg_latency = measure_latency(model, input_size=(1, 1, 64, 64), device=device, warmup=300, repeats=1000)
        print(f"  Latency: {avg_latency:.8f} ms")

        print("-" * 50 + "\n")


        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("===== 所有模型分析完成 =====")