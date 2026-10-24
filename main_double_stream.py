# encoding: utf-8
from __future__ import print_function, division

import copy
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from dataset import get_train_val_test_loaders

# from models import convnext, cwdr, densenet, efficientnet, mobilvit, resnet, swintransformer, vgg, vit
from plotConfusionMatrix import plot_Matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from tsne import test_my_model_with_tsne
import time
from sklearn.metrics import confusion_matrix
from datetime import datetime
import torch.nn.init as init
import warnings
import random

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
warnings.simplefilter(action='ignore')


def test_my_model(model, classes, dataloader_valid):
    model.eval()  # 将模型设置为评估模式
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for img, labels in dataloader_valid:
            if labels.dim() == 2:  # 检查是否是 one-hot 编码
                labels = torch.argmax(labels, dim=1)
            img = img.cuda()
            labels = labels.cuda()
            outputs = model(img)
            _, preds = torch.max(outputs.data, 1)
            # 保存预测结果和真实标签
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算性能指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    print(f'Accuracy: {accuracy:.8f}')
    print(f'Precision: {precision:.8f}')
    print(f'Recall: {recall:.8f}')
    print(f'F1 Score: {f1:.8f}')
    cm = confusion_matrix(all_labels, all_preds, labels=[i for i in range(len(classes))])
    plot_Matrix(cm, classes, title='Confusion matrix', save_dir='confusion_matrix')


def train_model(model, criterion, optimizer, scheduler, start_time, num_epochs):
    best_model_wts = None
    best_acc = 0.0

    # --- 1. 初始化所有指标列表 ---
    train_loss = []
    test_loss = []
    train_acc = []  # 新增：训练准确率列表
    test_acc = []  # 新增：测试准确率列表
    train_f1_micro = []
    train_f1_macro = []
    train_precision_micro = []  # 新增
    train_precision_macro = []  # 新增
    train_recall_micro = []  # 新增
    train_recall_macro = []  # 新增
    test_f1_micro = []
    test_f1_macro = []
    test_precision_micro = []  # 新增
    test_precision_macro = []  # 新增
    test_recall_micro = []  # 新增
    test_recall_macro = []  # 新增
    lr_list = []
    epoch_list = []

    since = time.time()
    save_dir = os.path.join('./result', start_time)
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        current_lr = scheduler.get_last_lr()[0]
        lr_list.append(current_lr)
        epoch_list.append(epoch)
        print('Epoch {}/{}, lr: {:.8f}'.format(epoch + 1, num_epochs, current_lr))
        print('-' * 30)

        for phase in ['train', 'val']:
            if phase == 'train':
                print('>>> Training phase')
                model.train()
            else:
                print('>>> Validation phase')
                model.eval()

            running_loss = 0.0
            all_preds = []
            all_labels = []

            # --- 2. 迭代数据 ---
            for inputs, labels in data_loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 处理 one-hot 标签
                if labels.dim() == 2:
                    labels = torch.argmax(labels, dim=1)

                optimizer.zero_grad()

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs).float()
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(torch.softmax(outputs, dim=1), 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            # --- 3. 计算所有指标 ---
            epoch_loss = running_loss / len(all_labels)
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)

            # 基础准确率
            epoch_acc = (all_preds == all_labels).mean()

            # 核心分类指标 (添加 zero_division=0 防止警告)
            f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
            f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
            precision_micro = precision_score(all_labels, all_preds, average='micro', zero_division=0)
            precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
            recall_micro = recall_score(all_labels, all_preds, average='micro', zero_division=0)
            recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)

            # --- 4. 保存指标到列表 ---
            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
                train_f1_micro.append(f1_micro)
                train_f1_macro.append(f1_macro)
                train_precision_micro.append(precision_micro)
                train_precision_macro.append(precision_macro)
                train_recall_micro.append(recall_micro)
                train_recall_macro.append(recall_macro)
                scheduler.step()  # 仅在训练结束后更新学习率
            else:
                test_loss.append(epoch_loss)
                test_acc.append(epoch_acc)
                test_f1_micro.append(f1_micro)
                test_f1_macro.append(f1_macro)
                test_precision_micro.append(precision_micro)
                test_precision_macro.append(precision_macro)
                test_recall_micro.append(recall_micro)
                test_recall_macro.append(recall_macro)

            # --- 5. 打印完整指标 ---
            print('{} Loss: {:.4f} | Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print('  Micro: P: {:.4f} | R: {:.4f} | F1: {:.4f}'.format(precision_micro, recall_micro, f1_micro))
            print('  Macro: P: {:.4f} | R: {:.4f} | F1: {:.4f}'.format(precision_macro, recall_macro, f1_macro))

            #

            if phase == 'val' and epoch_acc >= best_acc:
                best_acc = epoch_acc
                # best_model_wts = model.state_dict()
                best_model_wts = copy.deepcopy(model.state_dict())  # 使用深拷贝保存当前模型参数
            if phase == 'val' and epoch_acc > 0.995:
                # 判断是否需要保存模型
                save_path = os.path.join(save_dir, f'{start_time}_model_{epoch + 1}.pth')
                torch.save(model.state_dict(), save_path)
        # torch.save(model.state_dict(), f'./result/{start_time}' + '_model_' + str(epoch + 1) + '.pth')
    time_elapsed = time.time() - since

    # save_file('fea_train', train_loss, './result/')
    # plt_lr_decay(epoch_list, lr_list)
    print('Training complete in {:.0f}s'.format(time_elapsed))
    print('Best val Acc: {:4f}:'.format(best_acc))
    # now = datetime.now().strftime("%Y%m%d_%H%M%S")  # 格式：年月日_小时分钟秒
    save_path = os.path.join(save_dir, f'{start_time}_best_model.pth')
    torch.save(best_model_wts, save_path)
    print(f'Model saved to: {save_path}')
    model.load_state_dict(torch.load(save_path), False)
    print(f'Model loaded from: {save_path}')

    return model


# ##### 初始化模型参数 #######
# def weight_init(m):
#     # 使用isinstance来判断m属于什么类型
#     if isinstance(m, nn.Conv2d):
#         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#         m.weight.data.normal_(0, math.sqrt(2. / n))
#     elif isinstance(m, nn.BatchNorm2d):
#         m.weight.data.fill_(1)
#         m.bias.data.zero_()

def weight_init(m):
    """Kaiming 初始化 + BatchNorm 初始化"""
    # init.xavier_normal_(m.weight)
    if isinstance(m, nn.Conv2d):
        # Kaiming 初始化（针对 ReLU 激活函数）
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:  # 如果卷积层有偏置项，初始化为0
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        # BatchNorm 初始化：权重为1，偏置为0
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


def test_model_output(model_factory, input_shape=(1, 1, 64, 64)):
    try:
        model = model_factory()
        print(f"[INFO] Model initialized: {model.__class__.__name__}")

        x = torch.randn(input_shape)  # 随机输入
        output = model(x)

        if isinstance(output, tuple):
            output = output[0]  # 如果模型返回多个输出，取第一个

        print(f"[SUCCESS] Forward pass OK. Output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed for model: {model_factory.__qualname__}")
        print(f"Error: {e}")
        return False

if __name__ == '__main__':
    import os
    import zipfile

    zip_file_path = 'dataset_mix38/Wafer_Map_Datasets.npz.zip'
    extracted_npz_path = 'dataset_mix38/Wafer_Map_Datasets.npz'

    # 解压文件（如果未解压）
    if not os.path.exists(extracted_npz_path):
        if not os.path.exists(zip_file_path):
            raise FileNotFoundError(f"压缩包文件不存在: {zip_file_path}")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall('dataset_mix38/')

    # # 加载数据
    # dataset_path = extracted_npz_path
    # data = np.load(dataset_path)
    # print("数据加载成功:", data.files)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # 定义训练参数
    dataset_path = 'dataset_mix38/Wafer_Map_Datasets.npz'
    num_epochs = 100
    # warmup_switch = False  # 控制是否启用warmup
    # warmup_epochs = 20  # warmup的epoch数
    # cycle_epochs = 200  # 每个余弦周期的epoch数（T_max）
    batch_size = 128
    test_size = 0.2

    train_DataLoader, val_DataLoader, test_DataLoader = get_train_val_test_loaders(dataset_path,
                                                                                   batch_size=128,
                                                                                   test_size=0.2,
                                                                                   random_state=42)

    data_loaders = {'train': train_DataLoader,
                    'val': val_DataLoader}

    classes = ('Normal', 'Center', 'Donut', 'Edge_Loc', 'Edge_Ring', 'Loc', 'Near_Full', 'Scratch', 'Random', 'C+EL',
               'C+ER', 'C+L', 'C+S', 'D+EL', 'D+ER', 'D+L', 'D+S', 'EL+L', 'EL+S', 'ER+L', 'ER+S', 'L+S',
               'C+EL+L', 'C+EL+S', 'C+ER+L', 'C+ER+S', 'C+L+S', 'D+EL+L', 'D+EL+S', 'D+ER+L', 'D+ER+S',
               'D+L+S', 'EL+L+S', 'ER+L+S', 'C+L+EL+S', 'C+L+ER+S', 'D+L+EL+S', 'D+L+ER+S')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # todo 可选对比模型：convnext,cwdr,densenet,efficientnet,mobilvit,resnet,swintransformer,vgg,vit
    from wafer_models import cwdr, mernet,msftrans,WMPeleeNet,WSCN,ACLDN,DPFEENet,FKMA
    from DSHF_xr_stru import (SFFNet_wo_cross_layer, SFFNet_wo_cross_encoder, SFFNet_conv_spa_oct_freq,
                              SFFNet_oct_spa_conv_freq,SFFNet_vil_conv,
                              SFFNet_sffm_add, SFFNet_sffm_mul, SFFNet_sffm_crossatten,
                              SFFNet_sffmatten_cc, SFFNet_sffmatten_cs, SFFNet_sffmatten_ss,
                              SFFNet_clfm_mul, SFFNet_clfm_fpnd2u, SFFNet_clfm_fpnu2d, SFFNet_clfm_add
                              )
    from models import convnext, densenet, efficientnet, mobilvit, resnet, swintransformer, vgg, vit,peleenet,repvit
    from wafer_models import cwdr, mernet, msftrans, WMPeleeNet, WSCN,ACLDN,DPFEENet,FKMA
    from wafer_models import mlr_wm_vit,MRWANet,msmpdcnn

    model_factories = [
        # todo 对比实验模型

        # lambda: peleenet.PeleeNet(),
        # lambda: convnext.convnext_small(38, 1),
        lambda: repvit.repvit_m1_5(num_classes=38, in_channels=1),
        # lambda: mlr_wm_vit.MLR_WM_ViT(1, 38),
        lambda: MRWANet.MRWANet(1, 38),
        # lambda: msmpdcnn.Conv5_MSMP_DCNN(1, 38),

        # lambda: efficientnet.efficientnet_b0(38, 1),

        # lambda: swintransformer.swin_tiny_patch4_window7_224(38, 1),
        # lambda: vit.vit(image_size=64, patch_size=4, num_classes=38, channels=1),
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
        # todo 消融--alpha参数
        # lambda: model_18(num_classes=38, alpha=0.875),
        # lambda: model_18(num_classes=38, alpha=0.75),
        # lambda: model_18(num_classes=38, alpha=0.625),
        # lambda: model_18(num_classes=38, alpha=0.5),
        # lambda: model_18(num_classes=38, alpha=0.375),
        # lambda: model_18(num_classes=38, alpha=0.25),
        # lambda: model_18(num_classes=38, alpha=0.125),
        # todo 消融--模型结构
        # todo 跨编码器融合结构
        # lambda: SFFNet_sffm_add(num_classes=38, alpha=0.5),
        # lambda: SFFNet_sffm_mul(num_classes=38, alpha=0.5),
        # lambda: SFFNet_sffm_crossatten(num_classes=38, alpha=0.5),
        #
        # todo SFFM注意力设置
        # lambda: SFFNet_sffmatten_cs(num_classes=38, alpha=0.5),
        # lambda: SFFNet_sffmatten_cc(num_classes=38, alpha=0.5),
        # lambda: SFFNet_sffmatten_ss(num_classes=38, alpha=0.5),
        #
        # todo 跨层融合结构
        # lambda: SFFNet_clfm_add(num_classes=38, alpha=0.5),
        # lambda: SFFNet_clfm_mul(num_classes=38, alpha=0.5),
        # lambda: SFFNet_clfm_fpnd2u(num_classes=38, alpha=0.5),
        # lambda: SFFNet_clfm_fpnu2d(num_classes=38, alpha=0.5),
    ]
    model_names = [
        # # todo 对比实验模型

        # 'peleenet',
        # 'convnext',
        'repvit',
        # 'mlr_wm_vit',
        'MRWANet',
        # 'Conv5_MSMP_DCNN',

        # "efficientnet",

        # "swintransformer",
        # "vit",
        # "ResNet18","DenseNet121",
        # "WaferSegClassNet","WMPeleeNet","CWDR_model",
        # "MERNet",
        # "mobilevit_xxs","MSFTransformer",
        # "ACLDN",
        # "DPEENet",
        # "FMKA",
        # todo 消融--alpha
        # "proposed_alpha_0.875", "proposed_alpha_0.75", "proposed_alpha_0.625",   "proposed_alpha_18_0.5",
        # "proposed_alpha_0.375", "proposed_alpha_0.25",  "proposed_alpha_0.125",
        # "proposed_alpha_vallina",
        # todo 消融-模型结构
        # "proposed", "proposed_vallina_conv", "proposed_wo_cross_encoder",
        # "proposed_wo_cross_layer",
        # todo 消融-双分支与OCT
        # "proposed_only_spa",
        # "proposed_only_freq", "proposed_only_spa_vallina_conv", "proposed_only_freq_vallina_conv",
        # todo 普通跨域融合方式的消融
        # "SFFNet_sffm_add",
        # "SFFNet_sffm_mul",
        # "SFFNet_sffm_crossatten",

        # todo sffm注意力设置的消融
        # "SFFNet_sffmatten_cs",
        # "SFFNet_sffmatten_cc",
        # "SFFNet_sffmatten_ss",

        # todo 跨层结构消融
        # "SFFNet_clfm_add",
        # "SFFNet_clfm_mul",
        # "SFFNet_clfm_fpnd2u",
        # "SFFNet_clfm_fpnu2d",

    ]

    # results = {}
    # for i, factory in enumerate(model_factories):
    #     input_shape = (1, 1, 64, 64)
    #     result = test_model_output(factory, input_shape)
    #     results[f"Model {i + 1}"] = {
    #         "status": "Passed" if result else "Failed",
    #         "input_shape": input_shape,
    #     }

    print(f'numbers of model_names: {len(model_names)}, model_factories: {len(model_factories)}')
    for i, (model_factory, model_name) in enumerate(zip(model_factories, model_names)):
        # 每个模型运行3次
        for run_id in range(1, 4):
            print(f"\n===== 开始训练 {model_name} (第{run_id}/3次) =====")

            # 初始化模型（每次重新初始化，确保独立性）
            model = model_factory()
            weight_init(model)
            model = model.to(device)

            # 定义优化器和损失函数
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            lr_schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

            # 生成带运行序号的时间戳，便于区分保存的模型文件
            start_time = datetime.now().strftime(f"{model_name}_run{run_id}_mc_%Y%m%d_%H%M%S")

            # 训练模型
            print(f'Start training at {start_time}')
            trained_model = train_model(
                model, criterion, optimizer, lr_schedule,
                start_time, num_epochs=num_epochs
            )
            print(f'Finished training {model_name} (第{run_id}/3次) at {datetime.now().strftime("%Y%m%d_%H%M%S")}')

            # 测试模型
            test_my_model(trained_model, classes, test_DataLoader)
            # test_my_model_with_tsne(trained_model, classes, test_DataLoader, model_name)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 每次运行后短暂休息（可选，避免资源冲突）
            time.sleep(10)

        # 每个模型的所有3次运行结束后，按照原逻辑处理延迟
        if i < 5:
            time.sleep(30)

    # 所有模型训练完成后，关机
    time.sleep(600)
    os.system("/usr/bin/shutdown")
