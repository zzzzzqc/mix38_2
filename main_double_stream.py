# encoding: utf-8
from __future__ import print_function, division

import copy
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from dataset import get_train_val_test_loaders
from DSHF_oct_conv import model_18, model18_only_freq, model18_only_spa, model_18_wo_cross_encoder, \
    model_18_wo_cross_layer
from DSHF_vallina_conv import model_18_vallina, model18_only_spa_vallina, model18_only_freq_vallina
from models import convnext, cwdr, densenet, efficientnet, mobilvit, resnet, swintransformer, vgg, vit
from plotConfusionMatrix import plot_Matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tsne import test_my_model_with_tsne
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
    train_loss = []
    test_loss = []
    train_f1_micro = []
    train_f1_macro = []
    test_f1_micro = []
    test_f1_macro = []
    lr_list = []
    epoch_list = []
    since = time.time()
    save_dir = os.path.join('./result', start_time)
    os.makedirs(save_dir, exist_ok=True)  # 如果目录已经存在，则不会报错
    for epoch in range(num_epochs):
        lr_list.append(scheduler.get_last_lr())
        epoch_list.append(epoch)
        print('Epoch {}/{}, lr {}'.format(epoch + 1, num_epochs, scheduler.get_last_lr()))
        print('-' * 20)
        # Each epoch has a training and validation phase

        for phase in ['train', 'val']:
            if phase == 'train':
                print('model is training')
                model.train()
            else:
                print('model is evaling')
                model.eval()

            running_loss, running_corrects, total_samples = 0.0, 0, 0
            all_preds, all_labels = [], []

            for inputs, labels in data_loaders[phase]:
                # 数据移动到设备上
                inputs = inputs.to(device)
                labels = labels.to(device)
                # print(labels.shape)
                if labels.dim() == 2:  # 检查是否是 one-hot 编码
                    labels = torch.argmax(labels, dim=1)
                # 清零梯度
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs).float()
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # 统计信息
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(torch.softmax(outputs, dim=1), 1)
                corrects = (preds == labels)
                running_corrects += corrects.sum().item()
                total_samples += inputs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                # print(f'dataloder down')
            # 计算损失、准确率和F1分数
            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects / total_samples
            all_preds, all_labels = np.array(all_preds), np.array(all_labels)
            f1_micro = f1_score(all_labels, all_preds, average='micro')
            f1_macro = f1_score(all_labels, all_preds, average='macro')

            if phase == 'train':
                train_loss.append(epoch_loss)
                # 可以选择记录不同的F1分数
                train_f1_micro.append(f1_micro)
                train_f1_macro.append(f1_macro)
                scheduler.step()
            else:
                test_loss.append(epoch_loss)
                test_f1_micro.append(f1_micro)
                test_f1_macro.append(f1_macro)

            print('{} Loss: {:.8f} {} Acc: {:.8f} F1 Micro: {:.8f} F1 Macro: {:.8f}'.format(
                phase, epoch_loss, phase, epoch_acc, f1_micro, f1_macro))

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
    from wafer_models import cwdr, mernet,msftrans,WMPeleeNet,WSCN
    model_factories = [
        # todo 对比实验模型
        lambda: resnet.ResNet18(1, 38),
        lambda: densenet.DenseNet121(38, 1),
        lambda: WSCN.WaferSegClassNet(in_channels=1, num_class=38),
        lambda: WMPeleeNet.WMPeleeNet(in_ch=1, num_classes=38),
        lambda: cwdr.CWDR_model(num_of_classes=38),
        lambda: mernet.MERNet(in_channels=1, num_classes=38),
        lambda: mobilvit.mobilevit_xxs(38, 1),
        lambda: msftrans.MSFTransformer(num_classes=38, image_size=64, embed_dim=64, num_heads=8, num_layers=6),
        # todo 消融--alpha参数
        # lambda: model_18(num_classes=38, alpha=0.875),
        # lambda: model_18(num_classes=38, alpha=0.75),
        # lambda: model_18(num_classes=38, alpha=0.625),
        # lambda: model_18(num_classes=38, alpha=0.5),
        # lambda: model_18(num_classes=38, alpha=0.375),
        # lambda: model_18(num_classes=38, alpha=0.25),
        # lambda: model_18(num_classes=38, alpha=0.125),
        # todo 消融--模型结构
        # lambda: model_18(num_classes=38, alpha=0.5),
        # lambda: model_18_vallina(num_classes=38),
        # lambda :model_18_wo_cross_encoder(num_classes=38),
        lambda: model_18_wo_cross_layer(num_classes=38),

        # todo 消融--双分支结构
        # lambda: model18_only_spa(num_classes=38),
        # lambda: model18_only_freq(num_classes=38),
        # lambda: model18_only_spa_vallina(num_classes=38),
        # lambda: model18_only_freq_vallina(num_classes=38),
    ]
    model_names = [
        # todo 对比实验模型
        "ResNet18","DenseNet121",
        "WaferSegClassNet","WMPeleeNet","CWDR_model",
        "MERNet",
        "mobilevit_xxs","MSFTransformer",
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
    ]

    # results = {}
    # for i, factory in enumerate(model_factories):
    #     input_shape = (1, 1, 64, 64)
    #     result = test_model_output(factory, input_shape)
    #     results[f"Model {i + 1}"] = {
    #         "status": "Passed" if result else "Failed",
    #         "input_shape": input_shape,
    #     }


    print(f'numbers of model_names: {len(model_names)},model_factories: {len(model_factories)}')
    for i, (model_factory, model_name) in enumerate(zip(model_factories, model_names)):
        print(f"\n===== 开始训练 {model_name}=====")
        # 初始化模型
        model = model_factory()
        weight_init(model)
        model = model.to(device)

        # 定义优化器和损失函数（可根据模型调整参数）
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        lr_schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)  # or CosineAnnealingLR

        start_time = datetime.now().strftime(f"{model_name}_mc_%Y%m%d_%H%M%S")

        # 训练模型
        print(f'Start training at {start_time}')
        trained_model = train_model(
            model, criterion, optimizer, lr_schedule,
            start_time, num_epochs=num_epochs
        )
        print(f'Finished training {model_name} at {datetime.now().strftime("%Y%m%d_%H%M%S")}')
        test_my_model(trained_model, classes, test_DataLoader)
        #test_my_model_with_tsne(trained_model, classes, test_DataLoader, model_name)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if i < 5:
            time.sleep(30)

    time.sleep(600)
    os.system("/usr/bin/shutdown")
