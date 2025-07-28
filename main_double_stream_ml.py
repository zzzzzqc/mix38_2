# encoding: utf-8
from __future__ import print_function, division

import copy

import torch.nn as nn
import math
import os
import torch.optim as optim
from torch.optim import lr_scheduler
from dataloder import *
from DSHF_vallina_conv import model_18_vallina, model18_only_spa_vallina, model18_only_freq_vallina
from DSHF_oct_conv import model_18, model18_only_spa, model18_only_freq
from models.cwdr import CWDR_model
from torch.utils.data import DataLoader
from plotConfusionMatrix import plot_Matrix, convert_labels
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import confusion_matrix
from datetime import datetime
import torch.nn.init as init
import warnings
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.simplefilter(action='ignore')


def test_my_model_ml(model, classes, dataloader_valid):
    model.eval()  # 将模型设置为评估模式
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for img, labels in dataloader_valid:
            img = img.cuda()
            labels = labels.cuda()
            outputs = model(img)
            predicted = torch.sigmoid(outputs) > 0.5  # for any label, only has 0/1 two choice
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_labels_bull = [labels.astype(bool) for labels in all_labels]
            all_labels = all_labels_bull

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    print(f'Accuracy: {accuracy:.8f}')
    print(f'Precision: {precision:.8f}')
    print(f'Recall: {recall:.8f}')
    print(f'F1 Score: {f1:.8f}')
    all_labels, all_preds = convert_labels(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds, labels=[i for i in range(len(classes))])
    plot_Matrix(cm, classes, title='Normalized confusion matrix', save_dir='confusion_matrix')


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

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss, running_corrects, total_samples = 0.0, 0, 0
            all_preds, all_labels = [], []

            for inputs, labels in data_loaders[phase]:
                # 数据移动到设备上
                inputs = inputs.to(device)
                labels = labels.to(device)
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
                predicted = torch.sigmoid(outputs) > 0.5
                # print(predicted)
                running_corrects += (predicted == labels.byte()).sum().item()

                total_samples += inputs.size(0)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            all_labels_bull = [labels.astype(bool) for labels in all_labels]
            all_labels = all_labels_bull

            all_preds, all_labels = np.array(all_preds), np.array(all_labels)
            accuracy = accuracy_score(all_labels, all_preds)
            precision = precision_score(all_labels, all_preds, average='macro')  # 'samples', 'micro', None, 'bina'
            recall = recall_score(all_labels, all_preds, average='macro')
            f1 = f1_score(all_labels, all_preds, average='macro')
            epoch_loss = running_loss / total_samples

            if phase == 'train':
                train_loss.append(epoch_loss)
                # 可以选择记录不同的F1分数
                train_f1_micro.append(f1)
                scheduler.step()
            else:
                test_loss.append(epoch_loss)
                test_f1_micro.append(f1)

            print('{} Loss: {:.4f} {} Acc: {:.4f} pre: {:.4f} rec: {:.4f} f1: {:.4f} '.format(
                phase, epoch_loss, phase, accuracy, precision, recall, f1))

            if phase == 'test' and accuracy >= best_acc:
                best_acc = accuracy
                # best_model_wts = model.state_dict()
                best_model_wts = copy.deepcopy(model.state_dict())  # 使用深拷贝保存当前模型参数
            if phase == 'test' and accuracy > 0.983:
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
#         # m 中的 weight，bias 其实都是 Variable，为了能学习参数以及后向传播
#         m.weight.data.fill_(1)
#         m.bias.data.zero_()

# def weight_init(m):
#     """Kaiming 初始化 + BatchNorm 初始化"""
#     # init.xavier_normal_(m.weight)
#     if isinstance(m, nn.Conv2d):
#         # Kaiming 初始化
#         init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#         if m.bias is not None:  # 如果卷积层有偏置项，初始化为0
#             init.constant_(m.bias, 0)
#     elif isinstance(m, nn.BatchNorm2d):
#         # BatchNorm 初始化：权重为1，偏置为0
#         init.constant_(m.weight, 1)
#         init.constant_(m.bias, 0)

def weight_init(m):
    """Xavier 初始化 + BatchNorm 初始化"""
    if isinstance(m, nn.Conv2d):
        # Xavier 正态分布初始化（fan_in + fan_out 均值为0，方差为2/(fan_in+fan_out)）
        init.xavier_normal_(m.weight, gain=init.calculate_gain('relu'))
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        # BatchNorm 初始化保持不变：权重为1，偏置为0
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


if __name__ == '__main__':
    # 定义训练参数
    dataset_path = 'dataset_mix38/Wafer_Map_Datasets.npz'
    num_epochs = 100
    batch_size = 128
    test_size = 0.2
    target_size = (64, 64)  # 图片尺寸

    all_data = data_loader(dataset_path,
                           vis=True,
                           target_size=target_size,
                           interpolation_mode='bilinear')
    train_dataset, test_dataset = all_data.get_data()
    train_DataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_DataLoader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    data_loaders = {'train': train_DataLoader,
                    'test': test_DataLoader}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classes = ('Normal', 'Center', 'Donut', 'Edge_Loc', 'Edge_Ring', 'Loc', 'Near_Full', 'Scratch', 'Random', 'C+EL',
               'C+ER', 'C+L', 'C+S', 'D+EL', 'D+ER', 'D+L', 'D+S', 'EL+L', 'EL+S', 'ER+L', 'ER+S', 'L+S',
               'C+EL+L', 'C+EL+S', 'C+ER+L', 'C+ER+S', 'C+L+S', 'D+EL+L', 'D+EL+S', 'D+ER+L', 'D+ER+S',
               'D+L+S', 'EL+L+S', 'ER+L+S', 'C+L+EL+S', 'C+L+ER+S', 'D+L+EL+S', 'D+L+ER+S')
    model_factories = [
        # lambda: model_18(num_classes=38),
        # lambda: model18_only_spa(num_classes=38),
        # lambda: model18_only_freq(num_classes=38),
        # lambda: model_18_vallina(num_classes=38),
        # lambda: model18_only_spa_vallina(num_classes=38),
        # lambda: model18_only_freq_vallina(num_classes=38),
        lambda: CWDR_model(num_of_classes=8)
    ]

    model_names = [
        # "model_18", "model18_only_spa", "model18_only_freq",
        # "model_18_vallina", "model18_only_spa_vallina", "model18_only_freq_vallina",
        "CWDR_model"
    ]
    for i, (model_factory, model_name) in enumerate(zip(model_factories, model_names)):
        print(f"\n===== 开始以多标签训练 {model_name} =====")

        # 初始化模型
        model = model_factory()
        weight_init(model)
        model = model.to(device)

        # 定义优化器和损失函数（可根据模型调整参数）
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        lr_schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

        # 生成唯一时间戳
        start_time = datetime.now().strftime(f"{model_name}_ml_%Y%m%d_%H%M%S")

        # 训练模型
        print(f'Start training at {start_time}')
        trained_model = train_model(
            model, criterion, optimizer, lr_schedule,
            start_time, num_epochs=num_epochs
        )
        print(f'Finished training {model_name} at {datetime.now().strftime("%Y%m%d_%H%M%S")}')
        test_my_model_ml(trained_model, classes, test_DataLoader)
        # 释放显存（可选，顺序训练时帮助释放内存）
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 间隔一段时间（避免连续训练导致过热）
        if i < 5:  # 非最后一个模型
            time.sleep(30)  # 休息5分钟

    time.sleep(600)

    # os.system("/usr/bin/shutdown")
