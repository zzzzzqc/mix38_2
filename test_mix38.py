# encoding: utf-8
from __future__ import print_function, division

import torch
import torch.nn as nn
import numpy as np
import math
import os
# import pandas as pd
from dataset import *
from DSHF_Net import model18, model18_only_freq, model18_only_spa, model18_add, model18_cross_atten, model18_mul
from models.resnet import ResNet18
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from plotConfusionMatrix import plot_Matrix, plot_Matrix_with_number
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import confusion_matrix
import random

random.seed(888)
import numpy

numpy.random.seed(888)


def test_my_model(model, classes, dataloader_valid):
    model.eval()  # 将模型设置为评估模
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
    plot_Matrix(cm, classes, title='Normalized confusion matrix', save_dir='confusion_matrix')


def load_dataset(file_path='',
                 batch_size=128,
                 test_size=0.2,
                 random_state=42):
    # 加载数据
    data = np.load(file_path)
    images = data['arr_0']
    labels = data['arr_1']
    # 打印数据的基本信息
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)

    # 划分数据集为训练集和测试集
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=test_size, random_state=random_state, stratify=labels)

    # 创建训练集和测试集实例
    train_dataset = WaferDefectDataset(train_images, train_labels)
    test_dataset = WaferDefectDataset(test_images, test_labels)

    # 创建 DataLoader
    train_DataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_DataLoader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f'训练集有{len(train_DataLoader)}个dataloder，测试集有{len(test_DataLoader)}个dataloder')

    return train_DataLoader, test_DataLoader


if __name__ == '__main__':
    dataset_path = 'dataset_mix38/Wafer_Map_Datasets.npz'
    num_epochs = 300
    warmup_switch = False  # 控制是否启用warmup
    warmup_epochs = 20  # warmup的epoch数
    cycle_epochs = 300  # 每个余弦周期的epoch数（T_max）
    batch_size = 128
    test_size = 0.2

    train_dataloader, test_dataloader = load_dataset(file_path=dataset_path,
                                                     batch_size=batch_size,
                                                     test_size=0.2,
                                                     random_state=42)
    data_loaders = {'train': train_dataloader,
                    'test': test_dataloader}
    classes = ('Normal', 'Center', 'Donut', 'Edge_Loc', 'Edge_Ring', 'Loc', 'Near_Full', 'Scratch', 'Random', 'C+EL',
               'C+ER', 'C+L', 'C+S', 'D+EL', 'D+ER', 'D+L', 'D+S', 'EL+L', 'EL+S', 'ER+L', 'ER+S', 'L+S',
               'C+EL+L', 'C+EL+S', 'C+ER+L', 'C+ER+S', 'C+L+S', 'D+EL+L', 'D+EL+S', 'D+ER+L', 'D+ER+S',
               'D+L+S', 'EL+L+S', 'ER+L+S', 'C+L+EL+S', 'C+L+ER+S', 'D+L+EL+S', 'D+L+ER+S')
    import warnings

    warnings.simplefilter(action='ignore')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ### 选择网络
    ### 选择网络
    #net = resnet.ResNet18(9)
    net = model18()
    # net = model18_add(num_classes=38)
    # net= model18_cross_atten(num_classes=38)
    # net = model18_only_spa(num_classes=38)
    # net = model18_only_freq(num_classes=38)
    net.to(device)
    net.load_state_dict(torch.load('result/20250427_070956/20250427_070956_model_128.pth'), True)  # 128.192.215

    test_my_model(net, classes, test_dataloader)
    # test_model(net, classes)
