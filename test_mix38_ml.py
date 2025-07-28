# encoding: utf-8
from __future__ import print_function, division

import copy

import torch.nn as nn
import math
import os
import torch.optim as optim
from torch.optim import lr_scheduler
from dataloder import *
from DSHF_Net import model18, model18_only_freq, model18_only_spa, model18_add, model18_cross_atten, model18_mul
from models.resnet import ResNet18
from torch.utils.data import DataLoader
from plotConfusionMatrix import plot_Matrix,convert_labels
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import confusion_matrix
from datetime import datetime
import torch.nn.init as init
import warnings

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
            predicted = torch.sigmoid(outputs) > 0.5
            # 保存预测结果和真实标签
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



if __name__ == '__main__':
    
    dataset_path = 'dataset_mix38/Wafer_Map_Datasets.npz'
    num_epochs = 300
    warmup_switch = False  # 控制是否启用warmup
    warmup_epochs = 20  # warmup的epoch数
    cycle_epochs = 200  # 每个余弦周期的epoch数（T_max）
    batch_size = 128
    test_size = 0.2
    

    all_data = data_loader(dataset_path, vis=True)
    train_dataset, test_dataset = all_data.get_data()
    train_DataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_DataLoader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    data_loaders = {'train': train_DataLoader,
                    'test': test_DataLoader}

    classes = ('Normal', 'Center', 'Donut', 'Edge_Loc', 'Edge_Ring', 'Loc', 'Near_Full', 'Scratch', 'Random', 'C+EL',
               'C+ER', 'C+L', 'C+S', 'D+EL', 'D+ER', 'D+L', 'D+S', 'EL+L', 'EL+S', 'ER+L', 'ER+S', 'L+S',
               'C+EL+L', 'C+EL+S', 'C+ER+L', 'C+ER+S', 'C+L+S', 'D+EL+L', 'D+EL+S', 'D+ER+L', 'D+ER+S',
               'D+L+S', 'EL+L+S', 'ER+L+S', 'C+L+EL+S', 'C+L+ER+S', 'D+L+EL+S', 'D+L+ER+S')

    # label_keys = ["Center, ", "Donut, ", "Edge_Loc, ", "Edge_Ring, ", 
    #                        "Loc, ", "Near_Full, ", "Scratch, ", "Random, "]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = ResNet18(num_class = 8) # todo 用于验证训练流程与测试流程是否正确，已验证
    # net = model18(num_classes=8)# 这里对应多标签分类中的8中单类别，8位全为0代表无缺陷

    #net = resnet.ResNet18(9)
    # net = model18()
    # net = model18_add(num_classes=38)
    # net= model18_cross_atten(num_classes=38)
    # net = model18_only_spa(num_classes=38)
    # net = model18_only_freq(num_classes=38)
    net.to(device)
    net.load_state_dict(torch.load('result\\20250522_163842\\20250522_163842_model_5.pth'), True)

    test_my_model_ml(net, classes, test_DataLoader)
    # test_model(net, classes)
