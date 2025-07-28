import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_curve, auc
# from scipy import interp
from itertools import cycle



def plt_roc(num_class,x_dict, y_dict, auc_dict):
    # 绘制所有类别平均的roc曲线
    fpr = x_dict["macro"]
    tpr = y_dict["macro"]
    roc_auc = auc_dict["macro"]
    plt.figure(figsize=(8,8))
    lw = 2
    plt.plot(fpr, tpr,
                label='macro-average ROC curve (area = {0:0.2f})'
                    ''.format(roc_auc),
                color='navy', linestyle=':', linewidth=4)
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'b', 'g', 'r', 'm', 'y', 'lime'])
    for i, color in zip(range(num_class), colors):
        plt.plot(x_dict[i], y_dict[i], color=color, lw=lw,
                    label='ROC curve of class {0} (area = {1:0.2f})'
                        ''.format(i, auc_dict[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Each Action')
    plt.legend(loc="best",prop={'size':8})
    plt.savefig('./result' + '/' + 'roc.eps')
    plt.ion()
    plt.close()


def roc_value(num_class, score_list, real):
    score_array = np.array(score_list)
    label_tensor = torch.tensor(real)
    label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
    label_onehot = torch.zeros(label_tensor.shape[0], num_class)
    label_onehot.scatter_(dim=1, index=label_tensor, value=1)
    label_onehot = np.array(label_onehot)
    # 调用sklearn库，计算每个类别对应的fpr和tpr
    fpr_dict = dict()
    tpr_dict = dict()
    roc_auc_dict = dict()
    for i in range(num_class):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(label_onehot[:, i], score_array[:, i])
        roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])

    # macro
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(num_class)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_class):
        mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
    # Finally average it and compute AUC
    mean_tpr /= num_class
    fpr_dict["macro"] = all_fpr
    tpr_dict["macro"] = mean_tpr
    roc_auc_dict["macro"] = auc(fpr_dict["macro"], tpr_dict["macro"])
    plt_roc(num_class, fpr_dict, tpr_dict, roc_auc_dict)


    #保存roc曲线的数据
    #创建一个文件夹，将保存csv文件
    npy_path = os.path.join(os.getcwd(), 'ROC_npy') 
    if not os.path.exists(npy_path):
        os.makedirs(npy_path)
    csv_path = os.path.join(os.getcwd(), 'ROC_csv') 
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    #将fpr_dict和tpr_dict存储为csv文件
    np.save(npy_path + "/" + "fpr.npy", fpr_dict["macro"])
    np.save(npy_path + "/" + "tpr.npy", tpr_dict["macro"])
    # path处填入npy文件具体路径
    fpr_file = np.load(npy_path + "/" + "fpr.npy")
    tpr_file = np.load(npy_path + "/" + "tpr.npy")
    # 将npy文件的数据格式转化为csv格式
    fpr_to_csv = pd.DataFrame(data = fpr_file)
    tpr_to_csv = pd.DataFrame(data = tpr_file)
    # 存入具体目录下的np_to_csv.csv 文件  
    fpr_to_csv.to_csv(csv_path + "/" + "fpr.csv")
    tpr_to_csv.to_csv(csv_path + "/" + "tpr.csv")
    return roc_auc_dict["macro"]

