from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from models.resnet import ResNet18
from models.cwdr import CWDR_model
from plotConfusionMatrix import plot_Matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from datetime import datetime
import warnings
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
from get_colors import load_colors_from_json
import matplotlib
matplotlib.use('Agg')  # 必须在导入 pyplot 之前设置


seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
warnings.simplefilter(action='ignore')


def visualize_tsne(model, test_loader, device, num_classes):
    """
    使用t-SNE可视化模型特征空间
    :param model: 训练好的模型
    :param test_loader: 测试数据加载器
    :param device: 计算设备 (cuda/cpu)
    :param num_classes: 类别数量
    """
    model.eval()
    features = []
    labels = []

    # 1. 提取特征和标签
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            # 获取特征向量（通常在分类层之前）
            outputs = model(images, return_features=True)  # 需要模型支持返回特征
            features.append(outputs.cpu().numpy())
            labels.append(targets.numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    # 2. 随机采样（可选，大数据集时）
    if len(features) > 1000:
        indices = np.random.choice(len(features), 1000, replace=False)
        features = features[indices]
        labels = labels[indices]

    # 3. 应用t-SNE降维
    tsne = TSNE(
        n_components=2,  # 降维到2D
        perplexity=30,  # 典型值5-50，根据数据量调整
        n_iter=1000,  # 迭代次数
        random_state=42  # 可重复性
    )
    features_2d = tsne.fit_transform(features)

    # 4. 可视化
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        features_2d[:, 0],
        features_2d[:, 1],
        c=labels,
        cmap=plt.cm.get_cmap('tab10', num_classes),
        alpha=0.7,
        s=15  # 点大小
    )

    # 添加颜色条和图例
    plt.colorbar(scatter, ticks=range(num_classes))
    plt.title('t-SNE Visualization of Feature Space')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')

    # 保存和显示
    plt.savefig('tsne_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()


class FeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.features = []
        self.labels = []
        # after avg_pool layer
        self.hook = self.model.avgpool.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features.append(output.detach().cpu())

    def add_labels(self, labels):
        self.labels.append(labels.cpu())

    def get_features(self):
        """get features and flatten"""
        features = torch.cat(self.features, dim=0)
        return features.squeeze(-1).squeeze(-1)  # [B, C, 1, 1] -> [B, C]

    def get_labels(self):
        return torch.cat(self.labels, dim=0)

    def remove_hook(self):
        self.hook.remove()

    def reset(self):
        self.features = []
        self.labels = []


def test_my_model_with_tsne(model, classes, dataloader_valid, model_name=""):
    model.eval()  # 将模型设置为评估模式

    # 创建特征提取器
    feature_extractor = FeatureExtractor(model)

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

            # 保存特征对应的真实标签
            feature_extractor.add_labels(labels)

    # 计算性能指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    print(f'Accuracy: {accuracy:.8f}')
    print(f'Precision: {precision:.8f}')
    print(f'Recall: {recall:.8f}')
    print(f'F1 Score: {f1:.8f}')

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds, labels=[i for i in range(len(classes))])
    plot_Matrix(cm, classes, title='Confusion matrix', save_dir='confusion_matrix')

    # 1. 获取特征和标签
    features = feature_extractor.get_features().numpy()
    labels = feature_extractor.get_labels().numpy()

    # 2. 随机采样（大数据集时）
    if len(features) > 2000:
        indices = np.random.choice(len(features), 2000, replace=False)
        features = features[indices]
        labels = labels[indices]

    # 3. 应用t-SNE
    print("Running t-SNE...")
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        n_iter=1000,
        random_state=42,
        learning_rate=200
    )
    tsne_results = tsne.fit_transform(features)

    # 4. 创建保存目录
    tsne_dir = os.path.join('tsne_results', datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(tsne_dir, exist_ok=True)

    # 5. 可视化
    plt.figure(figsize=(12, 10))
    from matplotlib.colors import ListedColormap

    # len_colors = len(classes)
    colors = load_colors_from_json("colors_38.json")
    custom_cmap = ListedColormap(colors)

    # 可视化
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1],
                          c=labels, cmap=custom_cmap, alpha=0.7)

    # 添加颜色条
    cbar = plt.colorbar(scatter, ticks=range(len(classes)))
    cbar.set_label('t-SNE Visualization: Class-color Mapping')

    # 生成 C1-C38 标签
    custom_labels = [f"C{i + 1}" for i in range(len(classes))]  # 生成 C1 到 C38
    cbar.set_ticklabels(custom_labels)  # 替换颜色条刻度标签

    # 可选：调整字体大小
    cbar.ax.tick_params(labelsize=10)

    # 设置标题（包含alpha和mode信息）
    title = f't-SNE Visualization of Feature Space\n{model_name}'

    plt.title(title)

    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')

    # 6. 保存和显示
    filename = f'tsne_{model_name}'
    filename += '.png'

    save_path = os.path.join(tsne_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"t-SNE visualization saved as {save_path}")
    plt.close()  # 关闭图形避免内存泄漏

    # 保存特征数据供后续分析
    np.savez(os.path.join(tsne_dir, f'features_{model_name}.npz'),
             features=features,
             labels=labels,
             tsne_results=tsne_results)

    # 移除钩子
    feature_extractor.remove_hook()

    return accuracy, precision, recall, f1


# 1. 定义简化模型（适配特征提取）
class SimpleModel(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleModel, self).__init__()
        self.base = models.resnet18(pretrained=False)
        self.base.fc = nn.Identity()  # 保留特征向量
        self.avgpool = self.base.avgpool  # 注册钩子的层
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x, return_features=False):
        features = self.base(x)  # [B, 512]
        if return_features:
            return features
        return self.classifier(features)


from torch.utils.data import DataLoader, TensorDataset


# 2. 创建测试数据
def create_dummy_data(batch_size=32, num_samples=100, input_size=(3, 64, 64), num_classes=10):
    # 随机生成图像数据和标签
    X = torch.rand(num_samples, *input_size)
    y = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


# 使用示例
if __name__ == "__main__":
    # 设置参数
    num_classes = 38
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化模型和数据
    model = SimpleModel(num_classes).to(device)
    test_loader = create_dummy_data(num_samples=7200, num_classes=38)  #

    # 测试函数调用
    classes = [f'Class {i}' for i in range(num_classes)]
    metrics = test_my_model_with_tsne(
        model=model,
        classes=classes,
        dataloader_valid=test_loader,
        model_name="SimpleResNet"
    )

    # print(f"Test Metrics - Acc: {metrics[0]:.4f}, F1: {metrics[3]:.4f}")

    # 验证文件生成
    output_dir = os.path.join('tsne_results', os.listdir('tsne_results')[-1])
    print(f"Generated files in: {output_dir}")
    pass
    import distinctipy

    # 第一次调用
    # colors1 = distinctipy.get_colors(38)
    # print(colors1)
    #
    # # 第二次调用
    # colors2 = distinctipy.get_colors(38)
    # print(colors2)
    #
    # # 结果通常不同
    # print(np.array_equal(colors1, colors2))  # 输出 False