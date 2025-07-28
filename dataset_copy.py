from sklearn.model_selection import train_test_split
from torchvision import transforms
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from label_maping import *


class WaferDefectDataset(Dataset):
    def __init__(self, images, labels, mode='train'):
        self.images = images
        self.labels = labels
        self.mode = mode
        self.label_mapping = label_mapping if label_mapping is not None else {}

        # Define transformations for train and test sets

        self.transform_resize64 = transforms.Compose([
            transforms.ToPILImage(),  # Convert numpy array to PIL Image
            transforms.Resize((64, 64)),
            transforms.ToTensor(),  # Convert PIL Image back to tensor
            # transforms.Normalize([0.5], [0.5])  # Normalize the image tensor
        ])
        self.transform_resize128 = transforms.Compose([
            transforms.ToPILImage(),  # Convert numpy array to PIL Image
            transforms.Resize((128, 128)),
            transforms.ToTensor(),  # Convert PIL Image back to tensor
            # transforms.Normalize([0.5], [0.5])  # Normalize the image tensor
        ])
        self.transform_resize192 = transforms.Compose([
            transforms.ToPILImage(),  # Convert numpy array to PIL Image
            transforms.Resize((192, 192)),
            transforms.ToTensor(),  # Convert PIL Image back to tensor
            # transforms.Normalize([0.5], [0.5])  # Normalize the image tensor
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].astype(np.float32) / 2.0  # 归一化到 [0, 1]
        original_label = ''.join(map(str, self.labels[idx]))  # 将one-hot编码转换为字符串形式

        # 更新标签为新的38分类标签
        new_label = self.update_label(original_label)  # new_label 是 one-hot 编码

        # 将 one-hot 编码转换为类别索引
        label_index = np.argmax(new_label)  # 通过one-hot编码获取类别索引
        label_tensor = torch.tensor(label_index, dtype=torch.long)  # 转换为张量，类型为 torch.long

        # 将图像数据转换为 PyTorch 张量，并添加通道维度 (C, H, W)
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # (1, 52, 52)
        image_tensor = image_tensor.squeeze(0)

        # 图像金字塔操作
        image_tensor_resize64 = self.transform_resize64(image_tensor.numpy())
        if image_tensor_resize64.dim() == 2:
            image_tensor_resize64 = image_tensor_resize64.unsqueeze(0)
        image_tensor_resize128 = self.transform_resize128(image_tensor.numpy())
        if image_tensor_resize128.dim() == 2:
            image_tensor_resize128 = image_tensor_resize128.unsqueeze(0)
        image_tensor_resize192 = self.transform_resize192(image_tensor.numpy())
        if image_tensor_resize192.dim() == 2:
            image_tensor_resize192 = image_tensor_resize192.unsqueeze(0)

        return image_tensor_resize192, image_tensor_resize128, image_tensor_resize64, label_tensor

    def update_label(self, original_label_str):
        """
        根据新的标签映射规则更新标签。
        :param original_label_str: 原始的one-hot编码标签 (8维)，作为字符串
        :return: 新的one-hot编码标签 (38维)
        """
        # 使用label_mapping将原始标签字符串映射到新的索引
        if original_label_str in self.label_mapping:
            new_indices = self.label_mapping[original_label_str]
        else:
            # 如果没有找到对应的映射，默认保持原样或设置为未知类别
            new_indices = [0]  # 假设0是“未知”或“未定义”的类别

        # 创建新的one-hot编码标签
        new_label = np.zeros(38, dtype=np.float32)
        new_label[new_indices] = 1.0  # 多标签情况下的处理

        return new_label


def get_train_and_test_dataset(datapath, test_size):
    data = np.load(datapath)

    # 获取数据和标签
    images = data['arr_0']
    labels = data['arr_1']

    # 划分数据集为训练集和测试集
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=test_size, random_state=42, stratify=labels)

    # 创建训练集和测试集实例
    train_dataset = WaferDefectDataset(train_images, train_labels, mode='train')
    test_dataset = WaferDefectDataset(test_images, test_labels, mode='test')
    return train_dataset, test_dataset


if __name__ == '__main__':
    train_dataset, test_dataset = get_train_and_test_dataset('dataset_mix38/Wafer_Map_Datasets.npz', 0.8)
    batch_size = 32
    num_workers = 4

    # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 测试数据加载器
    for phase in ['train', 'test']:
        dataloader = train_dataloader if phase == 'train' else test_dataloader

        for batch_idx, (images1, images2, images3, labels) in enumerate(dataloader):
            print(f"{phase.capitalize()} Batch {batch_idx + 1}")
            print("Images1 shape:", images1.shape)
            print("Images2 shape:", images2.shape)
            print("Images3 shape:", images3.shape)
            print("Labels shape:", labels.shape)
            print("Labels values:", labels)

            # 只打印前两个批次的数据
            if batch_idx >= 1:
                break