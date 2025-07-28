import torch
# import cv2
import matplotlib.pyplot as plt


def generate_heatmap(feature_map, method='mean', resize_shape=None):
    """
    Args:
        feature_map: Tensor of shape [B, C, H, W]
        method: 'mean' 或 'max'，选择通道聚合方式
        resize_shape: 调整热图大小（可选）
    Returns:
        heatmap: 归一化的热图（0-1范围）
    """
    # 聚合通道维度
    if method == 'mean':
        heatmap = torch.mean(feature_map, dim=1)  # [B, H, W]
    elif method == 'max':
        heatmap, _ = torch.max(feature_map, dim=1)  # [B, H, W]
    else:
        raise ValueError("method 必须是 'mean' 或 'max'")

    # 归一化到 [0, 1]
    heatmap = heatmap - heatmap.min()
    heatmap = heatmap / (heatmap.max() + 1e-8)

    # 调整大小（可选）
    if resize_shape is not None:
        heatmap = torch.nn.functional.interpolate(
            heatmap.unsqueeze(1),
            size=resize_shape,
            mode='bilinear'
        ).squeeze(1)

    return heatmap.detach().cpu().numpy()


def visualize_heatmaps(model, input_image, layer_index=0, method='mean'):
    """
    Args:
        model: 加载的模型
        input_image: 输入图像 Tensor [1, C, H, W]
        layer_index: 选择层（0对应第2层，1对应第3层，2对应第4层）
        method: 'mean' 或 'max'
    """
    # 前向传播获取中间特征
    model.eval()
    with torch.no_grad():
        pred, intermediates = model(input_image)

    # 提取指定层的特征
    spa_feat = intermediates['spa'][layer_index]  # [B, C, H, W]
    freq_feat = intermediates['freq'][layer_index]

    # 生成热图
    spa_heatmap = generate_heatmap(spa_feat, method, resize_shape=input_image.shape[-2:])
    freq_heatmap = generate_heatmap(freq_feat, method, resize_shape=input_image.shape[-2:])

    # 转换为 RGB 图像
    input_img = input_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min())  # 归一化

    # 叠加热图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(input_img)
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    axes[1].imshow(input_img)
    axes[1].imshow(spa_heatmap[0], cmap='jet', alpha=0.5)
    axes[1].set_title(f'Spatial Branch (Layer {layer_index + 2})')
    axes[1].axis('off')

    axes[2].imshow(input_img)
    axes[2].imshow(freq_heatmap[0], cmap='jet', alpha=0.5)
    axes[2].set_title(f'Frequency Branch (Layer {layer_index + 2})')
    axes[2].axis('off')

    plt.show()

if __name__ == '__main__':
    # 加载模型和输入数据
    Model = None
    model = Model  # 替换为你的模型
    model.load_state_dict(torch.load('your_model.pth'))
    input_image = torch.randn(1, 3, 224, 224)  # 示例输入

    # 可视化第2层（layer_index=0）的热图
    visualize_heatmaps(model, input_image, layer_index=0, method='mean')

    pass



