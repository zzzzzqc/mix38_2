from model import model18
import torch
import os
from atten_heatmap import *
from test_image import *

# print(model18(1,38))


target_layers = [
    'conv_out_1',
    'SpatialAttention',
    'conv_out_2',
    'CoordinateAttention',
    'mvit',
    'ChannelAttention'
]

from atten_heatmap import AttentionHeatmapGenerator
# from your_model_file import model18

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model18(in_channels=1, num_classes=38).to(device)
# model.load_state_dict(torch.load('path/to/your/model18_weights.pth', map_location=device))

analyzer = AttentionHeatmapGenerator(model)
analyzer.register_hooks(layer_names=target_layers)

# 批量处理图片
input_dir = 'test_samples'
output_dir = 'heatmap_results'
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(input_dir, filename)
        save_path = os.path.join(output_dir, f"{filename}_multi_layer.png")
        pred_class, heatmaps = predict_with_heatmap(
            model, image_path, device, analyzer,
            save_path=save_path, show=False
        )
        print(f"Processed {filename}, predicted: {pred_class}")