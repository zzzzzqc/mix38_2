from model import model18
import torch
import os
from atten_heatmap import *
from test_image import *

# 【关键修改】在这里填入上面新的层名列表
target_layers = [
    'conv_out_1.relu',
    'conv_out_2.relu',
    'channelconv1',
    'channelconv2',
    'channelconv3',
    'mvit.conv4.2'
]

from atten_heatmap import AttentionHeatmapGenerator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model18(in_channels=1, num_classes=38).to(device)
# model.load_state_dict(torch.load('path/to/your/model18_weights.pth', map_location=device))

analyzer = AttentionHeatmapGenerator(model)

# 【调试】注册前先打印一下，确认这些层在 model.named_modules() 里能找到
print("正在检查层是否存在...")
found_names = [name for name, _ in model.named_modules()]
for name in target_layers:
    if name in found_names:
        print(f"✅ 找到层: {name}")
    else:
        print(f"❌ 未找到层: {name}")

analyzer.register_hooks(layer_names=target_layers)

# 批量处理图片
input_dir = 'test_samples'
output_dir = 'heatmap_results'
os.makedirs(output_dir, exist_ok=True)

# 为了测试，先只处理一张图
test_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:1]

for filename in test_files:
    image_path = os.path.join(input_dir, filename)
    save_path = os.path.join(output_dir, f"{filename}_multi_layer.png")

    print(f"\n正在处理: {filename}")
    try:
        pred_class, heatmaps = predict_with_heatmap(
            model, image_path, device, analyzer,
            save_path=save_path, show=False
        )
        print(f"成功生成热图，预测类别: {pred_class}")
        print(f"生成的热图字典的 Key 有: {list(heatmaps.keys())}")  # 看看字典里有没有我们要的层
    except Exception as e:
        print(f"处理出错: {e}")
        import traceback

        traceback.print_exc()