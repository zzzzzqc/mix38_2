import torch
from DSHF_oct_conv import model_18

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model_18(in_channels=1, num_classes=38).to(device)

print("正在扫描模型所有层...\n")
found_layers = []

# 遍历所有模块
for name, module in model.named_modules():
    # 过滤掉容器（Sequential），只打印具体的层
    if len(list(module.children())) == 0:
        # 或者为了看到你关心的模块，我们打印包含特定关键词的
        found_layers.append(name)

# 打印你目标列表附近的层名
target_keywords = ['conv_out', 'Attention', 'mvit', 'channel']
print("可能相关的层名：")
for name in found_layers:
    if any(key in name for key in target_keywords):
        print(f"- {name}")