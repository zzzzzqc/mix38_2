import torch
import torch.nn as nn

class VGG(nn.Module):
    def __init__(self, vgg_name, cfg, num_classes=10, bn=False, in_channels=3):
        super(VGG, self).__init__()
        self.in_channels = in_channels
        self.vgg_base = self.make_layer(cfg, bn)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))  # 自适应池化层
        self.fc1 = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )
        self.fc3 = nn.Linear(4096, num_classes)

    def make_layer(self, cfg, bn=False):
        layers = []
        in_channels = self.in_channels
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d((2, 2), stride=2)]
            else:
                out_channels, s = v.strip().split('_')
                out_channels, s = int(out_channels), int(s)
                padding = 1 if s > 1 else 0
                if bn:
                    layers += [nn.Conv2d(in_channels, out_channels, (s, s), padding=padding),
                               nn.BatchNorm2d(out_channels),
                               nn.ReLU(inplace=True)]
                else:
                    layers += [nn.Conv2d(in_channels, out_channels, (s, s), padding=padding),
                               nn.ReLU(inplace=True)]
                in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.vgg_base(x)
        x = self.adaptive_pool(x)  # 使用自适应池化层
        # print(x.shape)  # 确认特征图形状
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


cfg = {
    'vgg11_A': ['64_3', 'M',
                '128_3', 'M',
                '256_3', '256_3', 'M',
                '512_3', '512_3', 'M',
                '512_3', '512_3', 'M'],
    'vgg13_B': ['64_3', '64_3', 'M',
                '128_3', '128_3', 'M',
                '256_3', '256_3', 'M',
                '512_3', '512_3', 'M',
                '512_3', '512_3', 'M'],
    'vgg16_C': ['64_3', '64_3', 'M',
                '128_3', '128_3', 'M',
                '256_3', '256_3', '256_1', 'M',
                '512_3', '512_3', '512_1', 'M',
                '512_3', '512_3', '512_1', 'M'],
    'vgg16_D': ['64_3', '64_3', 'M',
                '128_3', '128_3', 'M',
                '256_3', '256_3', '256_3', 'M',
                '512_3', '512_3', '512_3', 'M',
                '512_3', '512_3', '512_3', 'M'],
    'vgg19_E': ['64_3', '64_3', 'M',
                '128_3', '128_3', 'M',
                '256_3', '256_3', '256_3', '256_3', 'M',
                '512_3', '512_3', '512_3', '512_3', 'M',
                '512_3', '512_3', '512_3', '512_3', 'M'],
}

def vgg16(num_classes=10,in_channels=3):
    model = VGG('vgg16_C',cfg['vgg16_C'], num_classes=num_classes, in_channels=in_channels)
    return model

if __name__ == '__main__':
    input_tensor = torch.randn((1, 1, 64, 64))  # 支持任意输入尺寸
    # vgg_name = 'vgg19_E'
    # model = VGG(vgg_name, cfg[vgg_name], num_classes=38, in_channels=1)
    model = vgg16(38,1)
    output = model(input_tensor)
    print(output.shape)  # 期望输出: torch.Size([1, 38])