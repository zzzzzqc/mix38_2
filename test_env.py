import torch
import torchvision
import numpy as np
import sklearn


def check_environment():
    print("当前环境信息：\n")

    # 检查 Python 和库的版本
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"TorchVision 版本: {torchvision.__version__}")
    print(f"NumPy 版本: {np.__version__}")
    print(f"Scikit-learn 版本: {sklearn.__version__}")

    # 检查 CUDA 是否可用
    print(f"\nCUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 版本（通过 PyTorch）: {torch.version.cuda}")
        print(f"当前 GPU 设备名称: {torch.cuda.get_device_name(0)}")
        print(f"GPU 数量: {torch.cuda.device_count()}")
    else:
        print("警告：CUDA 不可用，将使用 CPU 计算。")

    # 测试张量计算
    try:
        x = torch.rand(5, 3)
        print("\n测试随机张量生成:")
        print(x)

        # 测试 GPU
        if torch.cuda.is_available():
            x = x.to('cuda')
            print("\n张量已移动到 GPU:")
            print(x)
    except Exception as e:
        print("出现错误:", e)


if __name__ == "__main__":
    # check_environment()
    # import psutil
    # available_memory = psutil.virtual_memory().available
    # available_gb = available_memory / (1024 ** 3)  # 转换为 GB
    # print(available_gb)
    import matplotlib

    print(matplotlib.matplotlib_fname())