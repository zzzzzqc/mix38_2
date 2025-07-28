import numpy as np
import distinctipy
import os
import json


class ColorManager:
    _instance = None  # 单例实例

    def __new__(cls, n_colors=None, seed=42, cache_file=None):
        # 如果实例不存在，创建新实例
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, n_colors=None, seed=42, cache_file=None):
        # 确保初始化只执行一次
        if not hasattr(self, '_initialized') or not self._initialized:
            self.seed = seed
            self.n_colors = n_colors
            self.colors = None
            self.cache_file = cache_file or "color_cache.json"
            self._initialized = True

    def _save_to_file(self):
        """将颜色数组保存到文件"""
        if self.colors is not None:
            try:
                # 转换为列表以便JSON序列化
                color_list = self.colors.tolist()
                data = {
                    "seed": self.seed,
                    "n_colors": self.n_colors,
                    "colors": color_list
                }
                with open(self.cache_file, 'w') as f:
                    json.dump(data, f)
                print(f"颜色已保存到 {self.cache_file}")
            except Exception as e:
                print(f"保存颜色到文件时出错: {e}")

    def _load_from_file(self):
        """从文件加载颜色数组"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                # 检查种子和颜色数量是否匹配
                if data.get("seed") == self.seed and data.get("n_colors") == self.n_colors:
                    self.colors = np.array(data["colors"])
                    print(f"颜色已从 {self.cache_file} 加载")
                    return True
                else:
                    print(
                        f"文件中的颜色配置与当前设置不匹配 (种子: {data.get('seed')} vs {self.seed}, 数量: {data.get('n_colors')} vs {self.n_colors})")
            except Exception as e:
                print(f"从文件加载颜色时出错: {e}")
        return False

    def get_colors(self, n_colors=None):
        """获取或生成颜色数组"""
        # 如果未指定颜色数量，使用初始化时的数量
        if n_colors is None:
            n_colors = self.n_colors

        # 如果请求的颜色数量与当前缓存的不同，需要重新生成
        if self.colors is not None and len(self.colors) != n_colors:
            self.colors = None

        # 如果颜色数组未生成，尝试从文件加载或重新生成
        if self.colors is None:
            self.n_colors = n_colors
            if not self._load_from_file():
                # 文件加载失败或不存在，生成新的颜色
                old_state = np.random.get_state()
                try:
                    np.random.seed(self.seed)
                    self.colors = np.array(distinctipy.get_colors(n_colors))
                    self._save_to_file()  # 保存到文件
                finally:
                    np.random.set_state(old_state)

        return self.colors


def load_colors_from_json(file_path):
    """从JSON文件加载颜色数组"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        # 提取颜色数据
        colors = np.array(data.get("colors", []))

        # 可选：验证是否包含必要信息
        if len(colors) == 0:
            print(f"错误：文件 '{file_path}' 中未找到颜色数据")
            return None

        # 可选：打印颜色配置信息
        seed = data.get("seed")
        n_colors = data.get("n_colors")
        print(f"从 {file_path} 加载了 {len(colors)} 种颜色")
        if seed is not None:
            print(f"颜色种子: {seed}")
        if n_colors is not None:
            print(f"预期颜色数量: {n_colors}")

        return colors

    except Exception as e:
        print(f"加载颜色时出错: {e}")
        return None


# 使用示例
if __name__ == "__main__":
    # 创建颜色管理器实例，指定需要38种颜色，并保存到指定文件
    color_manager = ColorManager(n_colors=38, seed=42, cache_file="colors_38.json")

    # 第一次调用会生成颜色并保存到文件
    colors1 = color_manager.get_colors()

    # 后续调用会从文件加载颜色（如果文件存在且配置匹配）
    colors2 = color_manager.get_colors()

    # 验证两次获取的颜色是否相同
    print(np.array_equal(colors1, colors2))  # 输出: True

    # 创建新的管理器实例，请求10种颜色
    color_manager_10 = ColorManager(n_colors=10, seed=42, cache_file="colors_10.json")
    colors3 = color_manager_10.get_colors()
    print(len(colors3))  # 输出: 10
    colors = load_colors_from_json("colors_38.json")

    if colors is not None:
        print(f"颜色数组形状: {colors.shape}")  # 输出: (38, 3)
        print(f"前3种颜色: {colors[:3]}")