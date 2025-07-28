# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量输出生成器 - 用于测试输出显示、日志记录等需要大量输出的场景
"""

import argparse
import time
from datetime import datetime


def generate_output(lines=10000, prefix="Line", delay=0):
    """
    生成指定数量的输出行

    Args:
        lines: 输出的总行数
        prefix: 每行的前缀文本
        delay: 每行输出的间隔时间(秒)，0表示无延迟
    """
    print(f"开始生成 {lines} 行输出... (当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")

    for i in range(1, lines + 1):
        # 生成带时间戳的输出行
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        line_content = f"[{timestamp}] {prefix} {i}: 这是第 {i} 行输出"

        print(line_content)

        # 刷新输出缓冲区，确保实时显示
        import sys
        sys.stdout.flush()

        # 延迟指定时间
        if delay > 0:
            time.sleep(delay)

    print(f"输出生成完成! (共 {lines} 行，结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量输出生成器")
    parser.add_argument("-l", "--lines", type=int, default=20000, help="输出的行数，默认10000行")
    parser.add_argument("-p", "--prefix", type=str, default="Line", help="每行的前缀文本")
    parser.add_argument("-d", "--delay", type=float, default=0, help="每行输出的间隔时间(秒)，默认无延迟")

    args = parser.parse_args()

    generate_output(
        lines=args.lines,
        prefix=args.prefix,
        delay=args.delay
    )