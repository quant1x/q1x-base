#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project : q1x-base
@Package : core
@File    : waves_stock.py
@Author  : wangfeng
@Date    : 2025/7/29 13:51
@Desc    : 股票K线波浪结构可视化（支持 DataFrame 输入）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from q1x.base import cache
# 导入你的核心函数
from q1x.core.waves import detect_peaks_and_valleys
from q1x.core.waves_test import (
    detect_complete_wave_structure,
    plot_wave_structure
)

# 假设 cache 是你的数据模块
# from your_module import cache


def analyze_stock_waves(code: str, max_bars: int = 100):
    """
    分析指定股票的K线波浪结构并绘图

    Args:
        code: 股票代码，如 'sz000158'
        max_bars: 最大显示K线索引数（避免图表过宽）
    """
    # 获取股票名称
    name = cache.stock_name(code)
    print(f'{name}({code})')

    # 加载K线数据（DataFrame）
    klines = cache.klines(code)

    if not isinstance(klines, pd.DataFrame) or len(klines) == 0:
        print("数据加载失败或为空")
        return

    if len(klines) < 3:
        print("数据不足，无法分析波浪结构")
        return

    # 确保字段名小写，并提取 high/low
    klines = klines.copy()
    klines.columns = [col.lower() for col in klines.columns]

    # 提取 high 和 low 序列
    high_list = klines['high'].astype(float).values  # → numpy array
    low_list = klines['low'].astype(float).values

    # 可选：限制显示范围（最近 N 根K线）
    if len(high_list) > max_bars:
        start_idx = -max_bars
        high_list = high_list[start_idx:]
        low_list = low_list[start_idx:]

    # 检测主波峰波谷
    peaks, valleys = detect_peaks_and_valleys(high_list, low_list)
    print(f"检测到波峰: {peaks}")
    print(f"检测到波谷: {valleys}")

    # 检测完整波浪结构（递归）
    waves = detect_complete_wave_structure(high_list, low_list)

    print("\n波浪段检测结果：")
    for i, (start, end, level, is_rising) in enumerate(waves):
        direction = "↑" if is_rising else "↓"
        print(f"  段{i + 1:2d}: {start:2d} → {end:2d} (L{level}, {direction})")

    # 绘制波浪结构图
    plot_wave_structure(f'{name}({code}) 波浪结构', high_list, waves, peaks=peaks, valleys=valleys)


# ========================
# 执行分析
# ========================
if __name__ == "__main__":
    code = 'sz000158'
    analyze_stock_waves(code, max_bars=60)  # 只显示最近60根K线