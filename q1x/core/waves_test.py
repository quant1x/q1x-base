#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project : q1x-base
@Package : core
@File    : waves_test.py
@Author  : wangfeng
@Date    : 2025/7/29 13:51
@Desc    : 波浪检测 - 测试
"""
import numpy as np
from matplotlib import pyplot as plt

from q1x.core.waves import *

def build_wave_segments(data_list, peaks, valleys):
    """
    根据波峰波谷构建波浪段
    返回: [(start_idx, end_idx, is_rising)]
    """
    # 合并所有关键点并去重排序
    key_indices = sorted(set([0] + peaks + valleys + [len(data_list)-1]))

    segments = []
    for i in range(len(key_indices)-1):
        start = key_indices[i]
        end = key_indices[i+1]
        is_rising = data_list[end] > data_list[start]
        segments.append((start, end, is_rising))

    return segments

def detect_complete_wave_structure(high_list, low_list):
    """检测完整波浪结构（包含递归检测）"""
    high_list = high_list.tolist() if isinstance(high_list, np.ndarray) else list(high_list)
    if len(high_list) < 3:
        return []
    low_list = low_list.tolist() if isinstance(low_list, np.ndarray) else list(low_list)
    # 检测主波浪
    peaks, valleys = detect_peaks_and_valleys(high_list, low_list)
    main_segments = build_wave_segments(high_list, peaks, valleys)

    # 转换主波浪段
    all_segments = [
        (start, end, 0, is_rising)
        for start, end, is_rising in main_segments
    ]

    # 递归检测次级波浪
    for start, end, _ in main_segments:
        if end - start >= 3:
            sub_waves = detect_wave_recursive(high_list, start, end, 1)
            all_segments.extend(sub_waves)

    return sorted(all_segments, key=lambda x: (x[2], x[0]))

def detect_wave_recursive(data, start_idx, end_idx, level=0):
    """递归检测次级波浪"""
    if end_idx - start_idx < 3:
        return []

    subsection = data[start_idx:end_idx+1]
    peaks, valleys = detect_peaks_and_valleys(subsection)
    segments = build_wave_segments(subsection, peaks, valleys)

    if not segments:
        return []

    # 转换到全局索引
    global_segments = []
    for local_start, local_end, is_rising in segments:
        global_start = start_idx + local_start
        global_end = start_idx + local_end
        global_segments.append((global_start, global_end, level+1, is_rising))

    return global_segments

def plot_wave_structure(data, wave_segments):
    """绘制波浪结构图"""
    plt.figure(figsize=(14, 8))
    high_list = data.tolist() if isinstance(data, np.ndarray) else list(data)
    x = range(len(high_list))

    # 绘制原始数据
    plt.plot(x, high_list, 'k-', alpha=0.5, label='原始数据')
    plt.scatter(x, high_list, c='black', s=20, alpha=0.7)

    # 颜色配置
    color_map = {
        (0, True): 'red',      # 主上升
        (0, False): 'green',   # 主下降
        (1, True): '#ff9999',  # 次级上升
        (1, False): '#99cc99'  # 次级下降
    }

    # 绘制波浪段
    for start, end, level, is_rising in wave_segments:
        color = color_map.get((level, is_rising), 'gray')
        plt.plot([start, end], [high_list[start], high_list[end]],
                 color=color, linewidth=3-level, alpha=0.8)

    plt.title('波浪结构检测（三阶段严格实现）', fontsize=14)
    plt.xlabel('索引')
    plt.ylabel('值')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

# 测试
if __name__ == "__main__":
    # 测试数据
    test_data = np.array([2, 9, 1, 5, 3, 4, 2, 7, 4, 6, 2, 9, 4])

    # 检测波浪结构
    waves = detect_complete_wave_structure(test_data, test_data)

    # 打印结果
    print("波浪段检测结果：")
    for i, (start, end, level, is_rising) in enumerate(waves):
        direction = "↑" if is_rising else "↓"
        print(f"段{i+1}: 索引{start}→{end} (L{level}, {direction})")

    # 绘制图形
    plot_wave_structure(test_data, waves)