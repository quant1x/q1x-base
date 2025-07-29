#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project : q1x-base
@Package : core
@File    : waves.py
@Author  : wangfeng
@Date    : 2025/7/29 13:17
@Desc    : 波浪检测
"""

from typing import List, Tuple


def find_monotonic_peaks(data_list, direction='left') -> list[int]:
    """
    单调上升波峰检测（仅返回峰值索引）

    参数:
        data_list: 输入数据列表
        direction: 检测方向 ('left' 或 'right')

    返回:
        峰值索引列表，按原始顺序排列
    """
    if not data_list:
        return []

    peaks = []
    start_idx = 0 if direction == 'left' else len(data_list) - 1
    end_idx = len(data_list) if direction == 'left' else -1
    step = 1 if direction == 'left' else -1

    prev_idx = start_idx
    prev_val = data_list[start_idx]

    for current_idx in range(start_idx + step, end_idx, step):
        current_val = data_list[current_idx]

        if current_val > prev_val:
            prev_idx, prev_val = current_idx, current_val
        elif peaks and prev_val == data_list[peaks[-1]]:
            continue
        else:
            peaks.append(prev_idx)

    # 处理最后一个元素
    if not peaks or prev_val > data_list[peaks[-1]]:
        peaks.append(prev_idx)

    return peaks if direction == 'left' else peaks[::-1]


def detect_peaks_and_valleys(data_list):
    """
    核心检测函数：仅返回波峰和波谷的索引
    返回: (peaks_indices, valleys_indices)
    """
    if not data_list:
        return [], []

    # 找出全局最大值（第一个出现的）
    max_val = max(data_list)
    max_idx = next(i for i, v in enumerate(data_list) if v == max_val)

    # 分割序列
    left_data = data_list[:max_idx+1]
    right_data = data_list[max_idx+1:]

    # 检测波峰（仅索引）
    PL = find_monotonic_peaks(left_data, 'left')
    PR = [max_idx+1+i for i in find_monotonic_peaks(right_data, 'right')]
    peaks = sorted(PL + PR)

    # 第二阶段：检测相邻高点间的低点（按原始顺序处理）
    valleys = []
    for i in range(len(peaks)-1):
        start, end = peaks[i], peaks[i+1]
        if end - start > 1:  # 开区间条件
            between = data_list[start+1:end]
            min_idx = start + 1 + between.index(min(between))
            valleys.append(min_idx)

    # 第三阶段：补全边界低点
    if peaks:  # 确保有高点才补边界
        # 左侧低点（第一个高点前）
        if peaks[0] > 0:
            valleys.append(data_list.index(min(data_list[:peaks[0]])))

        # 右侧低点（最后一个高点后）
        if peaks[-1] < len(data_list)-1:
            right_min = min(data_list[peaks[-1]+1:])
            valleys.append(data_list.index(right_min, peaks[-1]+1))

    return peaks, valleys

def build_wave_segments(data_list, peaks, valleys):
    """
    根据波峰波谷构建波浪段
    返回: [(start_idx, end_idx, is_rising)]
    """
    # 合并所有关键点并排序
    all_points = sorted(set([0] + peaks + valleys + [len(data_list)-1]))

    segments = []
    for i in range(len(all_points)-1):
        start = all_points[i]
        end = all_points[i+1]
        is_rising = data_list[end] > data_list[start]
        segments.append((start, end, is_rising))

    return segments

# --------------------------
# 视图层（结果格式化）
# --------------------------

def format_wave_results(data_list, peaks, valleys, segments):
    """
    格式化检测结果为可视图层使用的结构
    返回: (path_points, peak_points, valley_points)
    """
    path_indices = sorted(set(
        [0] + peaks + valleys + [len(data_list)-1]
    ))

    return (
        [(i, data_list[i]) for i in path_indices],  # path_points
        [(i, data_list[i]) for i in peaks],         # peak_points
        [(i, data_list[i]) for i in valleys]         # valley_points
    )


# --------------------------
# 接口函数
# --------------------------

def detect_main_wave_in_range(data_list):
    """保持原有接口的兼容函数"""
    peaks, valleys = detect_peaks_and_valleys(data_list)
    segments = build_wave_segments(data_list, peaks, valleys)
    path_points, peak_points, valley_points = format_wave_results(
        data_list, peaks, valleys, segments
    )
    return segments, path_points, peak_points, valley_points
