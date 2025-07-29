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

import numpy as np


def _compare(a: float, b: float) -> int:
    """比较函数（完全复现C++的逻辑）"""
    return -1 if a < b else (1 if a > b else 0)


def find_peaks_valleys(high_list: np.ndarray, low_list: np.ndarray) -> Tuple[List[int], List[int]]:
    """
    完全复现C++的波峰波谷检测算法：
    1. 计算一阶差分
    2. 处理平台区域
    3. 通过二阶差分找极值点
    """
    n = len(high_list)
    if n != len(low_list) or n < 3:
        raise ValueError("输入序列长度不匹配或过短")

    # 1. 计算一阶差分
    diff_high = np.zeros(n, dtype=int)
    diff_low = np.zeros(n, dtype=int)

    for i in range(n - 1):
        diff_high[i] = _compare(high_list[i + 1], high_list[i])  # type: ignore
        diff_low[i] = _compare(low_list[i + 1], low_list[i])  # type: ignore

    # 2. 处理平台区域（差分值为0的情况）
    for i in range(n - 1):
        # 处理高价序列平台
        if diff_high[i] == 0:
            if i == 0:  # 首点平台
                for j in range(i + 1, n - 1):
                    if diff_high[j] != 0:
                        diff_high[i] = diff_high[j]
                        break
            elif i == n - 2:  # 末点平台
                diff_high[i] = diff_high[i - 1]
            else:  # 中间平台
                diff_high[i] = diff_high[i + 1]

        # 处理低价序列平台
        if diff_low[i] == 0:
            if i == 0:  # 首点平台
                for j in range(i + 1, n - 1):
                    if diff_low[j] != 0:
                        diff_low[i] = diff_low[j]
                        break
            elif i == n - 2:  # 末点平台
                diff_low[i] = diff_low[i - 1]
            else:  # 中间平台
                diff_low[i] = diff_low[i + 1]

    # 3. 识别波峰波谷
    peaks = []
    valleys = []

    for i in range(n - 1):
        d_high = diff_high[i + 1] - diff_high[i]
        d_low = diff_low[i + 1] - diff_low[i]

        # 波峰条件：高价差分由上升到下降（差分变化-2）
        if d_high == -2:
            peaks.append(i + 1)  # 注意索引偏移

        # 波谷条件：低价差分由下降到上升（差分变化+2）
        if d_low == 2:
            valleys.append(i + 1)

    return peaks, valleys


def find_monotonic_peaks(high_list, direction='left') -> list[int]:
    """
    单调上升波峰检测（仅返回峰值索引）

    参数:
        high_list: 输入数据列表
        direction: 检测方向 ('left' 或 'right')

    返回:
        峰值索引列表，按原始顺序排列
    """
    if not high_list:
        return []

    peaks = []
    start_idx = 0 if direction == 'left' else len(high_list) - 1
    end_idx = len(high_list) if direction == 'left' else -1
    step = 1 if direction == 'left' else -1

    prev_idx = start_idx
    prev_val = high_list[start_idx]

    for current_idx in range(start_idx + step, end_idx, step):
        current_val = high_list[current_idx]

        if current_val > prev_val:
            prev_idx, prev_val = current_idx, current_val
        elif peaks and prev_val == high_list[peaks[-1]]:
            continue
        else:
            peaks.append(prev_idx)

    # 处理最后一个元素
    if not peaks or prev_val > high_list[peaks[-1]]:
        peaks.append(prev_idx)

    return peaks if direction == 'left' else peaks[::-1]


def detect_peaks_and_valleys(high_list, low_list):
    """
    严格保持原有函数名和参数结构
    仅优化相邻高点处理逻辑：
    1. 保持原有三阶段检测流程
    2. 增加相邻高点过滤（n和n+1同时出现时保留更高的点）
    返回: (peaks_indices, valleys_indices)
    """
    # 数据预处理（完全保持原样）
    if isinstance(high_list, np.ndarray):
        high_list = high_list.tolist()
    if isinstance(low_list, np.ndarray):
        low_list = low_list.tolist()
    if not high_list:
        return [], []

    # 第一阶段：获取原始波峰（可能包含相邻点）
    max_val = max(high_list)
    max_idx = next(i for i, v in enumerate(high_list) if v == max_val)
    left_peaks = find_monotonic_peaks(high_list[:max_idx + 1], 'left')
    right_peaks = [max_idx + 1 + i for i in find_monotonic_peaks(high_list[max_idx + 1:], 'right')]
    raw_peaks = left_peaks + right_peaks

    # 新增：过滤相邻高点（核心修改）
    peaks = []
    for i in range(len(raw_peaks)):
        if i == 0 or raw_peaks[i] != raw_peaks[i - 1] + 1:  # 非相邻点
            peaks.append(raw_peaks[i])
        elif high_list[raw_peaks[i]] > high_list[raw_peaks[i - 1]]:  # 相邻时保留更高的
            peaks[-1] = raw_peaks[i]  # 替换前一个点

    # 第二阶段：检测波谷（保持原逻辑）
    valleys = []
    for i in range(len(peaks) - 1):
        start, end = peaks[i], peaks[i + 1]
        if end - start > 1:
            between = low_list[start + 1:end]
            min_idx = start + 1 + between.index(min(between))
            valleys.append(min_idx)

    # 第三阶段：补边界低点（保持原逻辑）
    if peaks:
        if peaks[0] > 0:
            valleys.append(low_list.index(min(low_list[:peaks[0]])))
        if peaks[-1] < len(high_list) - 1:
            right_min = min(low_list[peaks[-1] + 1:])
            valleys.append(peaks[-1] + 1 + low_list[peaks[-1] + 1:].index(right_min))

    return sorted(peaks), sorted(valleys)


def standardize_peaks_valleys(peaks, valleys, high_list, low_list):
    """
    标准化波峰波谷序列，确保严格交替出现
    参数:
        peaks: 原始波峰索引列表
        valleys: 原始波谷索引列表
        high_list: 高价序列（用于比较高度）
        low_list: 低价序列（用于比较低点）
    返回:
        (standard_peaks, standard_valleys)
    """
    # 合并并排序所有关键点
    all_points = sorted(set(peaks + valleys))
    if not all_points:
        return [], []

    # 初始化标准化结果
    standard_peaks = []
    standard_valleys = []

    # 确定第一个点的类型（波峰或波谷）
    if all_points[0] in peaks:
        current_type = 'peak'
        standard_peaks.append(all_points[0])
    else:
        current_type = 'valley'
        standard_valleys.append(all_points[0])

    # 遍历所有关键点，确保交替出现
    for point in all_points[1:]:
        if current_type == 'peak':
            # 当前需要找波谷（取最低点）
            candidates = [p for p in valleys if p > standard_peaks[-1]]
            if candidates:
                next_valley = min(candidates, key=lambda x: low_list[x])
                standard_valleys.append(next_valley)
                current_type = 'valley'
        else:
            # 当前需要找波峰（取最高点）
            candidates = [p for p in peaks if p > standard_valleys[-1]]
            if candidates:
                next_peak = max(candidates, key=lambda x: high_list[x])
                standard_peaks.append(next_peak)
                current_type = 'peak'

    return standard_peaks, standard_valleys


def standardize_peaks_valleys_v2(peaks, valleys, high_list, low_list):
    """
    标准化波峰波谷序列，确保严格交替出现
    参数:
        peaks: 原始波峰索引列表
        valleys: 原始波谷索引列表
        high_list: 高价序列（用于比较高度）
        low_list: 低价序列（用于比较低点）
    返回:
        (standard_peaks, standard_valleys)
    """
    # 确保交替出现
    extrema = sorted([(p, 'peak') for p in peaks] + [(v, 'valley') for v in valleys],
                     key=lambda x: x[0])

    final_peaks = []
    final_valleys = []

    if not extrema:
        return final_peaks, final_valleys

    # 使用栈的方式来处理，确保交替出现
    result = []

    for idx, typ in extrema:
        if not result:
            # 第一个点直接添加
            result.append((idx, typ))
            if typ == 'peak':
                final_peaks.append(idx)
            else:
                final_valleys.append(idx)
        else:
            last_idx, last_type = result[-1]

            if typ != last_type:
                # 类型不同，可以添加
                result.append((idx, typ))
                if typ == 'peak':
                    final_peaks.append(idx)
                else:
                    final_valleys.append(idx)
            else:
                # 类型相同，需要合并（保留更显著的）
                if typ == 'peak':
                    # 保留更高的波峰
                    if high_list[idx] > high_list[last_idx]:
                        # 替换最后一个波峰
                        result[-1] = (idx, typ)
                        final_peaks[-1] = idx
                else:
                    # 保留更低的波谷
                    if low_list[idx] < low_list[last_idx]:
                        # 替换最后一个波谷
                        result[-1] = (idx, typ)
                        final_valleys[-1] = idx

    return final_peaks, final_valleys


def build_wave_segments(high_list, peaks, valleys):
    """
    根据波峰波谷构建波浪段
    返回: [(start_idx, end_idx, is_rising)]
    """
    # 合并所有关键点并排序
    all_points = sorted(set([0] + peaks + valleys + [len(high_list) - 1]))

    segments = []
    for i in range(len(all_points) - 1):
        start = all_points[i]
        end = all_points[i + 1]
        is_rising = high_list[end] > high_list[start]
        segments.append((start, end, is_rising))

    return segments


# --------------------------
# 视图层（结果格式化）
# --------------------------

def format_wave_results(high_list, peaks, valleys, segments):
    """
    格式化检测结果为可视图层使用的结构
    返回: (path_points, peak_points, valley_points)
    """
    path_indices = sorted(set(
        [0] + peaks + valleys + [len(high_list) - 1]
    ))

    return (
        [(i, high_list[i]) for i in path_indices],  # path_points
        [(i, high_list[i]) for i in peaks],  # peak_points
        [(i, high_list[i]) for i in valleys]  # valley_points
    )


# --------------------------
# 接口函数
# --------------------------

def detect_main_wave_in_range(high_list):
    """保持原有接口的兼容函数"""
    peaks, valleys = detect_peaks_and_valleys(high_list)
    segments = build_wave_segments(high_list, peaks, valleys)
    path_points, peak_points, valley_points = format_wave_results(
        high_list, peaks, valleys, segments
    )
    return segments, path_points, peak_points, valley_points
