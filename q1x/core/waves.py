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

from operator import lt, le, gt, ge
from typing import List, Tuple

import numpy as np


def filter_sequence(arr, op:str='<='):
    """
    过滤数组，保留值升序且索引连续的元素

    Parameters:
        arr (list): 输入数组
        op (str): 比较符号

    Returns:
        list: 过滤后的数组
    """
    if not arr:
        return []

    filtered = [arr[0]]
    last_value = arr[0]

    cmp = {
        '<': lt,
        '<=': le,
        '>': gt,
        '>=': ge,
    }
    compare = cmp.get(op, lt)

    for num in arr[1:]:
        if compare(last_value, num):
            filtered.append(num)
            last_value = num

    return filtered


def filter_sequence_with_indices(arr, op:str='<='):
    """
    过滤数组，返回符合升序条件的元素及其原索引

    Parameters:
        arr (list): 输入数组
        op (str): 比较符号

    Returns:
        tuple: (values, indices)
    """
    if not arr:
        return [], []

    cmp = {
        '<': lt,
        '<=': le,
        '>': gt,
        '>=': ge,
    }
    compare = cmp.get(op, lt)

    values = [arr[0]]
    indices = [0]
    last_value = arr[0]

    for i in range(1, len(arr)):
        num = arr[i]
        if compare(last_value, num):
            values.append(num)
            indices.append(i)
            last_value = num

    return values, indices


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


def find_monotonic_extremes(data_list, direction='left', mode='peak') -> list[int]:
    """单调序列极值检测（支持波峰和波谷检测）。

    从指定方向遍历数据，找到单调变化中的极值点（最大值或最小值）。
    支持左向（正向）和右向（反向）扫描，并自动按原始顺序返回索引。

    Args:
        data_list: 输入数据列表，应为数值型列表。
        direction: 扫描方向，可选 'left'（从左到右）或 'right'（从右到左）。默认 'left'。
        mode: 检测模式，可选 'peak'（波峰，找最大值）或 'valley'（波谷，找最小值）。默认 'peak'。

    Returns:
        极值点的索引列表，按原始数据顺序排列。

    Examples:
        >>> data = [3, 2, 1, 2, 3, 2, 1, 4, 3, 2, 5]
        >>> # 波峰检测
        >>> find_monotonic_extremes(data, 'left', 'peak')
        [0, 7, 10]
        >>> # 波谷检测
        >>> find_monotonic_extremes(data, 'left', 'valley')
        [2]
        >>> # 右向扫描（结果仍按原始顺序返回）
        >>> find_monotonic_extremes(data, 'right', 'peak')
        [10]
    """
    if not data_list:
        return []
    if len(data_list) == 1:
        return [0]  # 单点就是极值点

    extremes = []
    start_idx = 0 if direction == 'left' else len(data_list) - 1
    end_idx = len(data_list) if direction == 'left' else -1
    step = 1 if direction == 'left' else -1

    prev_idx = start_idx
    prev_val = data_list[start_idx]

    # 根据模式选择比较运算符
    compare = (lambda a, b: a > b) if mode == 'peak' else (lambda a, b: a < b)

    for current_idx in range(start_idx + step, end_idx, step):
        current_val = data_list[current_idx]

        if compare(current_val, prev_val):
            prev_idx, prev_val = current_idx, current_val
        elif extremes and prev_val == data_list[extremes[-1]]:
            continue
        else:
            extremes.append(prev_idx)

    # 处理最后一个元素
    if not extremes or compare(prev_val, data_list[extremes[-1]]):
        extremes.append(prev_idx)

    return extremes if direction == 'left' else extremes[::-1]


def find_monotonic_peaks_around_max(lst: List[float]) -> List[int]:
    max_val = max(lst)
    max_idx = lst.index(max_val)
    left = find_monotonic_extremes(lst[:max_idx + 1], 'left', 'peak')
    right = [max_idx + 1 + i for i in find_monotonic_extremes(lst[max_idx + 1:], 'right', 'peak')]
    raw = left + right
    # 去重：相邻索引保留更高者
    peaks = []
    for idx in raw:
        if not peaks or idx != peaks[-1] + 1:
            peaks.append(idx)
        elif lst[idx] > lst[peaks[-1]]:
            peaks[-1] = idx
    return peaks


def find_monotonic_valleys_around_min(lst: List[float]) -> List[int]:
    min_val = min(lst)
    min_idx = lst.index(min_val)
    left = find_monotonic_extremes(lst[:min_idx + 1], 'left', 'valley')
    right = [min_idx + 1 + i for i in find_monotonic_extremes(lst[min_idx + 1:], 'right', 'valley')]
    raw = left + right
    # 去重：相邻索引保留更低者
    valleys = []
    for idx in raw:
        if not valleys or idx != valleys[-1] + 1:
            valleys.append(idx)
        elif lst[idx] < lst[valleys[-1]]:
            valleys[-1] = idx
    return valleys


def refine_peaks_by_valleys(peaks, valleys, high_list):
    """
    根据波谷的存在性，优化波峰序列：若两峰之间无谷，则保留更高者。
    """
    # 1. 验证波峰之间是否有波谷 → 若无，则合并/剔除
    valid_peaks = []
    for i in range(len(peaks)):
        if i == 0:
            valid_peaks.append(peaks[i])
            continue

        prev_peak = valid_peaks[-1]
        curr_peak = peaks[i]

        # 检查 [prev_peak+1, curr_peak] 区间内是否有波谷
        has_valley_between = any(prev_peak < v < curr_peak for v in valleys)

        if has_valley_between:
            valid_peaks.append(curr_peak)
        else:
            # 无波谷 → 两个波峰“连续”，保留更高的
            if high_list[curr_peak] > high_list[prev_peak]:
                valid_peaks[-1] = curr_peak  # 替换为更高者
            # 否则保留原 peak（较小的被剔除）
    return valid_peaks


def refine_valleys_by_peaks(valleys, peaks, low_list):
    """
    根据波峰的存在性，优化波谷序列：若两谷之间无峰，则保留更低者。
    """
    # 2. 验证波谷之间是否有波峰 → 若无，则合并/剔除
    valid_valleys = []
    for i in range(len(valleys)):
        if i == 0:
            valid_valleys.append(valleys[i])
            continue

        prev_valley = valid_valleys[-1]
        curr_valley = valleys[i]

        # 检查 [prev_valley+1, curr_valley] 区间内是否有波峰
        has_peak_between = any(prev_valley < p < curr_valley for p in peaks)

        if has_peak_between:
            valid_valleys.append(curr_valley)
        else:
            # 无波峰 → 两个波谷“连续”，保留更低的
            if low_list[curr_valley] < low_list[prev_valley]:
                valid_valleys[-1] = curr_valley
            # 否则保留原 valley
    return valid_valleys


def normalize_peaks_and_valleys(peaks, valleys, high_list, low_list):
    """
    将波峰和波谷序列合并并规范化，确保它们交替出现。
    若连续出现同类型极值（如两个峰之间无谷），则保留更极端者（更高峰或更低谷）。
    """
    all_extremes = []
    peak_set = set(peaks)
    valley_set = set(valleys)
    n = len(high_list)
    i = 0
    while i < n:
        if i in peak_set:
            all_extremes.append(('peak', i))
        elif i in valley_set:
            all_extremes.append(('valley', i))
        i += 1

    if not all_extremes:
        return [], []

    # 修剪：确保交替
    cleaned = [all_extremes[0]]
    for t, idx in all_extremes[1:]:
        last_type, _ = cleaned[-1]
        if t != last_type:  # 类型不同（peak → valley 或反之）
            cleaned.append((t, idx))
        else:
            # 类型相同，保留更极端者
            prev_idx = cleaned[-1][1]
            if t == 'peak' and high_list[idx] > high_list[prev_idx]:
                cleaned[-1] = (t, idx)
            elif t == 'valley' and low_list[idx] < low_list[prev_idx]:
                cleaned[-1] = (t, idx)

    # 重新提取
    final_peaks = [idx for t, idx in cleaned if t == 'peak']
    final_valleys = [idx for t, idx in cleaned if t == 'valley']
    return final_peaks, final_valleys


def detect_peaks_and_valleys(high_list: List[float], low_list: List[float]) -> Tuple[List[int], List[int]]:
    """
    检测并交叉验证波峰与波谷，确保交替性。

    流程：
    1. 分别检测 high_list 的波峰 和 low_list 的波谷（对称逻辑）
    2. 交叉验证：相邻波峰之间必须有波谷，否则剔除较小者
    3. 同理验证波谷之间必须有波峰
    4. 最终确保波峰与波谷交替出现

    Args:
        high_list: 高价序列（K线 high）
        low_list: 低价序列（K线 low）

    Returns:
        (peaks, valleys): 经交叉验证后的波峰和波谷索引列表

    Examples:
        >>> high_list = [3, 5, 4, 6, 8, 7, 9, 6, 7]
        >>> low_list = [2, 4, 1, 3, 6, 5, 8, 4, 5]
        >>> detect_peaks_and_valleys(high_list, low_list)
        ([1, 4, 6], [2, 7])  # 示例输出（实际依数据而定）
    """
    # 数据预处理
    if isinstance(high_list, np.ndarray):
        high_list = high_list.tolist()
    if isinstance(low_list, np.ndarray):
        low_list = low_list.tolist()
    n = len(high_list)
    if not n or len(low_list) != n:
        return [], []

    if len(high_list) == 1 and len(low_list) == 1:
        return [0], [0]

    # ====================
    # 第一阶段：独立检测波峰（high）和波谷（low）
    # ====================
    peaks = find_monotonic_peaks_around_max(high_list)
    valleys = find_monotonic_valleys_around_min(low_list)

    if not peaks or not valleys:
        return sorted(peaks), sorted(valleys)

    # ====================
    # 第二阶段：交叉验证与修剪
    # ====================
    valid_peaks = refine_peaks_by_valleys(peaks, valleys, high_list)
    valid_valleys = refine_valleys_by_peaks(valleys, peaks, low_list)

    peaks, valleys = valid_peaks, valid_valleys

    # ====================
    # 第三阶段：确保交替性（可选增强）
    # ====================
    # 目标：最终序列应为 peak, valley, peak, valley... 交替
    # 方法：从左到右合并最近的极值点
    final_peaks, final_valleys = normalize_peaks_and_valleys(peaks, valleys, high_list, low_list)

    return sorted(final_peaks), sorted(final_valleys)


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
