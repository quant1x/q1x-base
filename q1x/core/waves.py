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
from dataclasses import dataclass
from datetime import datetime, date
from operator import lt, le, gt, ge
from typing import List, Tuple, Literal, Dict, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates


def filter_sequence(arr, op: str = '<='):
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


def filter_sequence_with_indices(arr, op: str = '<='):
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


def build_wave_segments(high_list: list, low_list: list, peaks: list, valleys: list):
    """
    根据波峰（high）和波谷（low）构建波浪段，严格按高低点判断趋势。

    规则：
    - 段起点和终点必须是关键点（0, peaks, valleys, -1）
    - 趋势判断基于：从“谷”到“峰”为上升，从“峰”到“谷”为下降
    - 不依赖平均价、收盘价等模糊逻辑

    Args:
        high_list: 高价序列
        low_list:  低价序列
        peaks:     波峰索引（基于 high_list）
        valleys:   波谷索引（基于 low_list）

    Returns:
        List[Tuple[start, end, is_rising]]
    """
    if not high_list or not low_list:
        return []

    n = len(high_list)
    # 合并所有关键点
    key_points = sorted(set([0] + peaks + valleys + [n - 1]))

    segments = []
    for i in range(len(key_points) - 1):
        start_idx = key_points[i]
        end_idx = key_points[i + 1]

        if start_idx >= end_idx:
            continue

        # 判断起点和终点的性质
        is_start_peak = start_idx in peaks
        is_start_valley = start_idx in valleys
        is_end_peak = end_idx in peaks
        is_end_valley = end_idx in valleys

        # 严格按波浪结构判断趋势
        if is_start_valley and is_end_peak:
            # 从波谷到波峰 → 上升段
            is_rising = True
        elif is_start_peak and is_end_valley:
            # 从波峰到波谷 → 下降段
            is_rising = False
        else:
            # 其他情况（如 0→peak, valley→end, 0→valley 等）
            # 使用明确的价格逻辑：
            # - 若终点是峰，且 high 更高 → 上升
            # - 若终点是谷，且 low 更低 → 下降
            # - 否则保持前一段趋势？或保守判断

            # 但我们坚持：只看结构，不猜趋势
            # 所以这里可以抛出警告，或按以下保守逻辑：

            start_price = low_list[start_idx] if is_start_valley else high_list[start_idx]
            end_price = high_list[end_idx] if is_end_peak else low_list[end_idx]

            # 如果起点是峰或终点是谷，优先用 high；否则用 low
            # 更简单：直接比较 high 和 low 的极端变化
            if is_end_peak:
                is_rising = high_list[end_idx] > high_list[start_idx]
            elif is_end_valley:
                is_rising = low_list[end_idx] < low_list[start_idx]
            else:
                # 两端都不是极值点（如 0→普通点），用 high 判断
                is_rising = high_list[end_idx] > high_list[start_idx]

        segments.append((start_idx, end_idx, is_rising))

    return segments


def detect_wave_recursive(high_list, low_list, start_idx, end_idx, level=1):
    """
    递归检测指定区间的次级波浪
    """
    if end_idx - start_idx < 3:
        return []

    # 提取子区间
    high_sub = high_list[start_idx:end_idx + 1]
    low_sub = low_list[start_idx:end_idx + 1]

    # 检测子区间波峰波谷
    peaks_sub, valleys_sub = detect_peaks_and_valleys(high_sub, low_sub)

    # print(f"\tL{level} 区间[{start_idx}:{end_idx}] 波峰: {[start_idx + p for p in peaks_sub]}, 波谷: {[start_idx + v for v in valleys_sub]}")

    # ✅ 修复：传入 high_sub, low_sub, peaks_sub, valleys_sub 四个参数
    segments = build_wave_segments(high_sub, low_sub, peaks_sub, valleys_sub)

    global_segments = []
    for local_start, local_end, is_rising in segments:
        global_start = start_idx + local_start
        global_end = start_idx + local_end
        if global_start != global_end:
            global_segments.append((global_start, global_end, level, is_rising))

    # 继续递归（可选）
    for seg_start, seg_end, _ in segments:
        seg_global_start = start_idx + seg_start
        seg_global_end = start_idx + seg_end
        if seg_global_end - seg_global_start >= 3 and level < 2:
            sub_sub = detect_wave_recursive(high_list, low_list, seg_global_start, seg_global_end, level + 1)
            global_segments.extend(sub_sub)

    return global_segments


def detect_complete_wave_structure(high_list, low_list):
    """
    检测完整波浪结构（主波 + 递归次级波）
    支持高低序列输入
    """
    # 转为列表
    high_list = high_list.tolist() if isinstance(high_list, np.ndarray) else list(high_list)
    low_list = low_list.tolist() if isinstance(low_list, np.ndarray) else list(low_list)

    n = len(high_list)
    if n < 3 or len(low_list) != n:
        return []

    # 第一阶段：检测主波浪（波峰 from high, 波谷 from low）
    peaks, valleys = detect_peaks_and_valleys(high_list, low_list)

    # print(f"主波峰索引: {peaks}")
    # print(f"主波谷索引: {valleys}")

    # 构建主波段
    main_segments = build_wave_segments(high_list, low_list, peaks, valleys)

    # 转换为主波段（level 0）
    all_segments = [
        (start, end, 0, is_rising)
        for start, end, is_rising in main_segments
    ]

    # 第二阶段：递归检测次级波浪
    for start, end, _ in main_segments:
        if end - start >= 3:
            sub_waves = detect_wave_recursive(high_list, low_list, start, end, level=1)
            all_segments.extend(sub_waves)

    # 按层级和起始索引排序
    return sorted(all_segments, key=lambda x: (x[2], x[0]))


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


# def build_wave_segments(high_list, peaks, valleys):
#     """
#     根据波峰波谷构建波浪段
#     返回: [(start_idx, end_idx, is_rising)]
#     """
#     # 合并所有关键点并排序
#     all_points = sorted(set([0] + peaks + valleys + [len(high_list) - 1]))
#
#     segments = []
#     for i in range(len(all_points) - 1):
#         start = all_points[i]
#         end = all_points[i + 1]
#         is_rising = high_list[end] > high_list[start]
#         segments.append((start, end, is_rising))
#
#     return segments


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


def detect_main_wave_in_range(high_list, low_list):
    """保持原有接口的兼容函数"""
    peaks, valleys = detect_peaks_and_valleys(high_list, low_list)
    segments = build_wave_segments(high_list, peaks, valleys)
    path_points, peak_points, valley_points = format_wave_results(
        high_list, peaks, valleys, segments
    )
    return segments, path_points, peak_points, valley_points


@dataclass
class WaveSegment:
    start: int
    end: int
    level: int
    is_rising: bool

    def duration(self) -> int:
        return self.end - self.start

    def __repr__(self):
        trend = "↑" if self.is_rising else "↓"
        return f"[{self.start}→{self.end}]{trend}(L{self.level})"


TrendType = Literal["up", "down", "sideways", "uncertain"]


def determine_current_trend(
        segments: list,
        high_list: list,
        low_list: list,
        lookback: int = 5
) -> dict:
    """
    基于波段结构判断当前趋势
    segments: List[Tuple[start, end, level, is_rising]]
    """
    if not segments:
        return {"trend": "uncertain", "confidence": 0.0, "reason": "no segments"}

    # 按 start 排序（元组第0个元素）
    sorted_segs = sorted(segments, key=lambda x: x[0])  # x[0] = start

    # 取最近若干波段
    recent = sorted_segs[-lookback:]

    # ----------------------------
    # 1. 最近波段方向
    # ----------------------------
    last_seg = recent[-1]
    primary_trend = "up" if last_seg[3] else "down"  # x[3] = is_rising

    # ----------------------------
    # 2. 波峰波谷演化
    # ----------------------------
    rising_segs = [s for s in recent if s[3]]  # is_rising
    falling_segs = [s for s in recent if not s[3]]

    # 提取波峰（上升段的 end）
    peaks = []
    for s in rising_segs:
        end_idx = s[1]
        if end_idx < len(high_list):
            peaks.append((end_idx, high_list[end_idx]))
    peaks.sort(key=lambda x: x[0])  # 按索引排序

    # 提取波谷（下降段的 end）
    valleys = []
    for s in falling_segs:
        end_idx = s[1]
        if end_idx < len(low_list):
            valleys.append((end_idx, low_list[end_idx]))
    valleys.sort(key=lambda x: x[0])

    hh = hl = lh = ll = False
    if len(peaks) >= 2:
        hh = peaks[-1][1] > peaks[-2][1]
        lh = peaks[-1][1] < peaks[-2][1]
    if len(valleys) >= 2:
        hl = valleys[-1][1] > valleys[-2][1]
        ll = valleys[-1][1] < valleys[-2][1]

    structural_trend = "uncertain"
    if hh and hl:
        structural_trend = "up"
    elif lh and ll:
        structural_trend = "down"
    elif hh and ll:
        structural_trend = "sideways"
    elif lh and hl:
        structural_trend = "sideways"
    else:
        structural_trend = primary_trend

    # ----------------------------
    # 3. 多层级动量支持
    # ----------------------------
    sub_level = [s for s in recent if s[2] > 0]  # level > 0
    if sub_level:
        sub_rising_ratio = sum(1 for s in sub_level if s[3]) / len(sub_level)
        momentum_support = "strong" if (primary_trend == "up" and sub_rising_ratio > 0.6) or \
                                       (primary_trend == "down" and sub_rising_ratio < 0.4) \
            else "weak"
    else:
        momentum_support = "neutral"

    # ----------------------------
    # 4. 趋势强度评分
    # ----------------------------
    durations = [s[1] - s[0] for s in recent]  # end - start
    magnitudes = []
    for s in recent:
        try:
            if s[3]:  # is_rising
                mag = high_list[s[1]] - low_list[s[0]]
            else:
                mag = low_list[s[1]] - high_list[s[0]]
            magnitudes.append(abs(mag))
        except:
            pass

    avg_duration = sum(durations) / len(durations) if durations else 0
    avg_magnitude = sum(magnitudes) / len(magnitudes) if magnitudes else 0

    consecutive = 1
    for i in range(len(recent) - 1, 0, -1):
        if recent[i][3] == recent[i - 1][3]:
            consecutive += 1
        else:
            break

    confidence = 0.3
    if structural_trend == primary_trend:
        confidence += 0.4
    if momentum_support == "strong":
        confidence += 0.2
    if consecutive >= 3:
        confidence += 0.1

    final_trend = structural_trend if structural_trend in ("up", "down") else primary_trend

    return {
        "trend": final_trend,
        "confidence": round(confidence, 2),
        "primary_signal": primary_trend,
        "structural_signal": structural_trend,
        "momentum_support": momentum_support,
        "consecutive_segments": consecutive,
        "avg_duration": round(avg_duration, 1),
        "avg_magnitude": round(avg_magnitude, 4),
        "last_segment": last_seg[:4],
        "peaks": [p[0] for p in peaks[-3:]],
        "valleys": [v[0] for v in valleys[-3:]],
        "reason": f"Structural: {structural_trend}, Momentum: {momentum_support}, Consecutive: {consecutive}"
    }

def plot_trend_from_extremes(
        klines: pd.DataFrame,
        trend_result: dict,
        figsize=(16, 9)
):
    """
    可视化基于5个极值点的趋势分析，并绘制波峰/波谷趋势线在当前时刻的延伸
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    # 确保日期是 datetime 类型
    dates = pd.to_datetime(klines['date'])
    high_list = klines['high'].values
    low_list = klines['low'].values
    close_price = klines['close'].iloc[-1]

    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制高低价背景线
    ax.plot(dates, high_list, color='lightgray', alpha=0.5, linewidth=1, label='High')
    ax.plot(dates, low_list, color='lightgray', alpha=0.5, linewidth=1, label='Low')

    # 提取极值点信息
    indices = trend_result['indices']
    types = trend_result['price_structure']
    extreme_prices = [high_list[i] if t == 'peak' else low_list[i] for i, t in zip(indices, types)]
    extreme_dates = dates.iloc[indices]

    # 分离波峰和波谷
    peaks = [(d, p) for d, p, t in zip(extreme_dates, extreme_prices, types) if t == 'peak']
    valleys = [(d, p) for d, p, t in zip(extreme_dates, extreme_prices, types) if t == 'valley']

    # ----------------------------
    # 绘制连接线（谷→峰，峰→谷）
    # ----------------------------
    # 1. 波谷 → 波峰（红实线，上升段）
    for i in range(len(types) - 1):
        if types[i] == 'valley' and types[i + 1] == 'peak':
            ax.plot([extreme_dates.iloc[i], extreme_dates.iloc[i + 1]],
                    [extreme_prices[i], extreme_prices[i + 1]],
                    color='red', linewidth=2.5, alpha=0.8, solid_capstyle='round')

    # 2. 波峰 → 波谷（绿实线，下降段）
    for i in range(len(types) - 1):
        if types[i] == 'peak' and types[i + 1] == 'valley':
            ax.plot([extreme_dates.iloc[i], extreme_dates.iloc[i + 1]],
                    [extreme_prices[i], extreme_prices[i + 1]],
                    color='green', linewidth=2.5, alpha=0.8, solid_capstyle='round')

    # ----------------------------
    # ✅ 绘制波峰和波谷的趋势延长线（从第一个极值点延伸到最新K线）
    # ----------------------------
    last_date = dates.iloc[-1]  # 最后一个交易日
    upper_at_last = None
    lower_at_last = None

    # 波峰趋势线（红虚线）：峰 → 峰
    if len(peaks) >= 2:
        p_dates, p_prices = zip(*peaks)
        # 将日期转为数值（时间戳）
        p_timestamps = [d.timestamp() for d in p_dates]
        # 拟合一次线性回归
        z = np.polyfit(p_timestamps, p_prices, 1)
        poly_upper = np.poly1d(z)
        # 计算从第一个峰到最后一个K线的延长线
        extended_x = [mdates.date2num(p_dates[0]), mdates.date2num(last_date)]
        extended_y = [poly_upper(p_dates[0].timestamp()), poly_upper(last_date.timestamp())]
        # 使用 matplotlib 绘图（支持 datetime）
        ax.plot(extended_x, extended_y, color='red', linestyle='--', linewidth=2,
                alpha=0.8, label='阻力趋势线（峰→当前）')
        # ✅ 记录当前上轨值
        upper_at_last = poly_upper(last_date.timestamp())
        ax.scatter(last_date, upper_at_last, color='red', s=80, zorder=6, marker='x', linewidth=2)

    # 波谷趋势线（绿虚线）：谷 → 谷
    if len(valleys) >= 2:
        v_dates, v_prices = zip(*valleys)
        v_timestamps = [d.timestamp() for d in v_dates]
        z = np.polyfit(v_timestamps, v_prices, 1)
        poly_lower = np.poly1d(z)
        extended_x = [mdates.date2num(v_dates[0]), mdates.date2num(last_date)]
        extended_y = [poly_lower(v_dates[0].timestamp()), poly_lower(last_date.timestamp())]
        ax.plot(extended_x, extended_y, color='green', linestyle='--', linewidth=2,
                alpha=0.8, label='支撑趋势线（谷→当前）')
        # ✅ 记录当前下轨值
        lower_at_last = poly_lower(last_date.timestamp())
        ax.scatter(last_date, lower_at_last, color='green', s=80, zorder=6, marker='x', linewidth=2)

    # ----------------------------
    # ✅ 绘制当前时刻的垂直线 & 通道标注
    # ----------------------------
    ax.axvline(last_date, color='gray', linestyle='-', linewidth=1.5, alpha=0.6)
    ax.text(last_date, ax.get_ylim()[1], ' 当前', fontsize=10, color='gray',
            verticalalignment='bottom', horizontalalignment='left',
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="gray"))

    # ----------------------------
    # 绘制极值点并标注序号
    # ----------------------------
    # 波峰（红圈）
    if peaks:
        p_dates, p_prices = zip(*peaks)
        ax.scatter(p_dates, p_prices, color='red', s=120, zorder=5, edgecolors='black', linewidth=1.5, label='波峰')

    # 波谷（绿圈）
    if valleys:
        v_dates, v_prices = zip(*valleys)
        ax.scatter(v_dates, v_prices, color='green', s=120, zorder=5, edgecolors='black', linewidth=1.5, label='波谷')

    # 标注序号
    for i, (d, p, t) in enumerate(zip(extreme_dates, extreme_prices, types)):
        ax.annotate(f'{i + 1}', (d, p), xytext=(0, 10 if t == 'valley' else -15),
                    textcoords='offset points', fontsize=12, ha='center', weight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8, edgecolor="darkred"))

    # ----------------------------
    # 图表装饰
    # ----------------------------
    title = f"趋势分析：{trend_result['trend'].upper()} (置信度: {trend_result['confidence']:.2f})"
    ax.set_title(title, fontsize=16, pad=20, color='darkblue', weight='bold')

    reason = f"依据: {trend_result['reason']}"
    props = dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.85, edgecolor="brown")
    ax.text(0.02, 0.98, reason, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props, family='SimHei')

    ax.set_xlabel("日期", fontsize=12)
    ax.set_ylabel("价格", fontsize=12)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    # 日期格式
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 显示
    plt.show()

    # ✅ 安全打印当前通道信息（只有当 upper_at_last 和 lower_at_last 被赋值时才打印）
    if upper_at_last is not None and lower_at_last is not None:
        print(f"📈 当前通道状态（{last_date.strftime('%Y-%m-%d')}）:")
        print(f"   上轨（阻力）: {upper_at_last:.4f}")
        print(f"   下轨（支撑）: {lower_at_last:.4f}")
        print(f"   通道宽度: {upper_at_last - lower_at_last:.4f}")
        print(f"   当前收盘价: {close_price:.4f}")
        print(f"   价格位置: {'突破上轨' if close_price > upper_at_last else '跌破下轨' if close_price < lower_at_last else '通道内'}")
    elif upper_at_last is not None:
        print(f"📉 仅波峰趋势线有效，当前阻力: {upper_at_last:.4f}")
    elif lower_at_last is not None:
        print(f"📉 仅波谷趋势线有效，当前支撑: {lower_at_last:.4f}")
    else:
        print("⚠️ 无法绘制趋势线：波峰或波谷不足2个")

def determine_trend_from_last_5_extremes(
        segments: List[Tuple[int, int, int, bool]],
        high_list: List[float],
        low_list: List[float],
        klines: pd.DataFrame
) -> Dict:
    # ----------------------------
    # Phase 0: 输入校验与预处理
    # ----------------------------
    if len(segments) < 2 or len(high_list) == 0 or len(low_list) == 0:
        return {"trend": "uncertain", "confidence": 0.0, "reason": "输入数据不足"}

    if 'date' not in klines.columns:
        return {"trend": "uncertain", "confidence": 0.0, "reason": "klines 缺少 'date' 列"}

    # 确保 date 是 pd.Timestamp 类型
    if not isinstance(klines['date'].iloc[0], pd.Timestamp):
        klines['date'] = pd.to_datetime(klines['date'])

    # ----------------------------
    # Phase 1: 提取极值点（使用 pd.Timestamp）
    # ----------------------------
    points = []  # (index, type, price, date: pd.Timestamp)

    prev_end = -1
    for seg in sorted(segments, key=lambda x: x[0]):
        start, end, level, is_rising = seg
        if seg[2] != 0:  # 只处理 level == 0 的段？
            continue
        if end <= prev_end or end >= len(high_list) or end >= len(low_list):
            continue
        prev_end = end

        bar_date = klines.iloc[end]['date']
        if pd.isna(bar_date):
            continue

        if is_rising:
            price = high_list[end]
            points.append((end, "peak", price, bar_date))
        else:
            price = low_list[end]
            points.append((end, "valley", price, bar_date))

    if len(points) < 3:
        return {"trend": "uncertain", "confidence": 0.0, "reason": f"有效极值点不足3个（{len(points)}个）"}

    recent = points[-5:]  # 最近5个极值点

    # ----------------------------
    # Phase 2: 基于结构模式赋初值（Define）
    # ----------------------------
    trend_scores = {"up": 0.0, "down": 0.0, "sideways": 0.0, "reversal": 0.0}
    reasons = []

    prices = [p[2] for p in recent]
    types = [p[1] for p in recent]

    peak_prices = [p[2] for p in recent if p[1] == "peak"]
    valley_prices = [p[2] for p in recent if p[1] == "valley"]

    def is_increasing(seq, threshold=0.02):
        return len(seq) >= 2 and all(seq[i+1] > seq[i] * (1 + threshold) for i in range(len(seq)-1))

    def is_decreasing(seq, threshold=0.02):
        return len(seq) >= 2 and all(seq[i+1] < seq[i] * (1 - threshold) for i in range(len(seq)-1))

    if len(peak_prices) >= 2 and len(valley_prices) >= 2:
        if is_increasing(peak_prices) and is_increasing(valley_prices):
            trend_scores["up"] += 0.8
            reasons.append("HH + HL")
        elif is_decreasing(peak_prices) and is_decreasing(valley_prices):
            trend_scores["down"] += 0.8
            reasons.append("LH + LL")
        elif is_increasing(peak_prices) and is_decreasing(valley_prices):
            trend_scores["sideways"] += 0.6
            reasons.append("HH + LL (扩散震荡)")
        elif is_decreasing(peak_prices) and is_increasing(valley_prices):
            trend_scores["reversal"] += 0.6
            reasons.append("LH + HL (收敛，潜在反转)")

    # ----------------------------
    # Phase 3: 基于通道结构修正（Refine）
    # ----------------------------
    peaks = [(p[3], p[2]) for p in recent if p[1] == "peak"]  # [(date, price)]
    valleys = [(v[3], v[2]) for v in recent if v[1] == "valley"]

    if len(peaks) >= 2 and len(valleys) >= 2:
        # 提取前两个波峰和波谷（最早两个）
        (p1_date, p1_price), (p2_date, p2_price) = peaks[0], peaks[1]
        (v1_date, v1_price), (v2_date, v2_price) = valleys[0], valleys[1]

        # 统一时间基准（以最早日期为0）
        base_date = min(p1_date, p2_date, v1_date, v2_date, klines['date'].iloc[-1])

        def date_to_days(date):
            return (date - base_date).total_seconds() / (24 * 3600)  # 转为天数（float）

        peak_days = [date_to_days(p1_date), date_to_days(p2_date)]
        valley_days = [date_to_days(v1_date), date_to_days(v2_date)]
        last_date_num = date_to_days(klines['date'].iloc[-1])  # 当前K线时间
        last_price = float(klines['close'].iloc[-1])

        # 拟合直线：y = kx + b
        def fit_line(x_vals, y_vals):
            x1, x2 = x_vals
            y1, y2 = y_vals
            if abs(x2 - x1) < 1e-8:
                k = 0.0
            else:
                k = (y2 - y1) / (x2 - x1)
            b = y1 - k * x1
            return k, b

        try:
            k_upper, b_upper = fit_line(peak_days, [p1_price, p2_price])
            k_lower, b_lower = fit_line(valley_days, [v1_price, v2_price])
        except Exception as e:
            reasons.append("通道拟合失败")
            # 跳过通道分析
        else:
            # 计算当前时刻（最新K线时间）对应的上下轨值
            current_upper = k_upper * last_date_num + b_upper
            current_lower = k_lower * last_date_num + b_lower

            if current_upper < current_lower:
                # 防止上下轨颠倒
                current_upper, current_lower = current_lower, current_upper

            # 判断通道形态（基于斜率）
            slope_diff = k_upper - k_lower  # 上轨斜率 - 下轨斜率

            if k_upper < 0 and k_lower > 0:
                channel_status = "converging"  # 上轨↓ 下轨↑ → 收敛
            elif k_upper > 0 and k_lower < 0:
                channel_status = "diverging"   # 上轨↑ 下轨↓ → 扩散
            elif abs(slope_diff) < 1e-5:
                channel_status = "parallel"
            elif slope_diff < 0:
                channel_status = "converging"
            else:
                channel_status = "diverging"

            # 更新评分
            if channel_status == "converging":
                reasons.append("通道收敛")
                trend_scores["reversal"] += 0.2
            elif channel_status == "diverging":
                reasons.append("通道扩散")
                if trend_scores["up"] > trend_scores["down"]:
                    trend_scores["up"] += 0.1
                elif trend_scores["down"] > trend_scores["up"]:
                    trend_scores["down"] += 0.1
            else:  # parallel
                reasons.append("通道平行")
                if trend_scores["up"] > trend_scores["down"]:
                    trend_scores["up"] += 0.1
                elif trend_scores["down"] > trend_scores["up"]:
                    trend_scores["down"] += 0.1

            # 检查价格与通道关系
            if last_price > current_upper:
                reasons.append("价格突破上轨")
            elif last_price < current_lower:
                reasons.append("价格跌破下轨")
            else:
                reasons.append("价格位于通道内")

    # ----------------------------
    # Phase 4: 归一化与决策
    # ----------------------------
    total = sum(trend_scores.values())
    if total > 1e-5:
        for k in trend_scores:
            trend_scores[k] /= total
    else:
        trend_scores = {k: round(1 / len(trend_scores), 2) for k in trend_scores}

    main_trend = max(trend_scores, key=trend_scores.get)
    confidence = trend_scores[main_trend]

    return {
        "trend": main_trend,
        "confidence": round(confidence, 2),
        "reason": "; ".join(reasons),
        "extreme_points": [
            (idx, typ, round(pri, 4), date.strftime('%Y-%m-%d'))
            for idx, typ, pri, date in recent
        ],
        "peak_prices": [round(p, 4) for p in peak_prices],
        "valley_prices": [round(p, 4) for p in valley_prices],
        "price_structure": types,
        "indices": [p[0] for p in recent],
        "trend_scores": {k: round(v, 3) for k, v in trend_scores.items()}
    }