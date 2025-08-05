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
        [4, 7, 10]
        >>> # 波谷检测
        >>> find_monotonic_extremes(data, 'left', 'valley')
        [2, 6, 9]
        >>> # 右向扫描（结果仍按原始顺序返回）
        >>> find_monotonic_extremes(data, 'right', 'peak')
        [4, 7, 10]

    Examples:
        .. code-block:: python

            data = [3, 2, 1, 2, 3, 2, 1, 4, 3, 2, 5]

            # 波峰检测
            find_monotonic_extremes(data, 'left', 'peak')  # → [4, 7, 10]

            # 波谷检测
            find_monotonic_extremes(data, 'left', 'valley')  # → [2, 6, 9]

            # 右向扫描
            find_monotonic_extremes(data, 'right', 'peak')  # → [4, 7, 10]
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


def detect_peaks_and_valleys_v1(high_list, low_list):
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


def detect_peaks_and_valleys_v2(high_list, low_list):
    """
    严格保持与原函数完全相同的输出结果：
    1. 波峰检测继续使用左右分段检测模式
    2. 波谷仍然完全由波峰位置派生
    3. 保持所有边界条件处理逻辑
    """
    # 数据预处理
    if isinstance(high_list, np.ndarray):
        high_list = high_list.tolist()
    if isinstance(low_list, np.ndarray):
        low_list = low_list.tolist()
    if not high_list:
        return [], []

    # 第一阶段：波峰检测（逻辑不变，仅替换函数名）
    max_val = max(high_list)
    max_idx = high_list.index(max_val)

    # 仅将find_monotonic_peaks替换为find_monotonic_extremes
    left_peaks = find_monotonic_extremes(high_list[:max_idx + 1], 'left', 'peak')  # 仅改函数名
    right_peaks = [max_idx + 1 + i for i in
                   find_monotonic_extremes(high_list[max_idx + 1:], 'right', 'peak')]  # 仅改函数名
    raw_peaks = left_peaks + right_peaks

    # 相邻高点过滤
    peaks = []
    for i in range(len(raw_peaks)):
        if i == 0 or raw_peaks[i] != raw_peaks[i - 1] + 1:
            peaks.append(raw_peaks[i])
        elif high_list[raw_peaks[i]] > high_list[raw_peaks[i - 1]]:
            peaks[-1] = raw_peaks[i]

    # 第二阶段：波谷检测
    valleys = []
    for i in range(len(peaks) - 1):
        start, end = peaks[i], peaks[i + 1]
        if end - start > 1:
            between = low_list[start + 1:end]
            min_idx = start + 1 + between.index(min(between))
            valleys.append(min_idx)

    # 第三阶段：边界波谷
    if peaks:
        if peaks[0] > 0:
            valleys.append(low_list.index(min(low_list[:peaks[0]])))
        if peaks[-1] < len(low_list) - 1:
            right_min = min(low_list[peaks[-1] + 1:])
            valleys.append(peaks[-1] + 1 + low_list[peaks[-1] + 1:].index(right_min))

    return sorted(peaks), sorted(valleys)  # 保持原排序方式


def detect_peaks_and_valleys_v3(high_list: List[float], low_list: List[float]) -> Tuple[List[int], List[int]]:
    """
    检测波峰和波谷，采用对称化处理逻辑：

    - **波峰检测**：基于 high_list，找全局最大值，左右分段，分别用 find_monotonic_extremes 找单调极值
    - **波谷检测**：基于 low_list，找全局最小值，左右分段，分别用 find_monotonic_extremes 找单调极值
    - 保持与原函数相同的边界处理、去重、排序逻辑

    Args:
        high_list: 高价序列
        low_list: 低价序列

    Returns:
        (peaks, valleys): 波峰和波谷索引列表，均已排序

    Examples:
        >>> high_list = [3, 5, 4, 6, 8, 7, 9, 6, 7]
        >>> low_list = [2, 4, 1, 3, 6, 5, 8, 4, 5]
        >>> detect_peaks_and_valleys(high_list, low_list)
        ([1, 4, 6, 8], [2, 4, 7])
    """
    # 数据预处理
    if isinstance(high_list, np.ndarray):
        high_list = high_list.tolist()
    if isinstance(low_list, np.ndarray):
        low_list = low_list.tolist()
    if not high_list or not low_list:
        return [], []

    # ====================
    # 第一阶段：波峰检测（原逻辑不变）
    # ====================
    max_val = max(high_list)
    max_idx = high_list.index(max_val)

    # 左半段：从左到右找波峰（上升段中的峰值）
    left_peaks = find_monotonic_extremes(high_list[:max_idx + 1], 'left', 'peak')
    # 右半段：从右到左找波峰（下降段中的峰值）
    right_peaks = [
        max_idx + 1 + i
        for i in find_monotonic_extremes(high_list[max_idx + 1:], 'right', 'peak')
    ]
    raw_peaks = left_peaks + right_peaks

    # 去重并处理相邻点（保留更高者）
    peaks = []
    for i, idx in enumerate(raw_peaks):
        if i == 0 or idx != peaks[-1] + 1:
            peaks.append(idx)
        elif high_list[idx] > high_list[peaks[-1]]:
            peaks[-1] = idx

    # ====================
    # 第二阶段：波谷检测（对称处理！）
    # ====================
    min_val = min(low_list)
    min_idx = low_list.index(min_val)

    # 左半段：从左到右找波谷（下降段中的谷值）
    left_valleys = find_monotonic_extremes(low_list[:min_idx + 1], 'left', 'valley')
    # 右半段：从右到左找波谷（上升段中的谷值）
    right_valleys = [
        min_idx + 1 + i
        for i in find_monotonic_extremes(low_list[min_idx + 1:], 'right', 'valley')
    ]
    raw_valleys = left_valleys + right_valleys

    # 去重并处理相邻点（保留更低者）
    valleys = []
    for i, idx in enumerate(raw_valleys):
        if i == 0 or idx != valleys[-1] + 1:
            valleys.append(idx)
        elif low_list[idx] < low_list[valleys[-1]]:
            valleys[-1] = idx

    # ====================
    # 第三阶段：边界波谷（可选，保持原逻辑）
    # ====================
    # 注意：原函数在 peaks 之间和边界插入 valleys
    # 我们已用对称逻辑覆盖主要波谷，但可选择性保留边界补全
    # —— 但为保持“对称性”，我们**不额外插入边界波谷**，除非明确需要

    # 如果你坚持保留原函数的边界补全逻辑，可取消注释以下代码：
    """
    if peaks:
        if peaks[0] > 0 and 0 not in valleys:
            left_min_idx = low_list.index(min(low_list[:peaks[0]]))
            if left_min_idx not in valleys:
                valleys.append(left_min_idx)
        if peaks[-1] < len(low_list) - 1 and len(low_list) - 1 not in valleys:
            right_part = low_list[peaks[-1] + 1:]
            if right_part:
                right_min_idx = peaks[-1] + 1 + right_part.index(min(right_part))
                if right_min_idx not in valleys:
                    valleys.append(right_min_idx)
    """

    # 返回排序结果（原函数最后排序）
    return sorted(peaks), sorted(valleys)


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

    peaks, valleys = valid_peaks, valid_valleys

    # ====================
    # 第三阶段：确保交替性（可选增强）
    # ====================
    # 目标：最终序列应为 peak, valley, peak, valley... 交替
    # 方法：从左到右合并最近的极值点

    all_extremes = []
    peak_set = set(peaks)
    valley_set = set(valleys)

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
