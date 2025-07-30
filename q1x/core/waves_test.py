#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project : q1x-base
@Package : core
@File    : waves_test.py
@Author  : wangfeng
@Date    : 2025/7/29 13:51
@Desc    : 波浪检测 - 可视化测试（支持高低序列、交叉验证、递归分段）
"""

import numpy as np
from matplotlib import pyplot as plt

# 导入你的核心函数（确保路径正确）
from q1x.core.waves import find_monotonic_extremes, detect_peaks_and_valleys


def build_wave_segments(
        high_list: list,
        low_list: list,
        peaks: list,
        valleys: list
):
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

    print(f"主波峰索引: {peaks}")
    print(f"主波谷索引: {valleys}")

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

    print(f"  L{level} 区间[{start_idx}:{end_idx}] 波峰: {[start_idx + p for p in peaks_sub]}, "
          f"波谷: {[start_idx + v for v in valleys_sub]}")

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

# 线宽和透明度按层级衰减
def get_wave_style(level):
    linewidth = max(1, 3 - level)
    alpha = max(0.4, 0.8 - 0.2 * level)
    return linewidth, alpha

from matplotlib.patches import FancyArrowPatch

def plot_wave_structure(title, high_list, wave_segments, peaks=None, valleys=None):
    """
    绘制波浪结构图，带方向箭头和不同线条样式。
    """
    plt.figure(figsize=(16, 9))
    data = high_list if isinstance(high_list, list) else high_list.tolist()
    x = np.arange(len(data))

    # 绘制原始数据
    plt.plot(x, data, 'k-', alpha=0.4, linewidth=2, label='High 序列')
    plt.scatter(x, data, c='black', s=25, alpha=0.7, zorder=5)

    # 颜色映射（主次分明，逐渐变淡）
    color_map = {
        (0, True): '#D62728',   # 主升：深红
        (0, False): '#2CA02C',  # 主降：深绿
        (1, True): '#F4A460',   # 次级升：浅橙
        (1, False): '#98D8D8',  # 次级降：淡青
        (2, True): '#D3D3D3',   # 三级升：浅灰
        (2, False): '#A9A9A9',  # 三级降：暗灰
    }

    # 绘制波浪段
    handles = []  # 存储图例句柄
    labels = []   # 存储图例标签

    for start, end, level, is_rising in wave_segments:
        color = color_map.get((level, is_rising), '#777777')
        linewidth, alpha = get_wave_style(level)

        # 区分上涨和下跌的线条样式
        linestyle = 'solid' if is_rising else 'dashed'

        # 绘制波浪段
        line, = plt.plot(
            [start, end], [data[start], data[end]],
            color=color, linewidth=linewidth, alpha=alpha,
            linestyle=linestyle,
            label=f'Level {level}'  # 为每个层级设置标签
        )

        # 添加到图例句柄和标签列表
        if f'Level {level}' not in labels:
            handles.append(line)
            labels.append(f'Level {level}')

        # 添加方向箭头（仅在主波和次级波的关键点）
        if level <= 1:  # 仅对 L0 和 L1 添加箭头
            arrow_length = 0.5  # 箭头长度占 y 轴范围的比例
            y_range = max(data) - min(data)
            arrow_dx = 0.5  # 箭头宽度（x 方向）
            arrow_dy = arrow_length * y_range  # 箭头高度（y 方向）

            # 计算箭头方向
            slope = (data[end] - data[start]) / (end - start)  # 波浪段斜率
            arrow_dy = arrow_dx * slope  # 箭头方向与波浪段一致

            # 使用 FancyArrowPatch 直接在波浪段末端绘制箭头
            arrow = FancyArrowPatch(
                posA=(end - arrow_dx, data[end] - arrow_dy),  # 起点（波浪段末端）
                posB=(end, data[end]),
                mutation_scale=10,  # 箭头大小
                arrowstyle="->",     # 箭头样式
                color=color,
                linewidth=linewidth,
                alpha=alpha,
            )
            plt.gca().add_patch(arrow)

    # 标记波峰和波谷
    if peaks is not None:
        for p in peaks:
            plt.scatter(p, data[p], c='red', s=100, edgecolor='black', linewidth=1.5, zorder=10)
            plt.text(p, data[p], f'{data[p]:.1f}', ha='center', va='bottom', fontsize=8)

    if valleys is not None:
        for v in valleys:
            plt.scatter(v, data[v], c='green', s=100, edgecolor='black', linewidth=1.5, zorder=10)
            plt.text(v, data[v], f'{data[v]:.1f}', ha='center', va='top', fontsize=8)

    # 优化图例
    plt.legend(handles, labels, loc='upper left')

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('时间索引', fontsize=12)
    plt.ylabel('价格', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ========================
# 测试主函数
# ========================
if __name__ == "__main__":
    # 示例1：简单锯齿（测试交替性）
    print("=" * 60)
    print("测试1：简单锯齿波")
    test_high = np.array([2, 9, 1, 5, 3, 4, 2, 7, 4, 6, 2, 9, 4])
    test_low = np.array([2, 9, 1, 5, 3, 4, 2, 7, 4, 6, 2, 9, 4])

    # 检测主波峰波谷
    peaks, valleys = detect_peaks_and_valleys(test_high, test_low)
    print(f"检测到波峰: {peaks}")
    print(f"检测到波谷: {valleys}")

    # 检测完整结构
    waves = detect_complete_wave_structure(test_high, test_low)

    print("\n波浪段检测结果：")
    for i, (start, end, level, is_rising) in enumerate(waves):
        direction = "↑" if is_rising else "↓"
        print(f"  段{i + 1:2d}: {start:2d} → {end:2d} (L{level}, {direction})")

    # 绘图
    plot_wave_structure(test_high, waves, peaks=peaks, valleys=valleys)

    # 示例2：你的原始测试数据
    print("\n" + "=" * 60)
    print("测试2：复杂波动")
    test_data = np.array([2, 9, 1, 5, 3, 4, 2, 7, 4, 6, 2, 9, 4])
    # 假设 high 和 low 相同（或可加噪声）
    waves2 = detect_complete_wave_structure(test_data, test_data)

    print("\n波浪段检测结果（复杂数据）：")
    for i, (start, end, level, is_rising) in enumerate(waves2):
        direction = "↑" if is_rising else "↓"
        print(f"  段{i + 1:2d}: {start:2d} → {end:2d} (L{level}, {direction})")

    plot_wave_structure('波浪结构检测（主次分明，颜色渐变，带方向箭头）', test_data, waves2)