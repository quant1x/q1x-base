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
from matplotlib.patches import FancyArrowPatch
from q1x.core import waves

# 线宽和透明度按层级衰减
def get_wave_style(level):
    linewidth = max(1, 3 - level)
    alpha = max(0.4, 0.8 - 0.2 * level)
    return linewidth, alpha


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
    # # 示例1：简单锯齿（测试交替性）
    # print("=" * 60)
    # print("测试1：简单锯齿波")
    # test_high = np.array([2, 9, 1, 5, 3, 4, 2, 7, 4, 6, 2, 9, 4])
    # test_low = np.array([2, 9, 1, 5, 3, 4, 2, 7, 4, 6, 2, 9, 4])
    #
    # # 检测主波峰波谷
    # peaks, valleys = waves.detect_peaks_and_valleys(test_high, test_low)
    # print(f"检测到波峰: {peaks}")
    # print(f"检测到波谷: {valleys}")
    #
    # # 检测完整结构
    # waves = detect_complete_wave_structure(test_high, test_low)
    #
    # print("\n波浪段检测结果：")
    # for i, (start, end, level, is_rising) in enumerate(waves):
    #     direction = "↑" if is_rising else "↓"
    #     print(f"  段{i + 1:2d}: {start:2d} → {end:2d} (L{level}, {direction})")
    #
    # # 绘图
    # plot_wave_structure('多级波段', test_high, waves, peaks=peaks, valleys=valleys)

    # 示例2：你的原始测试数据
    print("\n" + "=" * 60)
    print("测试2：复杂波动")
    test_data = np.array([2, 9, 1, 5, 3, 4, 2, 7, 4, 6, 2, 9, 4])
    # 假设 high 和 low 相同（或可加噪声）
    ws = waves.detect_complete_wave_structure(test_data, test_data)

    print("\n波浪段检测结果（复杂数据）：")
    for i, (start, end, level, is_rising) in enumerate(ws):
        direction = "↑" if is_rising else "↓"
        print(f"  段{i + 1:2d}: {start:2d} → {end:2d} (L{level}, {direction})")

    plot_wave_structure('波浪结构检测（主次分明，颜色渐变，带方向箭头）', test_data, ws)