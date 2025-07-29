#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project : q1x-base
@Package : core
@File    : waves_klines.py
@Author  : wangfeng
@Date    : 2025/7/29 14:09
@Desc    : 波浪检测 - k线测试
"""

import numpy as np
from typing import List, Tuple

from q1x.core import waves
from q1x.core.formula import *

GOLDEN_RATIO = 0.618  # 黄金分割比例 ← 新增变量

def calculate_t_values_with_positions(klines, peaks, valleys):
    """
    波峰波谷配对并计算T值，同时找到T值在数据中的实际位置

    Parameters:
    klines: K线数据，包含'high'和'low'列
    peaks: 波峰索引列表
    valleys: 波谷索引列表

    Returns:
    t_indices: T值在数据中的实际索引位置
    t_values: T值序列
    pairs: 配对的波峰波谷信息
    """

    # 将波峰和波谷合并并按时间排序
    extrema = sorted([(p, 'peak') for p in peaks] + [(v, 'valley') for v in valleys],
                     key=lambda x: x[0])

    t_indices = []
    t_values = []
    pairs = []

    # 寻找波谷-波峰配对
    for i in range(len(extrema) - 1):
        current_idx, current_type = extrema[i]
        next_idx, next_type = extrema[i + 1]

        # 确保是波谷后跟波峰的配对
        if current_type == 'valley' and next_type == 'peak':
            valley_idx = current_idx
            peak_idx = next_idx

            # 获取波峰和波谷的值
            peak_value = klines['high'].iloc[peak_idx]
            valley_value = klines['low'].iloc[valley_idx]

            # 计算T值：波峰 - (波峰-波谷) * GOLDEN_RATIO
            price_range = peak_value - valley_value
            t_value = peak_value - price_range * GOLDEN_RATIO

            # 在波峰之后的数据中寻找最接近T值的位置
            t_idx = find_closest_price_index(klines, peak_idx, t_value)

            t_indices.append(t_idx)
            t_values.append(t_value)
            pairs.append({
                'valley_idx': valley_idx,
                'peak_idx': peak_idx,
                'valley_value': valley_value,
                'peak_value': peak_value,
                't_target_value': t_value,
                't_actual_idx': t_idx,
                't_actual_value': klines['low'].iloc[t_idx] if t_idx < len(klines) else t_value
            })

    return t_indices, t_values, pairs

def find_closest_price_index(klines, start_idx, target_price):
    """
    在指定起始位置之后找到最接近目标价格的位置

    Parameters:
    klines: K线数据
    start_idx: 起始索引
    target_price: 目标价格

    Returns:
    closest_idx: 最接近目标价格的索引
    """
    min_diff = float('inf')
    closest_idx = start_idx

    # 在波峰之后的数据中寻找
    for i in range(start_idx + 1, min(start_idx + 50, len(klines))):  # 限制搜索范围
        # 可以检查high, low, close等价格
        prices = [klines['high'].iloc[i], klines['low'].iloc[i], klines['close'].iloc[i]]
        for price in prices:
            diff = abs(price - target_price)
            if diff < min_diff:
                min_diff = diff
                closest_idx = i

    return closest_idx if closest_idx != start_idx else start_idx + 5  # 默认向前5个位置

# 简化版本（推荐使用）
def get_peak_valley_pairs_with_t(klines, peaks, valleys):
    """
    获取波峰波谷配对及T值的简化版本

    Returns:
    results: 包含配对信息和T值的列表
    """

    # 合并并排序极值点
    extrema = sorted([(p, 'peak', klines['high'].iloc[p]) for p in peaks] +
                     [(v, 'valley', klines['low'].iloc[v]) for v in valleys],
                     key=lambda x: x[0])

    results = []

    # 寻找波谷-波峰配对
    for i in range(len(extrema) - 1):
        current_idx, current_type, current_value = extrema[i]
        next_idx, next_type, next_value = extrema[i + 1]

        # 波谷后跟波峰的配对
        if current_type == 'valley' and next_type == 'peak':
            valley_idx, valley_price = current_idx, current_value
            peak_idx, peak_price = next_idx, next_value

            # 计算T值
            price_range = peak_price - valley_price
            t_value = peak_price - price_range * 0.618

            results.append({
                'valley_index': valley_idx,
                'valley_price': valley_price,
                'peak_index': peak_idx,
                'peak_price': peak_price,
                'price_range': price_range,
                't_value': t_value,
                't_index': peak_idx  # T值标记在波峰位置
            })

    return results

from q1x.base import cache
import pandas as pd
import matplotlib.pyplot as plt

# 参数配置
VOLUME_SPIKE_THRESHOLD = 0.005
WINDOW = 13
MA_PERIODS = [5, 10, 20]
target_period = 'm' # 年线
period_name = cache.get_period_name(target_period)
target_tail = 100 # 尾部多少条数据

code = 'sz000158'
name = cache.stock_name(code)
print(f'{name}({code})')


# 数据加载
klines = cache.klines(code)
klines = cache.convert_klines_trading(klines, period=target_period)
if len(klines)>= target_tail:
    klines = klines.tail(target_tail)
klines['date'] = pd.to_datetime(klines['date'])
klines['x_pos'] = np.arange(len(klines))

# ====== 堆量检测 ======
klines['hist_ma'] = MA(klines['volume'], N=WINDOW, window_shift=1)
klines['spike_threshold'] = klines['hist_ma'] * (1 - VOLUME_SPIKE_THRESHOLD)
klines['is_spike'] = False
klines['locked_threshold'] = np.nan
klines['spike_high'] = np.nan  # 存储堆量区间最高价
klines['spike_low'] = np.nan  # 相邻区间最低价

current_threshold = None
spike_start = None  # 初始化spike_start变量
spike_intervals = []  # 存储所有堆量区间

# 标记最亮期间的最高价
for i in range(len(klines)):
    if current_threshold is None:
        if klines['volume'].iloc[i] >= klines['spike_threshold'].iloc[i]:
            current_threshold = klines['spike_threshold'].iloc[i]
            spike_start = i  # 记录堆量区间开始位置
            klines.loc[klines.index[i], 'is_spike'] = True
            klines.loc[klines.index[i], 'locked_threshold'] = current_threshold
    else:
        if klines['volume'].iloc[i] >= current_threshold:
            klines.loc[klines.index[i], 'is_spike'] = True
            klines.loc[klines.index[i], 'locked_threshold'] = current_threshold
        else:
            # 堆量区间结束，记录最高价
            if spike_start is not None:  # 确保spike_start已被赋值
                spike_end = i - 1
                # 记录堆量区间
                spike_intervals.append((spike_start, spike_end))
                max_high = klines['high'].iloc[spike_start:spike_end + 1].max()
                max_high_pos = klines['high'].iloc[spike_start:spike_end + 1].idxmax()
                klines.loc[max_high_pos, 'spike_high'] = max_high
            current_threshold = None
            spike_start = None

# 处理最后一个堆量区间（如果数据结束时仍在堆量区间）
if current_threshold is not None and spike_start is not None:
    spike_end = len(klines) - 1
    max_high = klines['high'].iloc[spike_start:spike_end + 1].max()
    max_high_pos = klines['high'].iloc[spike_start:spike_end + 1].idxmax()
    klines.loc[max_high_pos, 'spike_high'] = max_high

# ====== 新增：标记相邻区间最低价 ======
# 1. 首个堆量区间前的最低点
if len(spike_intervals) > 0:
    first_spike_start = spike_intervals[0][0]
    if first_spike_start > 0:  # 存在前置区间
        min_low = klines['low'].iloc[:first_spike_start].min()
        min_low_pos = klines['low'].iloc[:first_spike_start].idxmin()
        klines.loc[min_low_pos, 'spike_low'] = min_low

for i in range(1, len(spike_intervals)):
    prev_end = spike_intervals[i - 1][1]
    curr_start = spike_intervals[i][0]
    if curr_start > prev_end + 1:  # 确保区间不连续
        low_range_start = prev_end + 1
        low_range_end = curr_start - 1
        min_low = klines['low'].iloc[low_range_start:low_range_end + 1].min()
        min_low_pos = klines['low'].iloc[low_range_start:low_range_end + 1].idxmin()
        klines.loc[min_low_pos, 'spike_low'] = min_low

# ====== C点计算（修复版） ======
# 1. 找出所有A点和B点的位置
a_points = klines[klines['spike_low'].notna()].index
b_points = klines[klines['spike_high'].notna()].index

# 2. 构建有效的AB区间对（A在前，B在后）
ab_pairs = []
for a_idx in a_points:
    # 找到A点之后的第一个B点
    later_b = b_points[b_points > a_idx]
    if len(later_b) > 0:
        b_idx = later_b[0]
        A = klines.at[a_idx, 'spike_low']
        B = klines.at[b_idx, 'spike_high']
        if B > A:  # 只处理B>A的有效区间
            ab_pairs.append((A, B, a_idx, b_idx))

# 3. 为每个AB对计算C点
# ====== 正确的C点计算 ======
klines['c_point'] = np.nan
for A, B, a_idx, b_idx in ab_pairs:
    # 正确使用0.618黄金分割比例
    C = B - (B - A) * GOLDEN_RATIO

    # 严格在B点之后搜索
    window = klines.loc[b_idx + 1:]  # b_idx+1 确保在B点右侧

    if len(window) > 0:
        nearest_idx = (window['low'] - C).abs().idxmin()
        klines.at[nearest_idx, 'c_point'] = C

        # 修复后的调试信息（使用klines中的日期）
        a_date = klines.at[a_idx, 'date'].strftime('%Y-%m-%d')
        b_date = klines.at[b_idx, 'date'].strftime('%Y-%m-%d')
        c_date = klines.at[nearest_idx, 'date'].strftime('%Y-%m-%d')

        print(f"\nAB区间 {a_date} 到 {b_date}:")
        print(f"A点价格: {A:.2f} | B点价格: {B:.2f}")
        print(f"计算过程: {B:.2f} - ({B:.2f}-{A:.2f})×{GOLDEN_RATIO} = {C:.2f}")
        print(f"标注位置: {c_date} (B点后第{nearest_idx - b_idx}根K线)")
        print(
            f"最近收盘价: {klines.at[nearest_idx, 'close']:.2f} (差值: {abs(klines.at[nearest_idx, 'close'] - C):.2f})")
    else:
        b_date = klines.at[b_idx, 'date'].strftime('%Y-%m-%d')
        print(f"警告: {b_date}之后无数据，无法计算C点")

# ====== 计算均线 ======
for period in MA_PERIODS:
    klines[f'ma{period}'] = klines['close'].rolling(period).mean()

#peaks, valleys = find_peaks_valleys(klines['high'].values, klines['low'].values)
## 2. 优化过滤
#peaks, valleys = filter_extrema(peaks, valleys, klines['high'].values, klines['low'].values)
#peaks, valleys = detect_pure_extrema(klines['high'].values, klines['low'].values)
peaks, valleys = waves.detect_peaks_and_valleys(klines['high'].values, klines['low'].values)
t_indices, t_values, pairs = calculate_t_values_with_positions(klines, peaks, valleys)
# ====== 绘图 ======
fig, (ax_kline, ax_vol) = plt.subplots(2, 1, figsize=(20, 9),
                                       gridspec_kw={'height_ratios': [3, 1]},
                                       sharex=True)
# ====== 修改2：缩小K线和成交量柱宽度 ======
candle_width = 0.6  # 原0.8改为0.6 ← 这里改
vol_width = candle_width * 1  # 保持比例

# 1. 绘制K线
for idx, row in klines.iterrows():
    color = 'red' if row['close'] >= row['open'] else 'green'
    ax_kline.plot([row['x_pos'], row['x_pos']],
                  [row['low'], row['high']],
                  color=color, lw=0.8)
    ax_kline.bar(row['x_pos'],
                 row['close'] - row['open'],
                 bottom=row['open'],
                 width=candle_width, color=color,
                 edgecolor=color)

# 2. 绘制均线
ma_colors = ['gold', 'deepskyblue', 'darkviolet']
for period, color in zip(MA_PERIODS, ma_colors):
    ax_kline.plot(klines['x_pos'], klines[f'ma{period}'],
                  color=color, lw=1.5, label=f'{period}{period_name}均线')

ax_kline.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.5)
ax_vol.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.5)

annotate_font_size = 9
# # 3. 标注堆量区间最高价
# for idx, row in klines[klines['spike_high'].notna()].iterrows():
#     price_text = f"B: {row['spike_high']:.2f}"  # 格式化价格为两位小数
#     ax_kline.annotate(price_text,
#                       xy=(row['x_pos'], row['spike_high']),
#                       xytext=(0, 10),
#                       textcoords='offset points',
#                       ha='center',
#                       va='bottom',
#                       color='red',
#                       fontsize=annotate_font_size,
#                       fontweight='bold',
#                       bbox=dict(boxstyle='round,pad=0.3',
#                                 fc='white',
#                                 ec='blue',
#                                 lw=1,
#                                 alpha=0.8))
#
# # 4. 标注相邻区间最低价（A点）（新增部分）
# for idx, row in klines[klines['spike_low'].notna()].iterrows():
#     price_text = f"A: {row['spike_low']:.2f}"
#     ax_kline.annotate(price_text,
#                       xy=(row['x_pos'], row['spike_low']),
#                       xytext=(0, -15),  # 向下偏移
#                       textcoords='offset points',
#                       ha='center',
#                       va='top',
#                       color='darkgreen',
#                       fontsize=annotate_font_size,
#                       fontweight='bold',
#                       bbox=dict(boxstyle='round,pad=0.3',
#                                 fc='white',
#                                 ec='darkgreen',
#                                 lw=1,
#                                 alpha=0.8))
#
# # 2. 绘制C点标记（新增）
# for idx, row in klines[klines['c_point'].notna()].iterrows():
#     # 标记点
#     ax_kline.scatter(row['x_pos'], row['c_point'],
#                      color='darkorange', s=100, zorder=5,
#                      edgecolors='white', linewidths=1)
#
#     # 标记文本
#     ax_kline.annotate(f'C: {row["c_point"]:.2f}',
#                       xy=(row['x_pos'], row['c_point']),
#                       xytext=(0, -25),
#                       textcoords='offset points',
#                       ha='center',
#                       va='top',
#                       color='darkorange',
#                       fontsize=annotate_font_size,
#                       fontweight='bold',
#                       bbox=dict(boxstyle='round,pad=0.3',
#                                 fc='white',
#                                 ec='darkorange',
#                                 lw=1,
#                                 alpha=0.8))
#
#     # 参考线
#     ax_kline.axhline(row['c_point'],
#                      color='darkorange',
#                      linestyle=':',
#                      alpha=0.3,
#                      lw=1)

# 标注波峰(P)
for i, p in enumerate(peaks):
    if p < len(klines):  # 防止索引越界
        ax_kline.scatter(p, klines['high'].iloc[p],
                         color='red', marker='^', s=100, zorder=5,
                         label='Peak' if p == peaks[0] else "")
        ax_kline.text(p, klines['high'].iloc[p], f'P{i+1}: {klines["high"].iloc[p]:.2f}',
                      ha='center', va='bottom', color='red')

# 标注波谷(V)
for i, v in enumerate(valleys):
    print('valleys:', v)
    if v < len(klines):  # 防止索引越界
        ax_kline.scatter(v, klines['low'].iloc[v],
                         color='blue', marker='v', s=100, zorder=5,
                         label='Valley' if v == valleys[0] else "")
        ax_kline.text(v, klines['low'].iloc[v], f'V{i+1}: {klines["low"].iloc[v]:.2f}',
                      ha='center', va='top', color='blue')


# 标注T点
if t_indices and t_values:
    t_x = [i for i in t_indices if i < len(klines)]
    t_y = []
    for i in t_x:
        # 找到最接近T值的实际价格点
        if i < len(klines):
            # 在该位置的所有价格中找最接近T值的
            prices = [klines['high'].iloc[i], klines['low'].iloc[i], klines['close'].iloc[i]]
            closest_price = min(prices, key=lambda x: abs(x - t_values[t_x.index(i)]))
            t_y.append(closest_price)
        else:
            t_y.append(t_values[t_x.index(i)])

    ax_kline.scatter(t_x, t_y, color='red', s=120, marker='+', label='C点', zorder=5)

    # 为T点添加标签
    for i, (x, y, t_val) in enumerate(zip(t_x, t_y, t_values[:len(t_x)])):
        ax_kline.annotate(f'C{i+1}: {t_val:.2f}',
                          (x, y),
                          xytext=(5, 10),
                          textcoords='offset points',
                          fontsize=8,
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    # 在波峰和T点之间连线
    for pair, t_idx, t_val in zip(pairs, t_indices, t_values):
        peak_idx = pair['peak_idx']
        if peak_idx < len(klines) and t_idx < len(klines):
            peak_price = klines['high'].iloc[peak_idx]

            # 找到T点的实际坐标
            prices = [klines['high'].iloc[t_idx], klines['low'].iloc[t_idx], klines['close'].iloc[t_idx]]
            closest_price = min(prices, key=lambda x: abs(x - t_val))

            # 绘制连线
            ax_kline.plot([peak_idx, t_idx], [peak_price, closest_price],
                          'green', linestyle='--', alpha=0.7, linewidth=1.5,
                          label='B->C' if pair == pairs[0] else "")

# 添加配对连线（可选）
for pair in pairs:
    valley_idx = pair['valley_idx']
    peak_idx = pair['peak_idx']
    if valley_idx < len(klines) and peak_idx < len(klines):
        valley_price = klines['low'].iloc[valley_idx]
        peak_price = klines['high'].iloc[peak_idx]
        ax_kline.plot([valley_idx, peak_idx], [valley_price, peak_price],
                      'red', linestyle='--', alpha=0.7, linewidth=1.5, label='A->B' if pair == pairs[0] else "")

# 4. 绘制成交量
for idx, row in klines.iterrows():
    ax_vol.bar(row['x_pos'], row['volume'],
               width=vol_width, alpha=0.6,
               color='red' if row['is_spike'] else 'gray')
    if row['is_spike']:
        ax_vol.plot(row['x_pos'], row['volume'] * 1.02,
                    '^', color='darkred', markersize=6, alpha=0.8)

# 5. 绘制量能指标
ax_vol.plot(klines['x_pos'], klines['hist_ma'],
            'orange', lw=1.5, label=f'{WINDOW}{period_name}量均线')
ax_vol.plot(klines['x_pos'], klines['locked_threshold'],
            '--', color='purple', lw=1, alpha=0.5, label='锁定阈值')

# ====== 坐标轴范围控制 ======
ax_kline.set_xlim(-0.5, len(klines) - 0.5)
ax_kline.set_ylim(
    klines[['low', 'ma5', 'ma10', 'ma20']].min().min() * 0.995,
    klines[['high', 'ma5', 'ma10', 'ma20']].max().max() * 1.005
)
ax_vol.set_ylim(0, klines['volume'].max() * 1.2)

# 设置X轴刻度（确保显示所有日期）
ax_vol.set_xticks(klines['x_pos'])  # 所有位置都设刻度
ax_vol.set_xticklabels(klines['date'].dt.strftime('%Y-%m-%d'))  # 所有日期转文本

plt.setp(ax_vol.get_xticklabels(), rotation=45, ha='right')

# ====== 图例与标题 ======
ax_kline.legend(loc='upper left')
ax_vol.legend(loc='upper left')
ax_kline.set_title(f'{name}({code}) CT模型+堆量{period_name}线', pad=20, fontsize=14)
ax_vol.set_ylabel('成交量(手)')

plt.subplots_adjust(left=0.1, right=0.85, hspace=0.1)
plt.show()
