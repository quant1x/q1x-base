#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project : q1x-base
@Package : core
@File    : waves_release.py
@Author  : wangfeng
@Date    : 2025/7/30 11:05
@Desc    : 波浪+多空力度
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from q1x.base import cache
from q1x.core import formula, waves

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


def find_closest_price_index_v1(klines, start_idx, target_price):
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
    for i in range(start_idx + 1, len(klines)):  # 限制搜索范围
        # 可以检查high, low, close等价格
        prices = [klines['high'].iloc[i], klines['low'].iloc[i], klines['close'].iloc[i]]
        for price in prices:
            diff = abs(price - target_price)
            if diff < min_diff:
                min_diff = diff
                closest_idx = i

    return closest_idx if closest_idx != start_idx else start_idx + 5  # 默认向前5个位置

def find_closest_price_index(klines, start_idx, target_price):
    """
    在指定起始位置之后，找到第一个最低价（low）低于或等于目标价格的K线索引。

    Parameters:
    klines: K线数据
    start_idx: 起始索引（搜索从此处之后开始）
    target_price: 目标价格

    Returns:
    idx: 找到的第一个符合条件的K线索引，如果未找到则返回默认值。
    """
    # 在起始索引之后的所有数据中寻找
    for i in range(start_idx + 1, len(klines)):
        if klines['low'].iloc[i] <= target_price:
            return i  # 找到第一个符合条件的，立即返回

    # 如果循环结束都没找到，则返回一个默认值（例如起始索引+5）
    return find_closest_price_index_v1(klines, start_idx, target_price)

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


# 参数配置
VOLUME_SPIKE_THRESHOLD = 0.005
WINDOW = 13
MA_PERIODS = [5, 10, 20]
target_period = 'd'  # 周期
period_name = cache.get_period_name(target_period)
target_tail = 89  # 尾部多少条数据
SHOW_SHADOW = False

code = 'sh000001'
name = cache.stock_name(code)
print(f'{name}({code})')

# 数据加载
klines = cache.klines(code)
klines = cache.convert_klines_trading(klines, period=target_period)
if target_tail>0 and len(klines) >= target_tail:
    klines = klines.tail(target_tail)
klines['date'] = pd.to_datetime(klines['date'])
klines['x_pos'] = np.arange(len(klines))

# ====== 堆量检测 ======
klines['hist_ma'] = formula.MA(klines['volume'], N=WINDOW, window_shift=1)
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

# peaks, valleys = find_peaks_valleys(klines['high'].values, klines['low'].values)
## 2. 优化过滤
# peaks, valleys = filter_extrema(peaks, valleys, klines['high'].values, klines['low'].values)
# peaks, valleys = detect_pure_extrema(klines['high'].values, klines['low'].values)
peaks, valleys = waves.detect_peaks_and_valleys(klines['high'].values, klines['low'].values)
#peaks, valleys = waves.standardize_peaks_valleys(peaks, valleys, klines['high'].values, klines['low'].values)
for p in peaks:
    print(klines.iloc[p]['high'])
t_indices, t_values, pairs = calculate_t_values_with_positions(klines, peaks, valleys)

# ====== 绘图（三子图：K线 + 成交量 + 多空力度）======
fig, (ax_kline, ax_vol, ax_power) = plt.subplots(3, 1, figsize=(16, 9),
                                                 gridspec_kw={'height_ratios': [3, 1, 1], 'hspace': 0.1},
                                                 sharex=True)

# ====== 1. 绘制K线 ======
candle_width = 0.6
ma_colors = ['gold', 'deepskyblue', 'darkviolet']

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

# 绘制均线
for period, color in zip(MA_PERIODS, ma_colors):
    ax_kline.plot(klines['x_pos'], klines[f'ma{period}'],
                  color=color, lw=1.5, label=f'{period}{period_name}均线')

# ====== 2. 标注波峰(P)、波谷(V)、C点(T)等 ======
# 标注波峰(P)
for i, p in enumerate(peaks):
    if p < len(klines):
        ax_kline.scatter(p, klines['high'].iloc[p],
                         color='red', marker='^', s=100, zorder=5)
        ax_kline.text(p, klines['high'].iloc[p], f'P{i + 1}: {klines["high"].iloc[p]:.2f}',
                      ha='center', va='bottom', color='red', fontsize=8)

# 标注波谷(V)
for i, v in enumerate(valleys):
    if v < len(klines):
        ax_kline.scatter(v, klines['low'].iloc[v],
                         color='blue', marker='v', s=100, zorder=5)
        ax_kline.text(v, klines['low'].iloc[v], f'V{i + 1}: {klines["low"].iloc[v]:.2f}',
                      ha='center', va='top', color='blue', fontsize=8)

# 标注T点（即C点）
if t_indices and t_values:
    t_x = [i for i in t_indices if i < len(klines)]
    t_y = [klines['low'].iloc[i] for i in t_x]  # 用low近似
    ax_kline.scatter(t_x, t_y, color='darkorange', s=120, marker='+', zorder=5, linewidths=2)
    for i, (x, y, t_val) in enumerate(zip(t_x, t_y, t_values[:len(t_x)])):
        ax_kline.annotate(f'C{i + 1}: {t_val:.2f}',
                          (x, y),
                          xytext=(5, 10),
                          textcoords='offset points',
                          fontsize=8,
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.7))

# 连接波谷->波峰->T点
for pair in pairs:
    valley_idx = pair['valley_idx']
    peak_idx = pair['peak_idx']
    if valley_idx < len(klines) and peak_idx < len(klines):
        ax_kline.plot([valley_idx, peak_idx],
                      [klines['low'].iloc[valley_idx], klines['high'].iloc[peak_idx]],
                      'red', linestyle='--', alpha=0.7, linewidth=1.5)
        t_idx = pair['t_actual_idx']
        if t_idx < len(klines):
            t_price = klines['low'].iloc[t_idx]
            ax_kline.plot([peak_idx, t_idx],
                          [klines['high'].iloc[peak_idx], t_price],
                          'green', linestyle='--', alpha=0.7, linewidth=1.5)

# ====== 新增：判断高点和低点连线是否存在未来交叉，并绘制延长线 ======
def line_intersection(p1, p2, p3, p4):
    """
    计算两条直线 (p1,p2) 和 (p3,p4) 的交点。
    p = (x, y)
    返回交点 (x, y) 或 None（无交点或平行）。
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if abs(denom) < 1e-10:
        return None

    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    if ua >= 0:
        x = x1 + ua * (x2 - x1)
        y = y1 + ua * (y2 - y1)
        return (x, y)
    return None

# 获取最近的两个高点和低点
last_index = len(klines) - 1  # 最后一根K线的索引

# 过滤掉在最后一根K线上的波峰和波谷
filtered_peaks = [p for p in peaks if p != last_index]
filtered_valleys = [v for v in valleys if v != last_index]

print(f"原始波峰: {sorted(peaks)}, 过滤后: {sorted(filtered_peaks)}")
print(f"原始波谷: {sorted(valleys)}, 过滤后: {sorted(filtered_valleys)}")

# 检查过滤后是否至少有两个波峰和两个波谷
if len(filtered_peaks) >= 2 and len(filtered_valleys) >= 2:
    # 取最近的两个高点
    recent_peaks = sorted(filtered_peaks)[-2:]
    p1_idx, p2_idx = recent_peaks[0], recent_peaks[1]
    p1 = (klines['x_pos'].iloc[p1_idx], klines['high'].iloc[p1_idx])
    p2 = (klines['x_pos'].iloc[p2_idx], klines['high'].iloc[p2_idx])

    # 计算高点趋势延长线的斜率和截距
    slope_high = (p2[1] - p1[1]) / (p2[0] - p1[0]) if (p2[0] - p1[0]) != 0 else 0
    high_line_x = [p1[0], klines['x_pos'].iloc[-1]]  # 延伸到最后一根K线
    high_line_y = [p1[1], p2[1] + slope_high * (klines['x_pos'].iloc[-1] - p2[0])]

    # 绘制高点趋势延长线
    ax_kline.plot(high_line_x, high_line_y, color='purple', linestyle='-', linewidth=2.5, alpha=0.9, label='高点趋势延长线')

    # 取最近的两个低点
    recent_valleys = sorted(filtered_valleys)[-2:]
    v1_idx, v2_idx = recent_valleys[0], recent_valleys[1]
    v1 = (klines['x_pos'].iloc[v1_idx], klines['low'].iloc[v1_idx])
    v2 = (klines['x_pos'].iloc[v2_idx], klines['low'].iloc[v2_idx])

    # 计算低点趋势延长线的斜率和截距
    slope_low = (v2[1] - v1[1]) / (v2[0] - v1[0]) if (v2[0] - v1[0]) != 0 else 0
    low_line_x = [v1[0], klines['x_pos'].iloc[-1]]  # 延伸到最后一根K线
    low_line_y = [v1[1], v2[1] + slope_low * (klines['x_pos'].iloc[-1] - v2[0])]

    # 绘制低点趋势延长线
    ax_kline.plot(low_line_x, low_line_y, color='teal', linestyle='-', linewidth=2.5, alpha=0.9, label='低点趋势延长线')

    # 计算两条延长线的交点
    intersection = line_intersection(p1, (klines['x_pos'].iloc[-1], high_line_y[1]), v1, (klines['x_pos'].iloc[-1], low_line_y[1]))

    if intersection is not None:
        x_intersect, y_intersect = intersection
        last_x_pos = klines['x_pos'].iloc[-1]

        # 判断是否为未来交叉（交点在当前数据右侧）
        print(x_intersect, y_intersect, last_x_pos)
        if x_intersect > last_x_pos:
            # 验证交点价格是否在某根K线的high和low之间
            valid = False
            for i in range(len(klines)):
                if (
                        klines['low'].iloc[i] <= y_intersect <= klines['high'].iloc[i]
                        or klines['low'].iloc[i] >= y_intersect >= klines['high'].iloc[i]
                ):
                    valid = True
                    break

            if valid:
                # # 计算预测的日期（线性插值）
                # date_range = pd.to_datetime(klines['date']).values
                # first_date = date_range[0].astype('datetime64[D]')
                # last_date = date_range[-1].astype('datetime64[D]')
                # total_days = (last_date - first_date).astype(int)
                # days_per_step = total_days / (len(klines) - 1) if len(klines) > 1 else 1
                # predicted_day = first_date + np.timedelta64(int(x_intersect * days_per_step), 'D')
                #
                # try:
                #     predicted_date_str = predicted_day.strftime('%Y-%m-%d')
                # except:
                #     predicted_date_str = f"Day{x_intersect:.1f}"

                # ✅ 修正：将 x_intersect 转换为实际日期
                # 方法：将浮点索引四舍五入到最近的整数索引
                intersect_index = int(round(x_intersect))
                # 确保索引在有效范围内
                if 0 <= intersect_index < len(klines):
                    # 从klines中获取实际日期
                    actual_date = klines['date'].iloc[intersect_index]
                    date_str = actual_date.strftime('%Y-%m-%d')
                else:
                    # 理论上不会发生，因为x_intersect <= last_x_pos
                    date_str = f"索引{intersect_index}"

                # 在图上标注预测交点
                ax_kline.scatter(x_intersect, y_intersect, color='darkred', s=150, marker='x', zorder=10, linewidth=3)
                ax_kline.annotate(f'预测交叉\n{date_str}\n价格: {y_intersect:.2f}',
                                  (x_intersect, y_intersect),
                                  xytext=(10, -15),
                                  textcoords='offset points',
                                  fontsize=9,
                                  color='darkred',
                                  bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.8),
                                  arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5))

                print(f"\n✅ 预测趋势线未来交叉:")
                print(f"预测日期: {date_str}, 价格: {y_intersect:.2f}")
            else:
                print(f"\n⚠️  虽然计算出交点，但价格不在任何K线的high和low之间，交点无效。")
        else:
            print(f"\nℹ️  趋势线已在历史或当前交叉，位置: x={x_intersect:.1f}, 价格={y_intersect:.2f}")
    else:
        print(f"\nℹ️  高/低点趋势线无交点（可能平行）。")
else:
    missing = "高点" if len(filtered_peaks) < 2 else ""
    missing += "和" if (len(filtered_peaks) < 2 and len(filtered_valleys) < 2) else ""
    missing += "低点" if len(filtered_valleys) < 2 else ""
    print(f"\nℹ️  无法绘制趋势线：{missing}不足两个（过滤后）。")

# ====== 3. 绘制成交量 ======
vol_width = candle_width * 1
for idx, row in klines.iterrows():
    ax_vol.bar(row['x_pos'], row['volume'],
               width=vol_width, alpha=0.6,
               color='red' if row['is_spike'] else 'gray')
    if row['is_spike']:
        ax_vol.plot(row['x_pos'], row['volume'] * 1.02,
                    '^', color='darkred', markersize=6, alpha=0.8)

ax_vol.plot(klines['x_pos'], klines['hist_ma'],
            'orange', lw=1.5, label=f'{WINDOW}{period_name}量均线')
ax_vol.plot(klines['x_pos'], klines['locked_threshold'],
            '--', color='purple', lw=1, alpha=0.5, label='锁定阈值')

# ====== 4. 新增：多空力度分析图 ======
# ====== 统一计算技术指标 ======
klines['Body'] = abs(klines['close'] - klines['open'])
klines['UpperShadow'] = klines['high'] - np.maximum(klines['open'], klines['close'])
klines['LowerShadow'] = np.minimum(klines['open'], klines['close']) - klines['low']
klines['BuyPower'] = klines['close'] - klines['low']
klines['SellPower'] = klines['high'] - klines['close']
klines['PowerDiff'] = klines['BuyPower'] - klines['SellPower']

# ATR 和 DynamicThreshold
atr_window = 14
tr0 = klines['high'] - klines['low']
tr1 = (klines['high'] - klines['close'].shift(1)).abs()
tr2 = (klines['low'] - klines['close'].shift(1)).abs()
klines['TR'] = pd.concat([tr0, tr1, tr2], axis=1).max(axis=1)
klines['ATR'] = klines['TR'].ewm(alpha=1/atr_window, adjust=False).mean()
klines['DynamicThreshold'] = klines['ATR'] * 0.3

# 4.3 绘制
ax_power.bar(klines['x_pos'] - 0.2, klines['BuyPower'], width=0.4, color='orange', label='主动买入', alpha=0.7)
ax_power.bar(klines['x_pos'] + 0.2, -klines['SellPower'], width=0.4, color='cyan', label='主动卖出', alpha=0.7)
ax_power.bar(klines['x_pos'], klines['PowerDiff'], width=0.6, color=np.where(klines['PowerDiff'] > 0, 'red', 'green'), alpha=0.5, label='力度差')
ax_power.plot(klines['x_pos'], klines['DynamicThreshold'], color='purple', linestyle='-', linewidth=1.5, label='动态阈值')
ax_power.plot(klines['x_pos'], -klines['DynamicThreshold'], color='purple', linestyle='-', linewidth=1.5)
ax_power.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax_power.set_ylabel('多空力度')
ax_power.legend(loc='upper right', fontsize=9)
ax_power.grid(True, alpha=0.3)

# ====== 新增：长影线预警（避免与P/V三角重合）======
if SHOW_SHADOW:
    # 2. 定义“中等K线”阈值（基于价格百分比）
    price_level = klines['close'].median()  # 用中位数作为参考价格
    min_body_pct = 0.005  # 最小体幅：0.5%
    max_body_pct = 0.03   # 最大体幅：3.0%
    min_body_threshold = price_level * min_body_pct
    max_body_threshold = price_level * max_body_pct

    # 3. 筛选中等K线
    klines['is_medium_candle'] = (klines['Body'] >= min_body_threshold) & (klines['Body'] <= max_body_threshold)
    SHADOW_ALERT_RATIO = 1
    # 4. 上影线预警：长上影 + 中等实体
    upper_alert = klines[
        (klines['UpperShadow'] >= SHADOW_ALERT_RATIO * klines['Body']) &
        (klines['is_medium_candle'])
        ]

    # 5. 下影线预警：长下影 + 中等实体
    lower_alert = klines[
        (klines['LowerShadow'] >= SHADOW_ALERT_RATIO * klines['Body']) &
        (klines['is_medium_candle'])
        ]

    # 绘制上影线预警（> 黄色）
    for idx in upper_alert.index:
        row = klines.loc[idx]  # 使用 loc 获取整行
        high_price = row['high']
        offset_high = high_price * 1.01
        x_pos = row['x_pos']  # 使用 x_pos 作为x坐标
        ax_kline.scatter(x_pos, offset_high,
                         color='yellow', marker='>', s=120, zorder=6,
                         edgecolors='green', linewidth=1)
        ax_kline.text(x_pos, offset_high * 1.005, 'S',
                      ha='center', va='bottom', color='green', fontsize=10, fontweight='bold')

    # 绘制下影线预警（< 洋红）
    for idx in lower_alert.index:
        row = klines.loc[idx]
        low_price = row['low']
        offset_low = low_price * 0.99
        x_pos = row['x_pos']
        ax_kline.scatter(x_pos, offset_low,
                         color='magenta', marker='<', s=120, zorder=6,
                         edgecolors='red', linewidth=1)
        ax_kline.text(x_pos, offset_low * 0.995, 'B',
                      ha='center', va='top', color='red', fontsize=10, fontweight='bold')

# ====== 坐标轴范围控制 ======
ax_kline.set_xlim(-0.5, len(klines) - 0.5)
ax_kline.set_ylim(
    klines[['low', 'ma5', 'ma10', 'ma20']].min().min() * 0.995,
    klines[['high', 'ma5', 'ma10', 'ma20']].max().max() * 1.005
)
ax_vol.set_ylim(0, klines['volume'].max() * 1.2)
ax_power.set_ylim(-klines['UpperShadow'].max() * 1.3, klines['UpperShadow'].max() * 1.3)  # 力度图y轴

# ====== ✅ 正确设置x轴标签：只在最下面的子图显示 ======
ax_kline.set_xlim(-0.5, len(klines) - 0.5)
ax_kline.set_ylim(
    klines[['low', 'ma5', 'ma10', 'ma20']].min().min() * 0.995,
    klines[['high', 'ma5', 'ma10', 'ma20']].max().max() * 1.005
)
ax_vol.set_ylim(0, klines['volume'].max() * 1.2)

# ❌ 关闭上面的标签显示
ax_kline.set_xticklabels([])

# 如果有力度图 ax_power，则重复上述逻辑
if 'ax_power' in locals():
    ax_power.set_xticks(klines['x_pos'])
    ax_power.set_xticklabels(klines['date'].dt.strftime('%Y-%m-%d'), rotation=45, ha='right')
    ax_power.grid(True, alpha=0.3)

# ====== 图例与标题 ======
ax_kline.legend(loc='upper left')
ax_vol.legend(loc='upper left')
ax_power.legend(loc='upper left')
ax_kline.set_title(f'{name}({code}) CT模型+堆量{period_name}线 + 买卖力度分析', pad=20, fontsize=14)
ax_vol.set_ylabel('成交量(手)')
ax_power.set_ylabel('力度分析')

plt.subplots_adjust(left=0.1, right=0.85, hspace=0.1, bottom=0.15)  # 增加底部空间以容纳日期
plt.show()