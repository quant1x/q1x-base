#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project : q1x-base
@Package : core
@File    : power_test.py
@Author  : wangfeng
@Date    : 2025/7/30 10:39
@Desc    : 买卖力度分析（修正版）
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates

from q1x.base import cache

# 初始化参数
target_period = 'd'  # 周线
target_tail = 89     # 获取最近89条数据
code = 'sz000158'

# 1. 加载真实K线数据
try:
    klines = cache.klines(code)
    klines = cache.convert_klines_trading(klines, period=target_period)
    klines = klines.tail(target_tail) if len(klines) >= target_tail else klines

    if len(klines) == 0:
        raise ValueError("获取到的K线数据为空，请检查数据源")

except Exception as e:
    raise SystemExit(f"数据加载失败: {str(e)}")

# 2. 数据预处理
df = klines.copy()
df['date'] = pd.to_datetime(df['date'])
df['x_pos'] = np.arange(len(df))  # 为绘图准备x轴坐标

# 3. 核心计算逻辑
# 在 calculate_power_indicators 函数中，替换 ATR 相关逻辑
def calculate_power_indicators(df):
    """计算买卖力量指标（含优化ATR）"""
    # 买卖力量
    df['BuyPower'] = df['close'] - df['low']
    df['SellPower'] = df['high'] - df['close']
    df['PowerDiff'] = df['BuyPower'] - df['SellPower']

    # 影线分析
    df['UpperShadow'] = df['high'] - np.maximum(df['open'], df['close'])
    df['LowerShadow'] = np.minimum(df['open'], df['close']) - df['low']
    df['Body'] = abs(df['close'] - df['open'])

    # === ✅ 优化版 ATR 计算 ===
    atr_window = 14

    # 1. 计算 True Range (TR)
    tr0 = df['high'] - df['low']
    tr1 = abs(df['high'] - df['close'].shift(1))
    tr2 = abs(df['low'] - df['close'].shift(1))
    df['TR'] = pd.concat([tr0, tr1, tr2], axis=1).max(axis=1)

    # 2. 使用 Wilder's 惯性平滑计算 ATR
    atr = [np.nan] * len(df)
    if len(df) > atr_window:
        # 初始ATR = 前N个TR的简单平均
        atr[atr_window - 1] = df['TR'][:atr_window].mean()

        # 递推公式：ATR = (前一日ATR × 13 + 当日TR) / 14
        for i in range(atr_window, len(df)):
            atr[i] = (atr[i-1] * (atr_window - 1) + df['TR'].iloc[i]) / atr_window

    df['ATR'] = pd.Series(atr, index=df.index)

    # 3. 动态阈值
    df['DynamicThreshold'] = df['ATR'] * 0.3

    # 成交量均线
    df['VolumeAvg'] = df['volume'].rolling(5).mean()

    # 信号生成（保持不变）
    df['BearishSignal'] = (df['UpperShadow'] >= 2 * df['Body']) & (df['PowerDiff'] < 0)
    df['BullishSignal'] = (df['LowerShadow'] >= 2 * df['Body']) & (df['PowerDiff'] > 0)
    df['DynamicBearish'] = (df['UpperShadow'] >= df['DynamicThreshold']) & (df['PowerDiff'] < 0) & (df['volume'] > df['VolumeAvg'])
    df['DynamicBullish'] = (df['LowerShadow'] >= df['DynamicThreshold']) & (df['PowerDiff'] > 0) & (df['volume'] > df['VolumeAvg'])

    return df

df = calculate_power_indicators(df)

# 4. 专业K线图绘制
def plot_pro_candlestick(ax, df):
    """绘制专业级K线图"""
    for idx, row in df.iterrows():
        # 确定颜色（涨跌）
        color = 'red' if row['close'] >= row['open'] else 'green'

        # 绘制影线
        ax.plot([row['x_pos'], row['x_pos']],
                [row['low'], row['high']],
                color=color, linewidth=1)

        # 绘制实体
        rect = Rectangle((row['x_pos'] - 0.3, min(row['open'], row['close'])),
                         0.6, abs(row['close'] - row['open']),
                         facecolor=color, edgecolor=color)
        ax.add_patch(rect)

        # 标记信号
        if row['DynamicBearish']:
            ax.scatter(row['x_pos'], row['high'], color='darkred', marker='v', s=80, zorder=5)
        if row['DynamicBullish']:
            ax.scatter(row['x_pos'], row['low'], color='darkgreen', marker='^', s=80, zorder=5)

# 5. 创建专业图表（修正版）
plt.style.use('ggplot')
fig = plt.figure(figsize=(16, 11))
gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 2])  # 调整高度：主图、成交量、力量图

# K线主图
ax1 = fig.add_subplot(gs[0])
plot_pro_candlestick(ax1, df)
ax1.set_title(f'{cache.stock_name(code)}({code}) - 买卖力量分析', fontsize=14)
ax1.grid(True, alpha=0.3)

# 成交量图
ax2 = fig.add_subplot(gs[1], sharex=ax1)
ax2.bar(df['x_pos'], df['volume'], width=0.6,
        color=np.where(df['close'] >= df['open'], 'red', 'green'))
ax2.set_ylabel('成交量')
ax2.grid(True, alpha=0.3)

# === 力量与影线分析图（核心：包含 DynamicThreshold）===
ax3 = fig.add_subplot(gs[2], sharex=ax1)

# 1. 绘制上影线（正值区域）
ax3.bar(df['x_pos'] - 0.2, df['UpperShadow'], width=0.4, color='orange', label='上影线（空头试探）')

# 2. 绘制下影线（负值区域）
ax3.bar(df['x_pos'] + 0.2, -df['LowerShadow'], width=0.4, color='cyan', label='下影线（多头试探）')

# 3. 绘制力量差（居中，透明度低）
ax3.bar(df['x_pos'], df['PowerDiff'], width=0.6, color=np.where(df['PowerDiff'] > 0, 'red', 'green'),
        alpha=0.5, label='力量差（收盘-开盘）')

# 4. 绘制动态阈值线（正区域）
ax3.plot(df['x_pos'], df['DynamicThreshold'], color='purple', linestyle='-', linewidth=1.5,
         label='动态阈值 (ATR×0.3)', zorder=5)

# 5. 绘制对称的负向阈值线（便于视觉对比）
ax3.plot(df['x_pos'], -df['DynamicThreshold'], color='purple', linestyle='-', linewidth=1.5)

# 6. 添加零轴
ax3.axhline(0, color='gray', linestyle='--', linewidth=1)

# 7. 图表设置
ax3.set_ylabel('买卖力量分解')
ax3.legend(loc='upper right', fontsize=10)
ax3.grid(True, alpha=0.3)

# 8. 设置x轴
ax3.set_xticks(df['x_pos'][::5])
ax3.set_xticklabels(df['date'].dt.strftime('%Y-%m-%d')[::5], rotation=45)
plt.tight_layout()
plt.show()

# 6. 输出关键信号
print("最近5个交易日的信号：")
print(df[['date', 'close', 'PowerDiff', 'DynamicBearish', 'DynamicBullish']].tail(5))