#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project : q1x-base
@Package : 
@File    : vix_test4.py
@Author  : wangfeng
@Date    : 2025/8/1 11:14
@Desc    : 
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from q1x.base import cache

# 参数设置
target_period = 'd'  # 日线数据
target_tail = 60    # 取最近60天数据
code = 'sh603730'
name = cache.stock_name(code)
print(f'{name}({code})')

# 数据加载
klines = cache.klines(code)
klines = cache.convert_klines_trading(klines, period=target_period)
if target_tail > 0 and len(klines) >= target_tail:
    klines = klines.tail(target_tail)
klines['date'] = pd.to_datetime(klines['date'])

# 确保数据类型正确
klines['high'] = klines['high'].astype(float)
klines['low'] = klines['low'].astype(float)
klines['close'] = klines['close'].astype(float)

# 计算Parkinson波动率
def parkinson_volatility(high, low, window=20):
    log_hl = np.log(high/low)
    parkinson = np.sqrt(1/(4*window*np.log(2)) * (log_hl**2).rolling(window).sum())
    return parkinson

window = 20  # 20日窗口
klines['Parkinson_Vol'] = parkinson_volatility(klines['high'], klines['low'], window)
klines['Parkinson_Vol_Annual'] = klines['Parkinson_Vol'] * np.sqrt(252)  # 年化
klines['Parkinson_Vol_Pct'] = klines['Parkinson_Vol_Annual'] * 100  # 百分比表示

# 与传统历史波动率比较
klines['Log_Return'] = np.log(klines['close']/klines['close'].shift(1))
klines['HV_20D'] = klines['Log_Return'].rolling(window).std() * np.sqrt(252) * 100

# 可视化
plt.figure(figsize=(14, 7))
plt.plot(klines['date'], klines['Parkinson_Vol_Pct'], label='Parkinson Volatility (20D)', color='red')
plt.plot(klines['date'], klines['HV_20D'], label='Traditional HV (20D)', color='blue', linestyle='--')
plt.title(f'{name}({code}) - Parkinson vs Traditional Historical Volatility')
plt.xlabel('Date')
plt.ylabel('Volatility (%)')
plt.legend()
plt.grid()
plt.show()

# 输出最新波动率
latest_parkinson = klines['Parkinson_Vol_Pct'].iloc[-1]
latest_hv = klines['HV_20D'].iloc[-1]
print(f'Latest 20D Parkinson Volatility: {latest_parkinson:.2f}%')
print(f'Latest 20D Traditional HV: {latest_hv:.2f}%')