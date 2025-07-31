#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project : q1x-base
@Package : 
@File    : factors.py
@Author  : wangfeng
@Date    : 2025/7/31 9:17
@Desc    : 因子
"""


import numpy as np
import pandas as pd

def hidden_spread(mid_price, bid, ask):
    """
    隐形价差(率)

    隐性价差 > 0.03% 时视为流动性缺口信号
    Args:
        mid_price: 中间价
        bid: 买入价
        ask: 卖出价

    Returns:
        隐形价差=(ask-bid) / mid_price
    """
    return (ask - bid) / mid_price


def volatility(price_series, window=30):
    """
    移动平均波动率
    Args:
        price_series: 价格序列
        window: 滑动窗口, 默认30

    Returns:

    """
    returns = np.log(price_series / price_series.shift(1)).dropna()
    return returns.rolling(window).std() * np.sqrt(252)
