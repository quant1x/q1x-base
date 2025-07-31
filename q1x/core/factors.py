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
