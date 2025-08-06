#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project : q1x-base
@Package : core
@File    : test-snapshot-premarket.py
@Author  : wangfeng
@Date    : 2025/8/2 7:54
@Desc    : 盘前集合竞价
"""

import akshare as ak
stock_zh_a_hist_pre_min_em_df = ak.stock_zh_a_hist_pre_min_em(symbol="000158")
print(stock_zh_a_hist_pre_min_em_df)
