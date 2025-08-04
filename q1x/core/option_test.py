#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project : q1x-base
@Package : base
@File    : option.py
@Author  : wangfeng
@Date    : 2025/7/31 8:41
@Desc    : 期权数据 - 测试
"""
from q1x.core import option
import akshare as ak

option_finance_board_df = option.option_finance_board(symbol="华泰柏瑞沪深300ETF期权", end_month="2306")
print(option_finance_board_df.columns)
print('------')
print(option_finance_board_df)

option_risk_indicator_sse_df = option.option_risk_indicator_sse(date="20250731")
option_risk_indicator_sse_df.to_csv("option_risk_indicator_sse.csv")
print(option_risk_indicator_sse_df.columns)
print(option_risk_indicator_sse_df)

# option_daily_stats_sse_df = ak.option_daily_stats_sse(date="20240626")
# print(option_daily_stats_sse_df)