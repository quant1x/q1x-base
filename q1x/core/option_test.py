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
import akshare as ak

option_finance_board_df = ak.option_finance_board(symbol="沪深300股指期权", end_month="2508")
print(option_finance_board_df.columns)
print(option_finance_board_df.head())
print(option_finance_board_df.tail())

import akshare as ak

option_risk_indicator_sse_df = ak.option_risk_indicator_sse(date="20250731")
option_risk_indicator_sse_df.to_csv("option_risk_indicator_sse.csv")
print(option_risk_indicator_sse_df.columns)
print(option_risk_indicator_sse_df)