#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project : q1x-base
@Package : core
@File    : formula.py
@Author  : wangfeng
@Date    : 2025/7/29 13:17
@Desc    : 通达信函数
"""

import numpy as np
import pandas as pd

from typing import Callable, Any, Tuple, List


def rolling_apply(S, N, func: Callable[..., Any], window_shift: int = 0) -> np.ndarray:
    """
    滚动窗口计算函数，对序列 S 进行滚动窗口计算，并返回结果数组。

    Parameters
    ----------
    S : array-like
        输入的序列数据，可以是 list、NumPy array 或 Pandas Series。
    N : int, float, array-like
        窗口大小。可以是固定值（所有窗口相同）或动态数组（每个位置窗口大小不同）。
    func : Callable[[array-like, int], Any]
        计算函数，接受窗口数据和窗口大小两个参数。
    window_shift : int, optional
        窗口偏移量：
        - 0：标准滚动窗口（包含当前数据）
        - 正整数n：窗口向右偏移n个位置（排除最近的n个数据）
        默认为0。

    Returns
    -------
    np.ndarray
        计算结果数组，长度与 S 相同。
    """
    s_len = len(S)
    ret = np.repeat(np.nan, s_len)

    if isinstance(N, (int, float)):
        N = np.repeat(N, s_len)

    for i in range(s_len):
        if np.isnan(N[i]):
            continue

        window = int(N[i])
        if window_shift > 0:
            # 偏移窗口：S[i+1-window-window_shift : i+1-window_shift]
            start = max(0, i + 1 - window - window_shift)
            end = max(0, i + 1 - window_shift)
            T = S[start:end]
        else:
            # 标准窗口：S[i+1-window : i+1]
            if window <= i + 1:
                T = S[i + 1 - window: i + 1]

        # 只有当窗口数据不为空时才计算
        if 'T' in locals() and len(T) > 0:
            ret[i] = func(T, len(T))

    return ret


def MA(S, N, window_shift: int = 0):
    """
    计算移动平均

    Parameters
    ----------
    S : array-like
        输入序列
    N : int
        窗口大小
    window_shift : int, optional
        窗口偏移量，默认为0

    Returns
    -------
    np.ndarray
        移动平均序列
    """
    return rolling_apply(S, N, lambda T, W: np.mean(T), window_shift)


VOLUME_SPIKE_THRESHOLD = 0.005  # 全大写表示常量
