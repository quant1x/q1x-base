# -*- coding: UTF-8 -*-
import os
import time
from enum import Enum
from functools import lru_cache

import pandas

from q1x.base import base, timestamp


class MarketType(Enum):
    SHENZHEN = 0  # 深圳
    SHANGHAI = 1  # 上海
    BEIJING = 2  # 北京
    HONGKONG = 21  # 香港
    USA = 22  # 美国


STOCK_DELISTING = "DELISTING"  # 退市

MARKET_SHANGHAI = "sh"  # 上海
MARKET_SHENZHEN = "sz"  # 深圳
MARKET_BEIJING = "bj"  # 北京
MARKET_HONGKONG = "hk"  # 香港
MARKET_USA = "us"  # 美国

MARKET_CN_FIRST_DATE = "19901219"  # 上证指数的第一个交易日
MARKET_CH_FIRST_LISTTIME = "1990-12-19"  # 个股上市日期

MARKET_FLAGS = ["sh", "sz", "SH", "SZ", "bj", "BJ", "hk", "HK", "us", "US"]
MARKET_A_SHARE_FLAGS = ["sh", "sz", "SH", "SZ", "bj", "BJ"]


# Helper functions for string operations
def starts_with(s: str, prefixes) -> bool:
    return any(s.startswith(prefix) for prefix in prefixes)


def ends_with(s: str, suffixes) -> bool:
    return any(s.endswith(suffix) for suffix in suffixes)


def get_security_code(market: MarketType, symbol: str) -> str:
    symbol = symbol.strip()
    if market == MarketType.USA:
        return f"{MARKET_USA}{symbol}"
    elif market == MarketType.HONGKONG:
        return f"{MARKET_HONGKONG}{symbol[:5]}"
    elif market == MarketType.BEIJING:
        return f"{MARKET_BEIJING}{symbol[:6]}"
    elif market == MarketType.SHENZHEN:
        return f"{MARKET_SHENZHEN}{symbol[:6]}"
    else:  # 默认上海
        return f"{MARKET_SHANGHAI}{symbol[:6]}"


def get_market(symbol: str) -> str:
    symbol = symbol.strip()
    # 检查前缀
    if starts_with(symbol, MARKET_FLAGS):
        return symbol[:2].lower()
    # 检查后缀（如.SZ）
    if ends_with(symbol, MARKET_FLAGS):
        return symbol[-2:].lower()
    # 详细匹配规则
    if starts_with(symbol, ["50", "51", "60", "68", "90", "110", "113", "132", "204"]):
        return "sh"
    elif starts_with(symbol, ["00", "12", "13", "18", "15", "16", "20", "30", "39", "115", "1318"]):
        return "sz"
    elif starts_with(symbol, ["5", "6", "9", "7"]):
        return "sh"
    elif starts_with(symbol, ["88"]):
        return "sh"
    elif starts_with(symbol, ["4", "8"]):
        return "bj"
    return "sh"  # 默认上海


def get_market_id(symbol: str) -> MarketType:
    market = get_market(symbol)
    if market == "sh":
        return MarketType.SHANGHAI
    elif market == "sz":
        return MarketType.SHENZHEN
    elif market == "bj":
        return MarketType.BEIJING
    elif market == "hk":
        return MarketType.HONGKONG
    elif market == "us":
        return MarketType.USA
    return MarketType.SHANGHAI


def detect_market(symbol: str) -> (MarketType, str, str):
    code = symbol.strip()
    market = "sh"
    # 处理前缀（如sh600000）
    if starts_with(code, MARKET_FLAGS):
        market = code[:2].lower()
        code = code[2:].lstrip('.')  # 兼容带点的格式
    # 处理后缀（如600000.SH）
    elif ends_with(code, MARKET_FLAGS):
        market = code[-2:].lower()
        code = code[:-3].rstrip('.')  # 移除后缀
    # 详细匹配规则
    elif starts_with(code, ["50", "51", "60", "68", "90", "110", "113", "132", "204"]):
        market = "sh"
    elif starts_with(code, ["00", "12", "13", "18", "15", "16", "20", "30", "39", "115", "1318"]):
        market = "sz"
    elif starts_with(code, ["5", "6", "9", "7"]):
        market = "sh"
    elif starts_with(code, ["88"]):
        market = "sh"
    elif starts_with(code, ["4", "8"]):
        market = "bj"
    # 确定MarketType
    market_id = get_market_id(market)
    return market_id, market, code


# 判断函数系列
def assert_index_by_market_and_code(market_id: MarketType, symbol: str) -> bool:
    symbol = symbol.strip()
    if market_id == MarketType.SHANGHAI and starts_with(symbol, ["000", "880", "881"]):
        return True
    elif market_id == MarketType.SHENZHEN and starts_with(symbol, ["399"]):
        return True
    return False


def assert_index_by_security_code(security_code: str) -> bool:
    market_id, _, code = detect_market(security_code)
    return assert_index_by_market_and_code(market_id, code)


def assert_etf_by_market_and_code(market_id: MarketType, symbol: str) -> bool:
    symbol = symbol.strip()
    return market_id == MarketType.SHANGHAI and starts_with(symbol, ["510"])


def assert_stock_by_market_and_code(market_id: MarketType, symbol: str) -> bool:
    symbol = symbol.strip()
    if market_id == MarketType.SHANGHAI and starts_with(symbol, ["60", "68", "510"]):
        return True
    elif market_id == MarketType.SHENZHEN and starts_with(symbol, ["00", "30"]):
        return True
    return False


# 其他工具函数
def correct_security_code(security_code: str) -> str:
    if not security_code:
        return ""
    _, market, code = detect_market(security_code)
    return f"{market}{code}"


class TargetKind(Enum):
    STOCK = 0
    INDEX = 1
    BLOCK = 2
    ETF = 3


def assert_code(security_code: str) -> TargetKind:
    market_id, market, code = detect_market(security_code)
    # 板块指数
    if market_id == MarketType.SHANGHAI and starts_with(code, ["880", "881"]):
        return TargetKind.BLOCK
    # 指数
    if market_id == MarketType.SHANGHAI and starts_with(code, ["000"]):
        return TargetKind.INDEX
    if market_id == MarketType.SHENZHEN and starts_with(code, ["399"]):
        return TargetKind.INDEX
    # ETF
    if market_id == MarketType.SHANGHAI and starts_with(code, ["510"]):
        return TargetKind.ETF
    return TargetKind.STOCK


exchange_start_time = '09:15:00'
exchange_end_time = '15:00:00'
# time_range = "09:15:00~11:30:00,13:00:00~15:00:00"
trade_session = timestamp.TimeRange(f'{exchange_start_time}~{exchange_end_time}')


@lru_cache(maxsize=None)
def __calendar() -> pandas.Series:
    """
    交易日历
    """
    fn = os.path.join(base.config.meta_path, "calendar")
    df = pandas.read_csv(fn)
    return df['date']


def get_today() -> str:
    """
    获取当前日期
    """
    date = time.strftime(timestamp.FORMAT_ONLY_DATE)
    return date


def is_session_pre() -> bool:
    """
    是否盘前
    """
    now = time.strftime(timestamp.FORMAT_ONLY_TIME)
    return now < exchange_start_time


def is_session_reg() -> bool:
    """
    是否盘中
    """
    now = time.strftime(timestamp.FORMAT_ONLY_TIME)
    return trade_session.is_trading(now)


def is_session_post() -> bool:
    """
    是否盘后
    """
    now = time.strftime(timestamp.FORMAT_ONLY_TIME)
    return now > exchange_end_time


def last_trade_date() -> str:
    """
    获取最近的一个交易日
    """
    list = __calendar()
    today = get_today()
    session_pre = trade_session.is_session_pre()
    idx = list.searchsorted(today)
    date = list[idx]
    # 如果在今日之后, 那么今天是节假日, 则取前一个交易日, 如果在盘前, 今天的数据没生成, 则取前一个交易日
    if date > today or (date == today and session_pre):
        date = list[idx - 1]
    return date


if __name__ == '__main__':
    # 获取市场代码
    print(get_market("600519"))  # 输出: sh
    print(get_market("SZ000001"))  # 输出: sz

    # 修正证券代码
    print(correct_security_code("600519.SH"))  # 输出: sh600519

    # 判断标的类型
    print(assert_code("sh000001"))  # TargetKind.INDEX
    print(assert_code("sz399001"))  # TargetKind.INDEX
    print(assert_code("sh510500"))  # TargetKind.ETF

    a1 = last_trade_date()
    print(type(a1))
    print(a1)
    d1 = last_trade_date()
    print(d1)
