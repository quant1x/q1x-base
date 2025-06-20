# -*- coding: UTF-8 -*-
# 缓存的数据
import json
import os
from functools import lru_cache

import pandas as pd

from q1x.base import exchange
from q1x.base import base


@lru_cache(maxsize=None)
def securities() -> pd.DataFrame:
    """
    证券列表
    """
    full_path = os.path.join(base.quant1x_data_meta, 'securities.csv')
    if not os.path.isfile(full_path):
        return pd.DataFrame()
    df = pd.read_csv(full_path)
    # 转换为小写
    df.columns = df.columns.str.lower()
    return df[['code', 'name']]


@lru_cache(maxsize=None)
def block_list():
    """
    板块列表
    """
    df = securities()
    return df[df['code'].astype(str).str.startswith('sh880', 'sh881')]


def stock_name(code: str) -> str:
    corrected_symbol = exchange.correct_security_code(code)
    df = securities()
    tmp = df[df['code'] == corrected_symbol]
    name = tmp['name'].iloc[0]
    return name


def klines(code: str) -> pd.DataFrame | None:
    """
    获取缓存的日线数据
    """
    corrected_symbol = exchange.correct_security_code(code)
    suffix_length = 3  # 修正拼写并明确表示后缀长度
    symbol_directory = os.path.join(base.quant1x_data_day, corrected_symbol[:-suffix_length])  # 更清晰表达目录用途
    file_extension = '.csv'
    filename = f"{corrected_symbol}{file_extension}"  # 使用f-string格式化
    full_path = os.path.join(symbol_directory, filename)

    if os.path.isfile(full_path):
        return pd.read_csv(full_path)
    return None


# SectorFilename 板块缓存文件名
def sector_filename(date: str = '') -> str:
    """
    板块缓存文件名
    """
    name = 'blocks'
    cache_date = date.strip()
    if len(cache_date) == 0:
        cache_date = exchange.last_trade_date()
    filename = os.path.join(base.quant1x_data_meta, f'{name}.{cache_date}')
    return filename


@lru_cache(maxsize=None)
def get_sector_list() -> pd.DataFrame:
    """
    获取板块列表
    """
    sfn = sector_filename()
    df = pd.read_csv(sfn)
    df['code'] = 'sh' + df['code'].astype(str)
    return df


def get_sector_constituents(code: str) -> list[str]:
    """
    获取板块成分股列表
    """
    code = code.strip()
    security_code = exchange.correct_security_code(code)
    df = get_sector_list().copy()
    cs = df[df['code'] == security_code]['ConstituentStocks']
    list = []
    if cs.empty:
        return list
    cs1 = cs.iloc[0]
    ConstituentStocks = json.loads(cs1)
    list = []
    for sc in ConstituentStocks:
        sc = sc.strip()
        sc = exchange.correct_security_code(sc)
        list.append(sc)
    return list


if __name__ == '__main__':
    print(base.get_quant1x_config_filename())
    print('basedir', base.basedir)
    print('quant1x_data_day', base.quant1x_data_day)
    code = '600600'
    df = klines(code)
    print(df)
    stock_name = stock_name(code)
    print(stock_name)
    security_list = securities()
    print(security_list)
    index_list = block_list()
    print(index_list)
    sfn = sector_filename()
    df = pd.read_csv(sfn)
    print(df)
    print(df['code'].dtype)
    df['code'] = 'sh' + df['code'].astype(str)
    s1 = df[df['code'] == 'sh881478']
    print(s1)

    l1 = get_sector_constituents('880675')
    print(l1)
    print(type(l1))
