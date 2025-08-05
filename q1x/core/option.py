#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project : q1x-base
@Package : core
@File    : option.py
@Author  : wangfeng
@Date    : 2025/8/4 5:57
@Desc    : 期权数据
"""
from io import BytesIO

import pandas as pd
import requests
from pandas import DataFrame

# 中国金融期货交易所

CFFEX_OPTION_URL_300 = "http://www.cffex.com.cn/quote_IO.txt"

# 深圳证券交易所

SZ_OPTION_URL_300 = "http://www.szse.cn/api/report/ShowReport?SHOWTYPE=xlsx&CATALOGID=ysplbrb&TABKEY=tab1&random=0.10432465776720479"

# 上海证券交易所

SH_OPTION_URL_50 = "http://yunhq.sse.com.cn:32041/v1/sh1/list/self/510050"
SH_OPTION_URL_KING_50 = "http://yunhq.sse.com.cn:32041/v1/sho/list/tstyle/510050_{}"

SH_OPTION_URL_300 = "http://yunhq.sse.com.cn:32041/v1/sh1/list/self/510300"
SH_OPTION_URL_KING_300 = "http://yunhq.sse.com.cn:32041/v1/sho/list/tstyle/510300_{}"

SH_OPTION_URL_500 = "http://yunhq.sse.com.cn:32041/v1/sh1/list/self/510500"
SH_OPTION_URL_KING_500 = "http://yunhq.sse.com.cn:32041/v1/sho/list/tstyle/510500_{}"

SH_OPTION_URL_KC_50 = "http://yunhq.sse.com.cn:32041/v1/sh1/list/self/588000"
SH_OPTION_URL_KC_KING_50 = "http://yunhq.sse.com.cn:32041/v1/sho/list/tstyle/588000_{}"

SH_OPTION_URL_KC_50_YFD = "http://yunhq.sse.com.cn:32041/v1/sh1/list/self/588080"
SH_OPTION_URL_KING_50_YFD = "http://yunhq.sse.com.cn:32041/v1/sho/list/tstyle/588080_{}"

SH_OPTION_PAYLOAD = {
    "select": "select: code,name,last,change,chg_rate,amp_rate,volume,amount,prev_close"
}

SH_OPTION_PAYLOAD_OTHER = {"select": "contractid,last,chg_rate,presetpx,exepx"}


def option_finance_board(
        symbol: str = "嘉实沪深300ETF期权", end_month: str = "2306"
) -> DataFrame:
    """
    期权当前交易日的行情数据
    主要为三个: 华夏上证50ETF期权, 华泰柏瑞沪深300ETF期权, 嘉实沪深300ETF期权,
    沪深300股指期权, 中证1000股指期权, 上证50股指期权, 华夏科创50ETF期权, 易方达科创50ETF期权
    http://www.sse.com.cn/assortment/options/price/
    http://www.szse.cn/market/product/option/index.html
    http://www.cffex.com.cn/hs300gzqq/
    http://www.cffex.com.cn/zz1000gzqq/
    :param symbol: choice of {"华夏上证50ETF期权", "华泰柏瑞沪深300ETF期权", "南方中证500ETF期权",
    "华夏科创50ETF期权", "易方达科创50ETF期权", "嘉实沪深300ETF期权", "沪深300股指期权", "中证1000股指期权", "上证50股指期权"}
    :type symbol: str
    :param end_month: 2003; 2020 年 3 月到期的期权
    :type end_month: str
    :return: 当日行情
    :rtype: pandas.DataFrame
    """
    end_month = end_month[-2:]
    if symbol == "华夏上证50ETF期权":
        r = requests.get(
            SH_OPTION_URL_KING_50.format(end_month),
            params=SH_OPTION_PAYLOAD_OTHER,
        )
        data_json = r.json()
        raw_data = pd.DataFrame(data_json["list"])
        raw_data.index = [str(data_json["date"]) + str(data_json["time"])] * data_json[
            "total"
        ]
        raw_data.columns = ["合约交易代码", "当前价", "涨跌幅", "前结价", "行权价"]
        raw_data["数量"] = [data_json["total"]] * data_json["total"]
        raw_data.reset_index(inplace=True)
        raw_data.columns = [
            "日期",
            "合约交易代码",
            "当前价",
            "涨跌幅",
            "前结价",
            "行权价",
            "数量",
        ]
        return raw_data
    elif symbol == "华泰柏瑞沪深300ETF期权":
        r = requests.get(
            SH_OPTION_URL_KING_300.format(end_month),
            params=SH_OPTION_PAYLOAD_OTHER,
        )
        data_json = r.json()
        raw_data = pd.DataFrame(data_json["list"])
        raw_data.index = [str(data_json["date"]) + str(data_json["time"])] * data_json[
            "total"
        ]
        raw_data.columns = ["合约交易代码", "当前价", "涨跌幅", "前结价", "行权价"]
        raw_data["数量"] = [data_json["total"]] * data_json["total"]
        raw_data.reset_index(inplace=True)
        raw_data.columns = [
            "日期",
            "合约交易代码",
            "当前价",
            "涨跌幅",
            "前结价",
            "行权价",
            "数量",
        ]
        return raw_data
    elif symbol == "南方中证500ETF期权":
        r = requests.get(
            SH_OPTION_URL_KING_500.format(end_month),
            params=SH_OPTION_PAYLOAD_OTHER,
        )
        data_json = r.json()
        raw_data = pd.DataFrame(data_json["list"])
        raw_data.index = [str(data_json["date"]) + str(data_json["time"])] * data_json[
            "total"
        ]
        raw_data.columns = ["合约交易代码", "当前价", "涨跌幅", "前结价", "行权价"]
        raw_data["数量"] = [data_json["total"]] * data_json["total"]
        raw_data.reset_index(inplace=True)
        raw_data.columns = [
            "日期",
            "合约交易代码",
            "当前价",
            "涨跌幅",
            "前结价",
            "行权价",
            "数量",
        ]
        return raw_data
    elif symbol == "华夏科创50ETF期权":
        r = requests.get(
            SH_OPTION_URL_KC_KING_50.format(end_month),
            params=SH_OPTION_PAYLOAD_OTHER,
        )
        data_json = r.json()
        raw_data = pd.DataFrame(data_json["list"])
        raw_data.index = [str(data_json["date"]) + str(data_json["time"])] * data_json[
            "total"
        ]
        raw_data.columns = ["合约交易代码", "当前价", "涨跌幅", "前结价", "行权价"]
        raw_data["数量"] = [data_json["total"]] * data_json["total"]
        raw_data.reset_index(inplace=True)
        raw_data.columns = [
            "日期",
            "合约交易代码",
            "当前价",
            "涨跌幅",
            "前结价",
            "行权价",
            "数量",
        ]
        return raw_data
    elif symbol == "易方达科创50ETF期权":
        r = requests.get(
            SH_OPTION_URL_KING_50_YFD.format(end_month),
            params=SH_OPTION_PAYLOAD_OTHER,
        )
        data_json = r.json()
        raw_data = pd.DataFrame(data_json["list"])
        raw_data.index = [str(data_json["date"]) + str(data_json["time"])] * data_json[
            "total"
        ]
        raw_data.columns = ["合约交易代码", "当前价", "涨跌幅", "前结价", "行权价"]
        raw_data["数量"] = [data_json["total"]] * data_json["total"]
        raw_data.reset_index(inplace=True)
        raw_data.columns = [
            "日期",
            "合约交易代码",
            "当前价",
            "涨跌幅",
            "前结价",
            "行权价",
            "数量",
        ]
        return raw_data
    elif symbol == "嘉实沪深300ETF期权":
        url = "http://www.szse.cn/api/report/ShowReport/data"
        params = {
            "SHOWTYPE": "JSON",
            "CATALOGID": "ysplbrb",
            "TABKEY": "tab1",
            "PAGENO": "1",
            "random": "0.10642298535346595",
        }
        r = requests.get(url, params=params)
        data_json = r.json()
        page_num = data_json[0]["metadata"]["pagecount"]
        big_df = pd.DataFrame()
        for page in range(1, page_num + 1):
            params = {
                "SHOWTYPE": "JSON",
                "CATALOGID": "ysplbrb",
                "TABKEY": "tab1",
                "PAGENO": page,
                "random": "0.10642298535346595",
            }
            r = requests.get(url, params=params)
            data_json = r.json()
            temp_df = pd.DataFrame(data_json[0]["data"])
            big_df = pd.concat([big_df, temp_df], ignore_index=True)

        big_df.columns = [
            "合约编码",
            "合约简称",
            "标的名称",
            "类型",
            "行权价",
            "合约单位",
            "期权行权日",
            "行权交收日",
        ]
        big_df["期权行权日"] = pd.to_datetime(big_df["期权行权日"])
        big_df["end_month"] = big_df["期权行权日"].dt.month.astype(str).str.zfill(2)
        big_df = big_df[big_df["end_month"] == end_month]
        del big_df["end_month"]
        big_df.reset_index(inplace=True, drop=True)
        return big_df
    elif symbol == "沪深300股指期权":
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36"
        }
        r = requests.get(CFFEX_OPTION_URL_300, headers=headers)
        raw_df = pd.read_table(BytesIO(r.content), sep=",")
        raw_df["end_month"] = (
            raw_df["instrument"]
            .str.split("-", expand=True)
            .iloc[:, 0]
            .str.slice(
                4,
            )
        )
        raw_df = raw_df[raw_df["end_month"] == end_month]
        del raw_df["end_month"]
        raw_df.reset_index(inplace=True, drop=True)
        return raw_df
    elif symbol == "中证1000股指期权":
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36"
        }
        url = "http://www.cffex.com.cn/quote_MO.txt"
        r = requests.get(url, headers=headers)
        raw_df = pd.read_table(BytesIO(r.content), sep=",")
        raw_df["end_month"] = (
            raw_df["instrument"]
            .str.split("-", expand=True)
            .iloc[:, 0]
            .str.slice(
                4,
            )
        )
        raw_df = raw_df[raw_df["end_month"] == end_month]
        del raw_df["end_month"]
        raw_df.reset_index(inplace=True, drop=True)
        return raw_df
    elif symbol == "上证50股指期权":
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36"
        }
        url = "http://www.cffex.com.cn/quote_HO.txt"
        r = requests.get(url, headers=headers)
        raw_df = pd.read_table(BytesIO(r.content), sep=",")
        raw_df["end_month"] = (
            raw_df["instrument"]
            .str.split("-", expand=True)
            .iloc[:, 0]
            .str.slice(
                4,
            )
        )
        raw_df = raw_df[raw_df["end_month"] == end_month]
        del raw_df["end_month"]
        raw_df.reset_index(inplace=True, drop=True)
        return raw_df
    else:
        return DataFrame()


def option_risk_indicator_sse(date: str = "20240626") -> pd.DataFrame:
    """
    上海证券交易所-产品-股票期权-期权风险指标
    http://www.sse.com.cn/assortment/options/risk/
    :param date: 日期; 20150209 开始
    :type date: str
    :return: 期权风险指标
    :rtype: pandas.DataFrame
    """
    url = "http://query.sse.com.cn/commonQuery.do"
    params = {
        "isPagination": "false",
        "trade_date": date,
        "sqlId": "SSE_ZQPZ_YSP_GGQQZSXT_YSHQ_QQFXZB_DATE_L",
        "contractSymbol": "",
    }
    headers = {
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Host": "query.sse.com.cn",
        "Pragma": "no-cache",
        "Referer": "http://www.sse.com.cn/",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/101.0.4951.67 Safari/537.36",
    }
    r = requests.get(url, params=params, headers=headers)
    data_json = r.json()
    temp_df = pd.DataFrame(data_json["result"])
    temp_df = temp_df[
        [
            "TRADE_DATE",
            "SECURITY_ID",
            "CONTRACT_ID",
            "CONTRACT_SYMBOL",
            "DELTA_VALUE",
            "THETA_VALUE",
            "GAMMA_VALUE",
            "VEGA_VALUE",
            "RHO_VALUE",
            "IMPLC_VOLATLTY",
        ]
    ]
    temp_df["TRADE_DATE"] = pd.to_datetime(
        temp_df["TRADE_DATE"], errors="coerce"
    ).dt.date
    temp_df["DELTA_VALUE"] = pd.to_numeric(temp_df["DELTA_VALUE"], errors="coerce")
    temp_df["THETA_VALUE"] = pd.to_numeric(temp_df["THETA_VALUE"], errors="coerce")
    temp_df["GAMMA_VALUE"] = pd.to_numeric(temp_df["GAMMA_VALUE"], errors="coerce")
    temp_df["VEGA_VALUE"] = pd.to_numeric(temp_df["VEGA_VALUE"], errors="coerce")
    temp_df["RHO_VALUE"] = pd.to_numeric(temp_df["RHO_VALUE"], errors="coerce")
    temp_df["IMPLC_VOLATLTY"] = pd.to_numeric(
        temp_df["IMPLC_VOLATLTY"], errors="coerce"
    )
    return temp_df