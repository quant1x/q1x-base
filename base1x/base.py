# -*- coding: UTF-8 -*-
import re
import threading
import time
from dataclasses import dataclass
from typing import List

TIME_FORMAT_TIMESTAMP = '%H:%M:%S'


class Singleton:
    """
    线程安全的单例模式
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if not cls._instance:
                # cls._instance = super().__new__(cls, *args, **kwargs)
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        pass


@dataclass
class TimeRange(object):
    """
    时间范围
    """
    begin: str
    end: str

    def __init__(self, time_range: str):
        """
        构造
        :param time_range: 
        :return: 
        """
        self.begin = ''
        self.end = ''

        time_range = time_range.strip()
        list = re.split(r"[~-]\s*", time_range)
        if len(list) != 2:
            raise RuntimeError("非法的时间格式")
        # TODO：时间格式校验
        # 时间排序
        self.begin = list[0].strip()
        self.end = list[1].strip()
        if self.begin > self.end:
            self.begin, self.end = self.end, self.begin

    def is_trading(self, timestamp: str = "") -> bool:
        """
        是否交易中
        :param timestamp:
        :return:
        """
        timestamp = timestamp.strip()
        if len(timestamp) == 0:
            timestamp = time.strftime(TIME_FORMAT_TIMESTAMP)
        if self.begin <= timestamp <= self.end:
            return True
        return False

    def is_valid(self) -> bool:
        """
        时段是否有效
        :return:
        """
        return self.begin != '' and self.end != ''


@dataclass
class TradingSession:
    """
    交易时段
    """
    sessions: List

    def __init__(self, time_range: str):
        """
        构造
        :param time_range:
        """
        self.sessions = []
        time_range = time_range.strip()
        list = re.split(r",\s*", time_range)
        for v in list:
            v = v.strip()
            r = TimeRange(v)
            self.sessions.append(r)

    # def __str__(self):
    #     sb = []
    #     for v in self.sessions:
    #         sb.append(v)
    #     return sb.__str__()

    def is_trading(self, timestamp: str = "") -> bool:
        """
        是否交易中
        :param timestamp:
        :return:
        """
        timestamp = timestamp.strip()
        if len(timestamp) == 0:
            timestamp = time.strftime(TIME_FORMAT_TIMESTAMP)
        for item in self.sessions:
            v = item
            if v.is_trading(timestamp):
                return True
        return False

    def is_valid(self) -> bool:
        """
        时段是否有效
        :return:
        """
        for item in self.sessions:
            if not item.is_valid():
                return False
        return True


if __name__ == '__main__':
    time_range = " 09:30:00 ~ 14:56:30 "
    time_range = " 14:56:30 - 09:30:00 "
    tr = TimeRange(time_range)
    print(tr)

    time_range = "11:30:00 ~ 09:15:00 , 15:00:00 - 13:00:00 "
    time_range = "15:00:00 - 13:00:00 "

    tr = TradingSession(time_range)
    print(tr)
    print(tr.is_trading('09:30:00'))
