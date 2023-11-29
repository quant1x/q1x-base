# -*- coding: UTF-8 -*-

__author__ = 'WangFeng'

from .base import Singleton
from .base import TimeRange, TIME_FORMAT_TIMESTAMP
from .base import TradingSession
from .device import cpu_num, max_procs
from .network import ip_is_private, ip_is_secure, lan_address
from .path import project_path, redirect
from .version import project_version
