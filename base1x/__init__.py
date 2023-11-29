# -*- coding: UTF-8 -*-

__author__ = 'WangFeng'

from .datetime import TimeRange, FORMAT_DATETIME
from .datetime import TradingSession
from .device import cpu_num, max_procs
from .devp import project_path, redirect
from .network import ip_is_private, ip_is_secure, lan_address
from .pattern import Singleton
from .version import project_version
