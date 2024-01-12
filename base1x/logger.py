# -*- coding: UTF-8 -*-
from loguru import logger as __logger

from .application import application
splitted_app_path = application()
log_file = f"{splitted_app_path[0]}/{splitted_app_path[1]}.log"
__logger.add(log_file, rotation="00:00", retention="10 days")

logger = __logger
