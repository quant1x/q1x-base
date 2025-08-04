# -*- coding: UTF-8 -*-
import os

import yaml

from q1x.base import file

def get_quant1x_config_filename() -> str:
    """
    获取quant1x.yaml文件路径
    :return:
    """
    # 默认配置文件名
    default_config_filename = 'quant1x.yaml'
    yaml_filename = os.path.join('~', 'runtime', 'etc', default_config_filename)
    user_home = file.homedir()
    if not os.path.isfile(yaml_filename):
        quant1x_root = os.path.join(user_home, '.quant1x')
        yaml_filename = os.path.join(quant1x_root, default_config_filename)
        yaml_filename = os.path.expanduser(yaml_filename)
    yaml_filename = os.path.expanduser(yaml_filename)
    return yaml_filename


# 安全加载YAML配置
def load_config(file_path: str) -> dict:
    """安全加载YAML配置"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
            return config
    except FileNotFoundError:
        raise ValueError(f"配置文件 {file_path} 不存在")
    except yaml.YAMLError as e:
        raise ValueError(f"YAML格式错误: {str(e)}")
    except Exception as e:
        raise ValueError(f"加载配置失败: {str(e)}")

class Quant1XConfig:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """初始化配置"""
        self.__home_path = file.homedir()
        self.__config_filename = get_quant1x_config_filename()
        self.__config = load_config(self.__config_filename)

        # 初始化路径
        self.__default_main_path = os.path.join(self.__home_path, '.quant1x')

        self.meta_path = os.path.join(self.__default_main_path, 'meta')
        """str: 元数据路径"""

        self.data_path = self.__config.get('basedir', '').strip()
        """str: 数据目录 """

        if not self.data_path:
            self.data_path = os.path.join(self.__default_main_path, 'data')
        self.data_path= os.path.expanduser(self.data_path)
        # 数据路径
        self.kline_path = os.path.join(self.data_path, 'day')
        """str: K线路径 """

# 创建配置单例
config = Quant1XConfig()