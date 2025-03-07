# -*- coding: UTF-8 -*-
# 缓存的数据
import os

import pandas as pd
from pandas import DataFrame

from base1x import exchange
from base1x import file
import yaml

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
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise ValueError(f"配置文件 {file_path} 不存在")
    except yaml.YAMLError as e:
        raise ValueError(f"YAML格式错误: {str(e)}")

# 获取用户目录, GOX_HOME环境变量优先宿主目录
quant1x_home = file.homedir()
# 获取配置文件路径
config_filename = get_quant1x_config_filename()
# 加载配置文件
config = load_config(config_filename)

# 读取basedir配置
basedir = config['basedir'].strip()
# 如果basedir配置无效, 则默认使用{$quant1x_home}/data
if len(basedir) == 0:
    basedir = os.path.join(quant1x_home, 'data')

# 日线数据路径
quant1x_data_day = os.path.join(basedir, 'day')

def klines(code: str) -> DataFrame | None:
    """
    获取缓存的日线数据
    """
    symbol = exchange.correct_security_code(code)
    endlenth = 3
    filepath = os.path.join(quant1x_data_day, symbol[:-endlenth])
    ext = '.csv'
    name = symbol + ext
    filename = os.path.join(filepath, name)
    if os.path.isfile(filename):
        df = pd.read_csv(filename)
        return df
    else:
        return None

if __name__ == '__main__':
    print(get_quant1x_config_filename())
    print('basedir', basedir)
    print('quant1x_data_day', quant1x_data_day)
    df = klines('600600')
    print(df)
