# -*- coding: UTF-8 -*-
# 缓存的数据
import os

import pandas as pd
import yaml
from functools import lru_cache

from base1x import exchange
from base1x import file


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

# quant1x在宿主目录的路径
quant1x_main_path = os.path.expanduser('~')
if quant1x_main_path == quant1x_home:
    quant1x_main_path = os.path.join(quant1x_main_path, '.quant1x')
# 元数据路径
quant1x_data_meta = os.path.join(quant1x_main_path, 'meta')
# 读取basedir配置
basedir = config['basedir'].strip()
# 如果basedir配置无效, 则默认使用{$quant1x_home}/data
if len(basedir) == 0:
    basedir = os.path.join(quant1x_home, 'data')

# 日线数据路径
quant1x_data_day = os.path.join(basedir, 'day')

@lru_cache(maxsize=None)
def securities() -> pd.DataFrame:
    """
    证券列表
    """
    full_path = os.path.join(quant1x_data_meta, 'securities.csv')
    if not os.path.isfile(full_path):
        return pd.DataFrame()
    df = pd.read_csv(full_path)
    # 转换为小写
    df.columns = df.columns.str.lower()
    return df[['code','name']]

@lru_cache(maxsize=None)
def block_list():
    """
    板块列表
    """
    df = securities()
    return df[df['code'].astype(str).str.startswith('sh880','sh881')]


def stock_name(code: str) -> str:
    corrected_symbol = exchange.correct_security_code(code)
    df = securities()
    tmp = df[df['code']==corrected_symbol]
    name = tmp['name'].iloc[0]
    return name


def klines(code: str) -> pd.DataFrame | None:
    """
    获取缓存的日线数据
    """
    corrected_symbol = exchange.correct_security_code(code)
    suffix_length = 3  # 修正拼写并明确表示后缀长度
    symbol_directory = os.path.join(quant1x_data_day, corrected_symbol[:-suffix_length])  # 更清晰表达目录用途
    file_extension = '.csv'
    filename = f"{corrected_symbol}{file_extension}"  # 使用f-string格式化
    full_path = os.path.join(symbol_directory, filename)

    if os.path.isfile(full_path):
        return pd.read_csv(full_path)
    return None

if __name__ == '__main__':
    print(get_quant1x_config_filename())
    print('basedir', basedir)
    print('quant1x_data_day', quant1x_data_day)
    code = '600600'
    df = klines(code)
    print(df)
    stock_name = stock_name(code)
    print(stock_name)
    security_list = securities()
    print(security_list)
    index_list = block_list()
    print(index_list)

