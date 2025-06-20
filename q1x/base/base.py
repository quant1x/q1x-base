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
