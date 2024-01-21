# -*- coding: UTF-8 -*-
import os


def mkdirs(path: str):
    """
    创建目录
    :param path:
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)


def touch(filename: str):
    """
    创建一个空文件
    :param filename:
    :return:
    """
    directory = os.path.dirname(filename)
    mkdirs(directory)
    with open(filename, 'w') as done_file:
        pass


def homedir() -> str:
    """
    获取宿主目录
    首先会获取环境变量GOX_HOME, 如果不存在则用~
    :return:
    """
    gox_home = os.getenv("GOX_HOME", '')
    gox_home = gox_home.strip()
    if len(gox_home) == 0:
        user_home = os.path.expanduser('~')
    else:
        user_home = gox_home
    return user_home

