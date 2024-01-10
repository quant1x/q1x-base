# -*- coding: UTF-8 -*-
import os.path
import sys


def application() -> tuple[str, str, str]:
    """
    获取应用程序的名字
    :return: 目录, 文件名, 扩展名
    """
    app = sys.argv[0]
    # print(app)
    # print(os.path.abspath(app))
    abspath = os.path.abspath(app)
    # print(os.path.realpath(app))
    dirname, filename = os.path.split(abspath)
    (filename, ext) = os.path.splitext(filename)
    # print(filename, ext)
    return dirname, filename, ext


if __name__ == '__main__':
    name = application()
    print(name)
