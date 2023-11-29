# -*- coding: UTF-8 -*-
import os
import sys

# 项目package顶层命名空间
namespace = "quant1x"


def project_path(filename: str = '') -> str:
    if len(filename) == 0:
        filename = __file__
    pos = filename.rfind(namespace)
    project_path = filename[:pos - 1]
    root_path = os.path.abspath(project_path)
    return root_path


def redirect(filename: str = ''):
    """
    重定向于当前工程目录
    :return:
    """
    root_path = project_path(filename)
    sys.path.insert(0, root_path)


if __name__ == '__main__':
    sys.exit(redirect())
