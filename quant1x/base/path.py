# -*- coding: UTF-8 -*-
import os
import sys

# 项目package顶层命名空间
namespace = "quant1x"


def project_path() -> str:
    pos = __file__.rfind(namespace)
    project_path = __file__[:pos - 1]
    root_path = os.path.abspath(project_path)
    return root_path


def redirect():
    """
    重定向于当前工程目录
    :return:
    """
    root_path = project_path()
    sys.path.insert(0, root_path)


if __name__ == '__main__':
    sys.exit(redirect())
