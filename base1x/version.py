# -*- coding: UTF-8 -*-
import sys

from git import Repo


def project_version(incr: bool = False) -> str:
    """
    获取项目版本号, 取自git的tag
    这个函数不能在发行后运行, 只能用于打包时自动检测最新的tag作为版本号
    :param incr:
    :return:
    """
    # path = project_path()
    path = r'./'
    repo = Repo(path)
    tags = []
    for __tag in repo.tags:
        tag = str(__tag)
        tag = tag[1:]
        tags.append(tag)
    tags.sort(key=lambda x: tuple(int(v) for v in x.split('.')))
    if len(tags) == 0:
        return "0.0.0"
    latest = tags[-1]
    if not incr:
        return latest
    # print(latest)
    last_vs = latest.split('.')
    last_vs[-1] = str(int(last_vs[-1]) + 1)
    new_version = '.'.join(last_vs)
    return new_version


if __name__ == '__main__':
    v1 = project_version()
    print(v1)
    v2 = project_version(True)
    print(v2)
    sys.exit(project_version())
