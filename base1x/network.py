# -*- coding: UTF-8 -*-
import sys
from ipaddress import IPv4Address

UNSAFE_ADDRESS = '0.0.0.0'


def ip_is_private(ip: str = '127.0.0.1') -> bool:
    """
    判断IPv4地址是否内网
    :param ip:
    :return: True为内网, False为外网
    """
    ip = ip.strip()
    return IPv4Address(ip).is_private


def ip_is_secure(ip: str = UNSAFE_ADDRESS) -> bool:
    """
    判断IP地址是否安全
    :param ip:
    :return:
    """
    ip = ip.strip()
    if ip == UNSAFE_ADDRESS:
        return False
    else:
        return ip_is_private(ip)


if __name__ == '__main__':
    ip = "192.168.0.1"
    print(ip_is_secure(ip))
    ip = "114.114.114.114"
    print(ip_is_secure(ip))
    ip = "0.0.0.0"
    print(ip_is_secure(ip))
    ip = "127.0.0.1"
    print(ip_is_secure(ip))
    sys.exit(0)
