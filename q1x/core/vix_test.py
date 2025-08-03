#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project : q1x-base
@Package : 
@File    : vix_test.py
@Author  : wangfeng
@Date    : 2025/8/1 9:10
@Desc    : 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from datetime import datetime

# -------------------------------
# 1. 解析 risk.csv（多合约拼接在一行）
# -------------------------------
def parse_risk_csv(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        line = f.read()

    # 正则匹配每条记录
    pattern = r'(\d{4}-\d{2}-\d{2})\s*,\s*(\d+)\s*,\s*([A-Z0-9]+)\s*,\s*([^,]+?)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)'
    matches = re.findall(pattern, line)

    data = []
    for m in matches:
        data.append({
            'TRADE_DATE': m[0],
            'SECURITY_ID': m[1],
            'CONTRACT_SYMBOL': m[2],
            'CN_NAME': m[3],
            'DELTA_VALUE': float(m[4]),
            'GAMMA_VALUE': float(m[5]),
            'THETA_VALUE': float(m[6]),
            'VEGA_VALUE': float(m[7]),
            'RHO_VALUE': float(m[8])
        })

    df = pd.DataFrame(data)
    df['TRADE_DATE'] = pd.to_datetime(df['TRADE_DATE'])
    return df

# -------------------------------
# 2. 提取 300ETF 期权数据
# -------------------------------
def get_300etf_options(df):
    df_300 = df[df['CONTRACT_SYMBOL'].str.startswith('510300')].copy()

    # 提取到期月份和类型
    df_300['EXPIRE_TYPE'] = df_300['CONTRACT_SYMBOL'].str.extract(r'M(\d{4})|A(\d{4})')[0].fillna(
        df_300['CONTRACT_SYMBOL'].str.extract(r'M(\d{3})|A(\d{3})')[0])
    df_300['TYPE'] = df_300['CONTRACT_SYMBOL'].str.extract(r'([CP])')[0]
    df_300['STRIKE'] = df_300['CN_NAME'].str.extract(r'(\d+\.?\d*)元?')[0].astype(float)

    return df_300

# -------------------------------
# 3. 计算“恐慌指数”（类VIX）——基于 Vega 和 Delta 的加权波动预期
# -------------------------------
def calculate_panic_index(df_300):
    """
    使用近月期权的 ATM 附近 Vega 和 Delta 构建“恐慌指数”
    虽无 IV，但 Vega 越高，说明市场对波动越敏感 → 可代理恐慌情绪
    """
    # 只取近月（假设 M04000 是最近到期）
    near_month = df_300[df_300['EXPIRE_TYPE'] == '04000']  # 请根据实际调整
    if near_month.empty:
        near_month = df_300.sort_values('EXPIRE_TYPE').groupby('TYPE').first().reset_index()

    # 找 ATM Call 和 Put（Delta 最接近 0.5 / -0.5）
    atm_call = near_month[near_month['TYPE'] == 'C'].iloc[(near_month[near_month['TYPE'] == 'C']['DELTA_VALUE'] - 0.5).abs().argsort()[:1]]
    atm_put = near_month[near_month['TYPE'] == 'P'].iloc[(near_month[near_month['TYPE'] == 'P']['DELTA_VALUE'] + 0.5).abs().argsort()[:1]]

    vega_call = atm_call['VEGA_VALUE'].values[0]
    vega_put = atm_put['VEGA_VALUE'].values[0]

    # 恐慌指数 = (Call Vega + Put Vega) / 2，放大为百分比
    panic_index = (vega_call + vega_put) / 2 * 100
    return panic_index, atm_call, atm_put

# -------------------------------
# 4. 计算组合风险（假设持仓为 1 张每合约）
# -------------------------------
def calculate_risk_exposure(df_300):
    df_300['position'] = 1  # 假设每合约持有1张，可替换为真实持仓

    df_300['net_delta'] = df_300['DELTA_VALUE'] * df_300['position']
    df_300['net_gamma'] = df_300['GAMMA_VALUE'] * df_300['position']
    df_300['net_vega'] = df_300['VEGA_VALUE'] * df_300['position']
    df_300['net_theta'] = df_300['THETA_VALUE'] * df_300['position']

    total_delta = df_300['net_delta'].sum()
    total_gamma = df_300['net_gamma'].sum()
    total_vega = df_300['net_vega'].sum()
    total_theta = df_300['net_theta'].sum()

    return {
        'Delta': total_delta,
        'Gamma': total_gamma,
        'Vega': total_vega,
        'Theta': total_theta
    }

# -------------------------------
# 5. 绘制波动率微笑（用 Vega 代理隐含波动率）
# -------------------------------
def plot_vol_smile(df_300):
    plt.figure(figsize=(10, 6))

    for exp, group in df_300.groupby('EXPIRE_TYPE'):
        calls = group[group['TYPE'] == 'C'].sort_values('STRIKE')
        puts = group[group['TYPE'] == 'P'].sort_values('STRIKE')

        plt.plot(calls['STRIKE'], calls['VEGA_VALUE'], 'bo-', label=f'Call Vega ({exp})', alpha=0.7, markersize=4)
        plt.plot(puts['STRIKE'], puts['VEGA_VALUE'], 'ro--', label=f'Put Vega ({exp})', alpha=0.7, markersize=4)

    plt.xlabel('行权价')
    plt.ylabel('Vega（代理隐含波动率）')
    plt.title('300ETF 波动率微笑（Vega 代理）')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# -------------------------------
# 6. 主函数：监控 300ETF
# -------------------------------
def monitor_300etf(file_path="risk.csv"):
    print("🔍 正在加载并解析 risk.csv...")
    df = parse_risk_csv(file_path)

    print("📊 提取 300ETF 期权数据...")
    df_300 = get_300etf_options(df)

    if df_300.empty:
        print("❌ 未找到 300ETF 期权数据")
        return

    print(f"✅ 成功加载 {len(df_300)} 条 300ETF 期权数据")

    # 计算恐慌指数
    panic_index, atm_call, atm_put = calculate_panic_index(df_300)
    print(f"\n🚨 A股恐慌指数（类VIX）估算值: {panic_index:.2f}")
    print(f"   ATM Call: {atm_call['CONTRACT_SYMBOL'].values[0]} (Delta={atm_call['DELTA_VALUE'].values[0]:.3f})")
    print(f"   ATM Put:  {atm_put['CONTRACT_SYMBOL'].values[0]} (Delta={atm_put['DELTA_VALUE'].values[0]:.3f})")

    # 计算风险敞口
    risk = calculate_risk_exposure(df_300)
    print(f"\n🛡️  组合风险敞口（假设每合约1张）:")
    for k, v in risk.items():
        print(f"   {k}: {v:.3f}")

    # 绘图
    print("\n📈 正在绘制波动率微笑曲线...")
    plot_vol_smile(df_300)

    # 保存处理后的数据（可选）
    df_300.to_csv("300etf_monitor_output.csv", index=False, encoding='utf-8-sig')
    print("💾 数据已保存至 300etf_monitor_output.csv")

# -------------------------------
# 7. 运行监控
# -------------------------------
if __name__ == "__main__":
    monitor_300etf("../../risk.csv")  # 请确保文件路径正确