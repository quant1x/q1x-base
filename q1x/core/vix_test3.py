import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from q1x.base import cache

target_period = 'd'  # 周期
period_name = cache.get_period_name(target_period)
target_tail = 89  # 尾部多少条数据

code = 'sz000158'
name = cache.stock_name(code)
print(f'{name}({code})')

# 数据加载
klines = cache.klines(code)
klines = cache.convert_klines_trading(klines, period=target_period)
if target_tail > 0 and len(klines) >= target_tail:
    klines = klines.tail(target_tail)
klines['date'] = pd.to_datetime(klines['date'])
klines['x_pos'] = np.arange(len(klines))

# 确保数值类型
klines['high'] = klines['high'].astype(float)
klines['low'] = klines['low'].astype(float)
klines['close'] = klines['close'].astype(float)

# -------------------------------
# 使用 Parkinson 波动率（基于最高价和最低价）
# -------------------------------
window = 20  # 20日滚动窗口

# 计算 ln(High / Low)
klines['ln_HL'] = np.log(klines['high'] / klines['low'])
#klines['ln_HL'] = np.log(klines['close'] / klines['close'].shift(1))

# Parkinson 方差
klines['parkinson_var'] = (1 / (4 * np.log(2))) * (klines['ln_HL'] ** 2)

# 滚动均值（相当于方差）
rolling_var = klines['parkinson_var'].rolling(window).mean()

# 年化波动率（标准差）
klines['HV_Parkinson'] = np.sqrt(rolling_var) * np.sqrt(252)  # 年化

# 转为百分比（类似 VIX）
klines['Pseudo_VIX'] = klines['HV_Parkinson'] * 100

# ✅ 使用 bfill() 替代 fillna(method='bfill')
klines['Pseudo_VIX'] = klines['Pseudo_VIX'].bfill()  # 向后填充（用后面的值填前面）

# -------------------------------
# 可视化
# -------------------------------
plt.figure(figsize=(12, 6))
plt.plot(klines['date'], klines['Pseudo_VIX'], label="Pseudo-VIX (Parkinson Vol)", color="red")
plt.title(f"{name}({code}) - Parkinson Historical Volatility")
plt.xlabel("Date")
plt.ylabel("Volatility (%)")
plt.axhline(y=klines['Pseudo_VIX'].mean(), color="blue", linestyle="--", label="Mean Volatility")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# -------------------------------
# 输出最新波动率
# -------------------------------
latest_vix = klines['Pseudo_VIX'].iloc[-1]
print(f"Latest Pseudo-VIX (Parkinson): {latest_vix:.2f}")

# 假设 klines 是你已有的数据，包含 'Pseudo_VIX' 列
# 计算历史分位数（修正版）
window_long = min(252, len(klines))  # 如果数据不足1年，用全部数据
if window_long < 20:
    print("📊 数据不足，无法计算历史分位数")
klines['HV_quantile'] = klines['Pseudo_VIX'].rolling(window_long).apply(
    lambda x: pd.Series(x).rank().iloc[-1] / len(x) if len(x) > 1 else np.nan
)

# 获取最新值
latest_vix = klines['Pseudo_VIX'].iloc[-1]
current_quantile = klines['HV_quantile'].iloc[-1]

# 划分区间
if current_quantile < 0.3:
    status = "低位 / 低波动区"
    suggestion = "可考虑买入布局，等待波动放大"
elif current_quantile < 0.7:
    status = "正常波动区"
    suggestion = "市场平稳，按常规策略操作"
else:
    status = "高位 / 恐慌区"
    suggestion = "波动剧烈，注意风险控制或对冲"

print(f"Latest Pseudo-VIX: {latest_vix:.2f}")
print(f"历史分位数: {current_quantile:.1%}")
print(f"当前状态: {status}")
print(f"建议: {suggestion}")