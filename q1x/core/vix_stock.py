import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from q1x.base import cache

# -------------------------------
# 1. 配置参数
# -------------------------------
target_period = 'd'      # 周期：日线
target_tail = 0          # 尾部数据量
code = 'sh603488'        # 股票代码
name = cache.stock_name(code)
print(f'{name}({code})')

# -------------------------------
# 2. 数据加载与预处理
# -------------------------------
klines = cache.klines(code)
klines = cache.convert_klines_trading(klines, period=target_period)

if target_tail > 0 and len(klines) >= target_tail:
    klines = klines.tail(target_tail).copy()  # 使用 copy() 避免 SettingWithCopyWarning

klines['date'] = pd.to_datetime(klines['date'])
klines['x_pos'] = np.arange(len(klines))

# 确保数值类型并处理可能的异常值
for col in ['open', 'high', 'low', 'close']:  # 注意：Garman-Klass 需要开盘价
    if col not in klines.columns:
        print(f"❌ 数据缺少必要列: {col}")
        # 如果没有开盘价，可能需要从其他地方获取或用其他方式估算
        # 这里为了演示，假设 open 等于 close
        klines['open'] = klines['close']
    klines[col] = pd.to_numeric(klines[col], errors='coerce')
    # 处理 0 或负值，避免 log(0) 错误
    klines[col] = klines[col].clip(lower=1e-10)

# -------------------------------
# 3. 计算个股“伪VIX” (使用 Garman-Klass 波动率)
# -------------------------------
window = 20  # 20日滚动窗口

# 计算 ln(High / Low) 和 ln(Close / Open)
klines['ln_HL'] = np.log(klines['high'] / klines['low'])
klines['ln_CO'] = np.log(klines['close'] / klines['open'])

# Garman-Klass 方差 (日度)
# 公式: 0.5 * ln(H/L)^2 - (2*ln(2)-1) * ln(C/O)^2
# 其中 2*ln(2)-1 ≈ 0.386294
klines['gk_var_daily'] = 0.5 * klines['ln_HL']**2 - 0.386294 * klines['ln_CO']**2

# 滚动均值（相当于方差）
rolling_gk_var = klines['gk_var_daily'].rolling(window=window, min_periods=1).mean()

# 年化波动率（标准差）
klines['HV_GarmanKlass'] = np.sqrt(rolling_gk_var) * np.sqrt(252)  # 年化

# 转为百分比（类似 VIX）
klines['Pseudo_VIX'] = klines['HV_GarmanKlass'] * 100

# ✅ 使用 bfill() 替代 fillna(method='bfill')
klines['Pseudo_VIX'] = klines['Pseudo_VIX'].bfill()  # 向后填充（用后面的值填前面）

# -------------------------------
# 4. 计算历史分位数 (5% 和 95%)
# -------------------------------
# 计算长期窗口，用于分位数和置信区间
window_long = min(252, len(klines))  # 如果数据不足1年，用全部数据
if window_long < 20:
    print("📊 数据不足，无法计算历史分位数")
else:
    # 计算滚动的历史分位数 (5% 和 95%)
    klines['HV_quantile_5'] = klines['Pseudo_VIX'].rolling(window=window_long, min_periods=20).apply(
        lambda x: (pd.Series(x).rank(pct=True).iloc[-1] <= 0.05) if len(x) > 1 else np.nan
    )
    klines['HV_quantile_95'] = klines['Pseudo_VIX'].rolling(window=window_long, min_periods=20).apply(
        lambda x: (pd.Series(x).rank(pct=True).iloc[-1] >= 0.95) if len(x) > 1 else np.nan
    )

    # 计算滚动均值和标准差，用于置信区间
    klines['HV_mean'] = klines['Pseudo_VIX'].rolling(window=window_long, min_periods=20).mean()
    klines['HV_std'] = klines['Pseudo_VIX'].rolling(window=window_long, min_periods=20).std()
    # 95% 置信区间 (±1.96个标准差)
    klines['CI_upper'] = klines['HV_mean'] + 1.96 * klines['HV_std']
    klines['CI_lower'] = klines['HV_mean'] - 1.96 * klines['HV_std']

# -------------------------------
# 5. 综合评估与警示
# -------------------------------
latest_vix = klines['Pseudo_VIX'].iloc[-1]
# 获取最新的分位数状态
current_quantile_5 = klines['HV_quantile_5'].iloc[-1] if 'HV_quantile_5' in klines.columns else False
current_quantile_95 = klines['HV_quantile_95'].iloc[-1] if 'HV_quantile_95' in klines.columns else False

# 划分区间 (基于5%和95%)
if current_quantile_5:
    status = "🟢 极低波动"
    suggestion = "波动率处于历史极低位，可考虑买入布局，等待波动放大。"
elif current_quantile_95:
    status = "🔴 极高波动"
    suggestion = "波动率处于历史极高位，市场恐慌，注意风险控制或对冲。警惕波动率快速回落。"
else:
    status = "🟡 正常波动"
    suggestion = "波动率处于正常范围，按常规策略操作。"

# -------------------------------
# 6. 可视化 (增强版)
# -------------------------------
fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

# 主图：波动率
ax1 = axes[0]
ax1.plot(klines['date'], klines['Pseudo_VIX'], label="Pseudo-VIX (Garman-Klass Vol)", color="red", linewidth=2)
ax1.axhline(y=klines['Pseudo_VIX'].mean(), color="blue", linestyle="--", label="长期均值")

# 如果计算了置信区间，则绘制
if 'CI_upper' in klines.columns and 'CI_lower' in klines.columns:
    ax1.fill_between(klines['date'], klines['CI_lower'], klines['CI_upper'],
                     color="gray", alpha=0.2, label="95% 置信区间")
    ax1.plot(klines['date'], klines['CI_upper'], color="gray", linestyle="--", alpha=0.7)
    ax1.plot(klines['date'], klines['CI_lower'], color="gray", linestyle="--", alpha=0.7)

# 标记当前值
ax1.axhline(y=latest_vix, color="red", linestyle=":", alpha=0.7, label=f"当前: {latest_vix:.2f}")

ax1.set_title(f"{name}({code}) - 个股波动率指数 (Pseudo-VIX)", fontsize=14, fontweight='bold')
ax1.set_ylabel("波动率 (%)", fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# 副图：历史分位数状态
ax2 = axes[1]
# 将布尔值转换为数值进行绘图
quantile_5_numeric = klines['HV_quantile_5'].astype(float).fillna(0)
quantile_95_numeric = klines['HV_quantile_95'].astype(float).fillna(0)

ax2.fill_between(klines['date'], 0, 1, where=quantile_5_numeric, color='green', alpha=0.3, label='低于 5%')
ax2.fill_between(klines['date'], 0, 1, where=quantile_95_numeric, color='red', alpha=0.3, label='高于 95%')

ax2.set_ylabel("分位状态", fontsize=12)
ax2.set_xlabel("日期", fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 1)
ax2.set_yticks([0.25, 0.75])
ax2.set_yticklabels(['低位', '高位'])

plt.tight_layout()
plt.show()

# -------------------------------
# 7. 输出结果
# -------------------------------
print(f"股票代码: {code}")
print(f"股票名称: {name}")
print(f"数据周期: {target_period}")
print("-" * 50)
print(f"最新 Pseudo-VIX: {latest_vix:.2f}%")
if 'HV_mean' in klines.columns and 'HV_std' in klines.columns:
    mean_vix = klines['HV_mean'].iloc[-1]
    std_vix = klines['HV_std'].iloc[-1]
    ci_lower = klines['CI_lower'].iloc[-1]
    ci_upper = klines['CI_upper'].iloc[-1]
    print(f"历史均值: {mean_vix:.2f}%")
    print(f"标准差: {std_vix:.2f}%")
    print(f"95% 置信区间: [{ci_lower:.2f}%, {ci_upper:.2f}%]")
print("-" * 50)
print(f"📊 当前状态: {status}")
print(f"💡 建议: {suggestion}")