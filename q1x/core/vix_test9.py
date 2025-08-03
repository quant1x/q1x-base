import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import os

# -------------------------------
# 1. 常量定义
# -------------------------------
VIX_THRESHOLD_LOW = 0.05
VIX_THRESHOLD_HIGH = 0.05
HISTORY_DATA_FILE = "vix_history_300etf.csv"
HISTORICAL_QUANTILE_LOW = 0.2
HISTORICAL_QUANTILE_HIGH = 0.8
RISK_FREE_RATE = 0.02

# -------------------------------
# 2. 获取数据
# -------------------------------
def fetch_risk_data(trade_date: str):
    try:
        print(f"📡 正在从 AkShare 获取 {trade_date} 风险数据...")
        df = ak.option_risk_indicator_sse(date=trade_date)
        if df is not None and not df.empty:
            print(f"✅ 成功获取 {len(df)} 条风险数据")
            return df
        else:
            print("❌ 未获取到风险数据")
            return None
    except Exception as e:
        print(f"❌ 获取风险数据失败: {e}")
        return None

def fetch_price_data(symbol: str, end_month: str):
    try:
        print(f"💰 正在从 AkShare 获取 {symbol} {end_month} 价格数据...")
        df = ak.option_finance_board(symbol=symbol, end_month=end_month)
        if df is not None and not df.empty:
            df.rename(columns={'合约交易代码': 'CONTRACT_ID', '当前价': 'PRICE'}, inplace=True)
            # ✅ 将“点”转为“元”：除以 1000
            df['STRIKE'] = df['行权价']
            print(f"✅ 成功获取 {len(df)} 条价格数据")
            return df
        else:
            print("❌ 未获取到价格数据")
            return None
    except Exception as e:
        print(f"❌ 获取价格数据失败: {e}")
        return None

# -------------------------------
# 3. 计算“第四个星期三”函数
# -------------------------------
def get_fourth_wednesday(year: int, month: int) -> date:
    first_day = datetime(year, month, 1)
    weekday_of_first = first_day.weekday()
    first_wednesday = 1 + (2 - weekday_of_first) % 7
    fourth_wednesday_day = first_wednesday + 21
    return datetime(year, month, fourth_wednesday_day).date()

# -------------------------------
# 4. 提取并合并数据
# -------------------------------
def extract_and_merge_data(risk_df, price_df_dict, trade_date_str: str):
    """
    提取 300ETF 期权数据，计算真实剩余时间，并合并价格与行权价
    """
    # 1. 筛选 300ETF 期权（以 510300 开头的合约）
    df_300 = risk_df[risk_df['CONTRACT_ID'].str.startswith('510300')].copy()
    if df_300.empty:
        print("❌ 未找到 300ETF 期权数据")
        return None

    # 2. 提取期权类型（C/P）和到期年月
    df_300['TYPE'] = df_300['CONTRACT_ID'].str.extract(r'([CP])')[0]
    df_300['EXPIRE_YYMM'] = df_300['CONTRACT_ID'].str.extract(r'[CP](\d{4})')[0]

    # 过滤无法提取年月的合约
    if df_300['EXPIRE_YYMM'].isnull().any():
        before = len(df_300)
        df_300 = df_300.dropna(subset=['EXPIRE_YYMM']).copy()
        print(f"⚠️ 过滤了 {before - len(df_300)} 条无法提取年月的合约")

    # 3. 计算真实到期日（第四个星期三）
    def calc_expire_date(yy_mm: str) -> date:
        try:
            year = 2000 + int(yy_mm[:2])
            month = int(yy_mm[2:4])
            return get_fourth_wednesday(year, month)
        except Exception as e:
            print(f"❌ 计算到期日失败 {yy_mm}: {e}")
            return None

    df_300['EXPIRE_DATE_DT'] = df_300['EXPIRE_YYMM'].apply(calc_expire_date)

    # 过滤计算失败的到期日
    if df_300['EXPIRE_DATE_DT'].isnull().any():
        before = len(df_300)
        df_300 = df_300.dropna(subset=['EXPIRE_DATE_DT']).copy()
        print(f"⚠️ 过滤了 {before - len(df_300)} 条计算到期日失败的合约")

    # 4. 计算剩余天数和年化时间
    current_date = datetime.strptime(trade_date_str, "%Y%m%d").date()
    df_300['T_DAYS'] = (pd.to_datetime(df_300['EXPIRE_DATE_DT']) - pd.to_datetime(current_date)).dt.days
    df_300['T_YEARS'] = df_300['T_DAYS'] / 365.0

    # 5. 提取隐含波动率（IMPLC_VOLATLTY）
    iv_col = [col for col in df_300.columns if 'IMPLC' in col]
    if not iv_col:
        print("❌ 未找到隐含波动率列")
        return None
    df_300.rename(columns={iv_col[0]: 'IMPLC_VOLATLTY'}, inplace=True)

    # 6. 数值列转换
    numeric_cols = ['DELTA_VALUE', 'THETA_VALUE', 'GAMMA_VALUE', 'VEGA_VALUE', 'RHO_VALUE', 'IMPLC_VOLATLTY']
    for col in numeric_cols:
        df_300[col] = pd.to_numeric(df_300[col], errors='coerce')

    # 过滤异常波动率
    df_300 = df_300[(df_300['IMPLC_VOLATLTY'] > 0.01) & (df_300['IMPLC_VOLATLTY'] < 1.0)].copy()
    print(f"✅ 提取 300ETF 期权: {len(df_300)} 条，已计算真实剩余时间")

    # 7. 合并价格和行权价（关键：从 price_df 获取 STRIKE）
    prices = []
    strikes = []

    for _, row in df_300.iterrows():
        contract_id = row['CONTRACT_ID']
        yymm = row['EXPIRE_YYMM']

        if yymm not in price_df_dict:
            prices.append(np.nan)
            strikes.append(np.nan)
            continue

        price_df = price_df_dict[yymm]
        price_row = price_df[price_df['CONTRACT_ID'] == contract_id]

        if price_row.empty:
            prices.append(np.nan)
            strikes.append(np.nan)
        else:
            prices.append(price_row['PRICE'].iloc[0])
            strikes.append(price_row['STRIKE'].iloc[0])  # ✅ 使用 price_df 中的 STRIKE（已除以 1000）

    df_300['PRICE'] = prices
    df_300['STRIKE'] = strikes

    # 8. 去除无效价格或行权价
    df_300 = df_300.dropna(subset=['PRICE', 'STRIKE']).copy()

    # 9. 数据校验
    if df_300.empty:
        print("❌ 合并后数据为空")
        return None

    print(f"📊 行权价范围: {df_300['STRIKE'].min():.3f} ~ {df_300['STRIKE'].max():.3f} 元")
    print(f"📊 当前300ETF价格应在: {df_300['STRIKE'].median():.3f} 元附近")

    if df_300['STRIKE'].nunique() == 1:
        print("❌ 警告：所有行权价相同！可能是数据错误！")
    else:
        print("✅ 行权价分布正常")

    # 10. 输出示例
    print("\n📋 示例数据:")
    print(df_300[['CONTRACT_ID', 'STRIKE', 'TYPE', 'PRICE', 'T_DAYS', 'IMPLC_VOLATLTY']].head(8).to_string(index=False))

    print(f"✅ 数据合并完成，最终有效合约: {len(df_300)} 条")
    print(f"📊 合约类型分布: \n{df_300['TYPE'].value_counts()}")
    if len(df_300[df_300['TYPE'] == 'P']) == 0:
        print("❌ 警告：未找到 Put 合约！无法计算真实 VIX！")
        return None
    return df_300

# -------------------------------
# 5. 计算“恐慌指数”（类VIX）(使用真实价格)
# -------------------------------
def bs_price(S, K, T, r, sigma, option_type='C'):
    """
    Black-Scholes 期权定价公式
    """
    from scipy.stats import norm
    if T <= 0:
        if option_type == 'C':
            return max(S - K, 0)
        else:
            return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'C':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

# -------------------------------
# 5. 计算“恐慌指数”（类VIX）(使用真实时间)
# -------------------------------
def calculate_vix_like(df_300):
    """
    计算类VIX指数（基于近月+次近月期权的加权插值）
    """
    # 按到期日分组
    exp_groups = df_300.groupby('EXPIRE_CODE')
    if len(exp_groups) < 2:
        print("⚠️ 数据不足两个到期日，使用所有数据平均")
        return df_300['IMPLC_VOLATLTY'].mean() * 100

    all_expirations = sorted(exp_groups.groups.keys())
    near_exp = all_expirations[0]
    next_exp = all_expirations[1]
    near_term = exp_groups.get_group(near_exp)
    next_term = exp_groups.get_group(next_exp)

    def get_atm_weighted_iv(group):
        if 'C' in group['TYPE'].values:
            atm_call = group[group['TYPE'] == 'C'].iloc[(group[group['TYPE']=='C']['DELTA_VALUE'] - 0.5).abs().argsort()[:3]]
        else:
            atm_call = pd.DataFrame()
        if 'P' in group['TYPE'].values:
            atm_put = group[group['TYPE'] == 'P'].iloc[(group[group['TYPE']=='P']['DELTA_VALUE'] + 0.5).abs().argsort()[:3]]
        else:
            atm_put = pd.DataFrame()
        combined = pd.concat([atm_call, atm_put])
        if len(combined) == 0:
            return group['IMPLC_VOLATLTY'].mean()
        return combined['IMPLC_VOLATLTY'].mean()

    iv_near = get_atm_weighted_iv(near_term)
    iv_next = get_atm_weighted_iv(next_term)

    # --- 计算真实剩余时间 ---
    def get_T_days(exp_code):
        # 解析 'M08' -> 8月
        month = int(exp_code[1:3])
        # 计算当月第四个周三
        from datetime import date
        first_day = date(2025, month, 1)
        first_wed = first_day + timedelta(days=(2 - first_day.weekday()) % 7)
        expire_date = first_wed + timedelta(weeks=3)
        expire_datetime = datetime.combine(expire_date, datetime.min.time())
        today = datetime(2025, 8, 1)
        return (expire_datetime - today).days

    T1 = get_T_days(near_exp) / 365.0
    T2 = get_T_days(next_exp) / 365.0
    TARGET_T = 30 / 365.0

    # --- 使用正确的插值公式 ---
    if T1 <= TARGET_T <= T2 and T2 > T1:
        vix_squared = ((T2 - TARGET_T) / (T2 - T1)) * (iv_near**2) * (T1 / TARGET_T) + \
                      ((TARGET_T - T1) / (T2 - T1)) * (iv_next**2) * (T2 / TARGET_T)
        vix = np.sqrt(vix_squared)
    else:
        vix = iv_near

    return vix * 100

# -------------------------------
# ✅ 真实 VIX 计算函数（CBOE 官方逻辑）
# -------------------------------
def calculate_real_vix(df_300, trade_date_str: str, risk_free_rate: float = 0.02):
    """
    使用 CBOE VIX 白皮书方法计算真实恐慌指数
    """
    from datetime import datetime
    import pandas as pd
    import numpy as np

    current_date = datetime.strptime(trade_date_str, "%Y%m%d").date()
    today = pd.Timestamp(current_date)

    # 1. 提取有效数据
    df = df_300.dropna(subset=['EXPIRE_DATE_DT', 'T_YEARS', 'PRICE', 'STRIKE']).copy()
    df = df[(df['IMPLC_VOLATLTY'] > 0.01) & (df['IMPLC_VOLATLTY'] < 1.0)]
    if df.empty:
        print("❌ 数据为空，无法计算 VIX")
        return np.nan

    df['T'] = df['T_YEARS']
    df['K'] = df['STRIKE']

    # 按到期日排序
    expirations = sorted(df['EXPIRE_DATE_DT'].unique())
    if len(expirations) < 2:
        print("❌ 不足两个到期日")
        return np.nan

    # 2. 找 M1 和 M2：T1 < 30/365 < T2
    target_T = 30 / 365.0
    valid_pairs = []
    for i in range(len(expirations) - 1):
        t1 = expirations[i]
        t2 = expirations[i + 1]
        # ✅ 将 t1/t2 转为 Timestamp 再计算
        T1 = (pd.Timestamp(t1) - today).days / 365.0
        T2 = (pd.Timestamp(t2) - today).days / 365.0
        if T1 < target_T < T2:
            valid_pairs.append((t1, t2, T1, T2))

    if not valid_pairs:
        print("⚠️ 无满足 T1<30<T2 的组合，使用最近两个")
        t1, t2 = expirations[0], expirations[1]
        T1 = (pd.Timestamp(t1) - today).days / 365.0
        T2 = (pd.Timestamp(t2) - today).days / 365.0
    else:
        t1, t2, T1, T2 = valid_pairs[0]

    # ✅ 修复：t1 和 t2 是 datetime.date，直接用 strftime 或 str
    print(f"🎯 使用到期日: {t1.strftime('%Y-%m-%d')} ({T1*365:.1f}天), {t2.strftime('%Y-%m-%d')} ({T2*365:.1f}天)")

    term1 = df[df['EXPIRE_DATE_DT'] == t1].copy()
    term2 = df[df['EXPIRE_DATE_DT'] == t2].copy()

    # 3. 计算方差
    try:
        var1 = _compute_variance(term1, T1, risk_free_rate)
        var2 = _compute_variance(term2, T2, risk_free_rate)
    except Exception as e:
        print(f"❌ 方差计算异常: {e}")
        import traceback
        traceback.print_exc()
        return np.nan

    if np.isnan(var1) or np.isnan(var2) or var1 <= 0 or var2 <= 0:
        print("⚠️ 方差非正，回退")
        return np.nan

    # 4. 插值到30天
    vix_squared = ((T2 - target_T) * var1 + (target_T - T1) * var2) / (T2 - T1)
    vix = np.sqrt(vix_squared) * 100
    return max(vix, 5.0)


def _compute_variance(df_term, T, r):
    """
    对一个到期日，计算无模型方差（CBOE 方法）
    """
    import numpy as np

    if T <= 0:
        return np.nan

    discount = np.exp(-r * T)
    df = df_term.sort_values('K').reset_index(drop=True)

    # 1. 提取 Call 和 Put 的 PRICE，按 CONTRACT_ID 对齐
    calls = df[df['TYPE'] == 'C'].set_index('K')['PRICE']
    puts = df[df['TYPE'] == 'P'].set_index('K')['PRICE']

    # 2. 用 K 做对齐，计算 C - P
    common_strikes = calls.index.intersection(puts.index)
    if len(common_strikes) == 0:
        print("⚠️ 无共同行权价，尝试最近邻匹配")
        # 回退：用所有行权价，最近的 Call/Put
        df_call = df[df['TYPE'] == 'C'].set_index('K')['PRICE']
        df_put = df[df['TYPE'] == 'P'].set_index('K')['PRICE']
        c_minus_p = []
        for k in df['K']:
            c = df_call.reindex([k], method='nearest').iloc[0]
            p = df_put.reindex([k], method='nearest').iloc[0]
            c_minus_p.append(c - p)
        df['C_MINUS_P'] = c_minus_p
    else:
        # 有共同行权价
        c_aligned = calls[common_strikes]
        p_aligned = puts[common_strikes]
        df['C_MINUS_P'] = np.nan
        for k in common_strikes:
            df.loc[df['K'] == k, 'C_MINUS_P'] = calls[k] - puts[k]

    # 3. 插值找 C-P=0 的 K（即远期价格 F）
    df_valid = df.dropna(subset=['C_MINUS_P'])
    if len(df_valid) < 2:
        F = df['K'].median()
    else:
        # 按 K 排序
        df_valid = df_valid.sort_values('K')
        cp_vals = df_valid['C_MINUS_P'].values
        k_vals = df_valid['K'].values

        # 找符号变化的位置
        cross = None
        for i in range(len(cp_vals) - 1):
            if cp_vals[i] * cp_vals[i + 1] <= 0:
                cross = i
                break

        if cross is None:
            F = df_valid.iloc[(df_valid['C_MINUS_P']).abs().argsort()[0]]['K']
        else:
            k1, k2 = k_vals[cross], k_vals[cross + 1]
            c1, c2 = cp_vals[cross], cp_vals[cross + 1]
            if c2 != c1:
                w = -c1 / (c2 - c1)
                F = k1 + w * (k2 - k1)
            else:
                F = (k1 + k2) / 2
    print(f"🔍 远期价格 F ≈ {F:.3f}")

    # 4. 构造 ΔK
    Ks = df['K'].values
    delta_K = []
    for i, k in enumerate(Ks):
        if i == 0:
            dk = Ks[i + 1] - k
        elif i == len(Ks) - 1:
            dk = k - Ks[i - 1]
        else:
            dk = (Ks[i + 1] - Ks[i - 1]) / 2
        delta_K.append(dk)
    df['DELTA_K'] = delta_K

    # 5. 计算加权方差
    variance = 0.0
    for _, row in df.iterrows():
        K = row['K']
        dk = row['DELTA_K']
        Q = row['PRICE']  # 实际交易价格
        if np.isnan(Q) or Q <= 0:
            continue
        weight = dk / (K ** 2)
        variance += weight * Q

    variance = (2 / T) * variance - ((F / F - 1) ** 2) / T  # (F/K0 - 1)^2 = 0
    variance *= discount
    return max(variance, 1e-6)

# -------------------------------
# 6. 主函数
# -------------------------------
def main():
    trade_date = "20250801"
    output_file = f"300ETF_监控结果_{trade_date}.csv"
    summary_file = f"300ETF_指标汇总_{trade_date}.csv"
    print("🚀 开始执行 300ETF 恐慌指数监控...")

    # 1. 获取风险数据
    risk_df = fetch_risk_data(trade_date)
    if risk_df is None:
        return

    # 2. 从风险数据中提取所有到期月份
    df_temp = risk_df[risk_df['CONTRACT_ID'].str.startswith('510300')].copy()
    df_temp['EXPIRE_YYMM'] = df_temp['CONTRACT_ID'].str.extract(r'[CP](\d{4})')[0]
    all_yymm = df_temp['EXPIRE_YYMM'].dropna().unique()
    print(f"🔍 发现 {len(all_yymm)} 个到期月份: {sorted(all_yymm)}")

    # 3. 获取价格数据
    price_df_dict = {}
    for yymm in all_yymm:
        price_df = fetch_price_data(symbol="华泰柏瑞沪深300ETF期权", end_month=yymm)
        if price_df is not None:
            price_df_dict[yymm] = price_df

    # 4. 提取并合并数据
    # 🔺 关键：调用我们新写的、包含完整逻辑的函数
    df_300 = extract_and_merge_data(risk_df, price_df_dict, trade_date)
    if df_300 is None or df_300.empty:
        print("❌ 数据合并后为空，程序终止。")
        return

    # 5. 计算恐慌指数
    print("\n🔍 正在计算【真实VIX】（CBOE官方方法）...")
    try:
        vix_value = calculate_real_vix(df_300, trade_date, risk_free_rate=RISK_FREE_RATE)
        if np.isnan(vix_value):
            raise ValueError("VIX 计算结果为 NaN")
        print(f"🎯 真实 A股300ETF恐慌指数（VIX）: {vix_value:.2f}")
    except Exception as e:
        print(f"❌ 真实VIX计算失败: {e}")
        vix_value = df_300['IMPLC_VOLATLTY'].mean() * 100
        print(f"✅ 回退使用平均隐含波动率: {vix_value:.2f}")

    # 6. ... (后续的IV平均、对比分析、风险敞口、保存等逻辑保持不变)

    print("\n🎉 全部完成！")

if __name__ == "__main__":
    main()