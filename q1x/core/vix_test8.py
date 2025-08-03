import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, date
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
    """
    获取指定标的和到期月的期权市场交易数据（包含价格）
    """
    try:
        print(f"💰 正在从 AkShare 获取 {symbol} {end_month} 价格数据...")
        df = ak.option_finance_board(symbol=symbol, end_month=end_month)
        if df is not None and not df.empty:
            # 🔺 关键：重命名列，使其与风险数据的列名一致
            df.rename(columns={'合约交易代码': 'CONTRACT_ID', '当前价': 'PRICE'}, inplace=True)
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
# 4. 提取 300ETF 期权数据 (使用真实剩余时间)
# -------------------------------
def extract_300etf_options(risk_df, price_df_dict, trade_date_str: str):
    """
    提取并合并风险与价格数据
    """
    # 1. 处理风险数据
    df_300 = risk_df[risk_df['CONTRACT_ID'].str.startswith('510300')].copy()
    if df_300.empty:
        print("❌ 未找到 300ETF 期权数据")
        return None

    # 🔺 关键修复：提取期权类型 (C/P)
    df_300['TYPE'] = df_300['CONTRACT_ID'].str.extract(r'([CP])')[0]

    # 🔺 关键修复：从合约代码中提取到期年月
    # 例如 510300C2508M04000 -> 提取 "2508"
    df_300['EXPIRE_YYMM'] = df_300['CONTRACT_ID'].str.extract(r'[CP](\d{4})')[0]

    # 过滤掉无法提取年月的行
    if df_300['EXPIRE_YYMM'].isnull().any():
        df_300 = df_300.dropna(subset=['EXPIRE_YYMM']).copy()
        print(f"⚠️ 过滤了无法提取年月的合约")

    # 🔺 关键修复：解析年月并计算真实到期日
    def calc_expire_date(yy_mm: str) -> date:
        try:
            year = 2000 + int(yy_mm[:2])  # "25" -> 2025
            month = int(yy_mm[2:4])       # "08" -> 8
            return get_fourth_wednesday(year, month)
        except Exception as e:
            print(f"❌ 计算到期日失败 {yy_mm}: {e}")
            return None

    df_300['EXPIRE_DATE_DT'] = df_300['EXPIRE_YYMM'].apply(calc_expire_date)
    if df_300['EXPIRE_DATE_DT'].isnull().any():
        df_300 = df_300.dropna(subset=['EXPIRE_DATE_DT']).copy()
        print(f"⚠️ 过滤了计算到期日失败的合约")

    # 🔺 关键修复：计算真实剩余时间 T (以年为单位)
    current_date = datetime.strptime(trade_date_str, "%Y%m%d").date()
    df_300['EXPIRE_DATE_DT'] = pd.to_datetime(df_300['EXPIRE_DATE_DT'])
    df_300['T_DAYS'] = (df_300['EXPIRE_DATE_DT'] - pd.to_datetime(current_date)).dt.days
    df_300['T_YEARS'] = df_300['T_DAYS'] / 365.0

    # 🔺 关键修复：提取行权价和IV
    df_300['STRIKE'] = df_300['CONTRACT_SYMBOL'].str.extract(r'(\d+(?:\.\d+)?)')[0].astype(float)
    iv_col = [col for col in df_300.columns if 'IMPLC' in col][0]
    df_300.rename(columns={iv_col: 'IMPLC_VOLATLTY'}, inplace=True)

    # 转换为数值型
    numeric_cols = ['DELTA_VALUE', 'THETA_VALUE', 'GAMMA_VALUE', 'VEGA_VALUE', 'RHO_VALUE', 'IMPLC_VOLATLTY']
    for col in numeric_cols:
        df_300[col] = pd.to_numeric(df_300[col], errors='coerce')
    df_300 = df_300[(df_300['IMPLC_VOLATLTY'] > 0.01) & (df_300['IMPLC_VOLATLTY'] < 1.0)]

    # 2. 合并价格数据
    prices = []
    for _, row in df_300.iterrows():
        contract_id = row['CONTRACT_ID']
        yymm = row['EXPIRE_YYMM'] # 现在 EXPIRE_YYMM 列已经存在

        if yymm not in price_df_dict:
            prices.append(np.nan)
            continue

        price_df = price_df_dict[yymm]
        price_row = price_df[price_df['CONTRACT_ID'] == contract_id]
        if price_row.empty:
            prices.append(np.nan)
        else:
            prices.append(price_row['PRICE'].iloc[0])

    df_300['PRICE'] = prices
    df_300 = df_300.dropna(subset=['PRICE']).copy()
    print(f"✅ 数据合并完成，最终有效合约: {len(df_300)} 条")
    return df_300

# -------------------------------
# 5. 合并风险与价格数据
# -------------------------------
def merge_risk_and_price(risk_df_300, price_df_dict):
    """
    将风险数据和价格数据合并。
    """
    # 创建一个空列表来存储价格
    prices = []

    # 遍历风险数据中的每一行
    for _, row in risk_df_300.iterrows():
        contract_id = row['CONTRACT_ID']
        yymm = row['EXPIRE_YYMM']

        # 检查该到期月是否有价格数据
        if yymm not in price_df_dict:
            print(f"❌ 无 {yymm} 月的价格数据")
            prices.append(np.nan)
            continue

        price_df = price_df_dict[yymm]

        # 在价格数据中查找该合约
        price_row = price_df[price_df['CONTRACT_ID'] == contract_id]
        if price_row.empty:
            print(f"❌ 未找到合约 {contract_id} 的价格")
            prices.append(np.nan)
        else:
            # 找到价格，添加到列表
            prices.append(price_row['PRICE'].iloc[0])

    # 🔺 关键修复3：将价格列表转换为Series，并与df_300的索引对齐
    # 这确保了长度匹配
    risk_df_300['PRICE'] = pd.Series(prices, index=risk_df_300.index)

    # 过滤掉价格缺失的合约
    df_300_merged = risk_df_300.dropna(subset=['PRICE']).copy()
    print(f"✅ 数据合并完成，最终有效合约: {len(df_300_merged)} 条")
    return df_300_merged

# -------------------------------
# 6. 计算“恐慌指数”（类VIX）(使用真实价格)
# -------------------------------
def calculate_vix_like(df_300):
    if df_300 is None or df_300.empty:
        raise ValueError("输入数据为空")

    def calculate_var(group):
        group = group.sort_values('STRIKE')
        strikes = group['STRIKE'].values
        T = group['T_YEARS'].iloc[0]

        # 计算ΔK
        delta_K = np.zeros_like(strikes)
        delta_K[0] = strikes[1] - strikes[0]
        delta_K[-1] = strikes[-1] - strikes[-2]
        if len(strikes) > 2:
            delta_K[1:-1] = (strikes[2:] - strikes[:-2]) / 2

        # 计算远期价F
        call_mask = (group['TYPE'] == 'C')
        put_mask = (group['TYPE'] == 'P')
        if not (call_mask.any() and put_mask.any()):
            raise ValueError("必须同时存在Call/Put合约")

        atm_call = group[call_mask].iloc[np.abs(group[call_mask]['DELTA_VALUE'] - 0.5).argmin()]
        atm_put = group[put_mask].iloc[np.abs(group[put_mask]['DELTA_VALUE'] + 0.5).argmin()]

        # 使用真实市场成交价计算F
        F = atm_call['STRIKE'] + np.exp(RISK_FREE_RATE * T) * (atm_call['PRICE'] - atm_put['PRICE'])
        if F <= 0:
            raise ValueError(f"计算出的远期价F无效: {F}")

        # 确定K0
        K0_candidates = strikes[strikes <= F]
        if len(K0_candidates) == 0:
            K0 = strikes[0]
        else:
            K0 = K0_candidates[-1]

        # 计算方差
        variance = 0
        for i, K in enumerate(strikes):
            # 在 calculate_var 函数中
            if K < F:
                Q = group[group['STRIKE'] == K]['MARKET_PRICE'].iloc[0]  # 使用真实市场价格
            else:
                Q = group[group['STRIKE'] == K]['MARKET_PRICE'].iloc[0]  # 使用真实市场价格

            variance += (delta_K[i] / (K**2)) * np.exp(RISK_FREE_RATE * T) * Q

        variance = (2 / T) * variance - (1 / T) * ((F / K0) - 1)**2

        if variance < 0:
            print(f"⚠️ 计算出的方差为负，调整为0.0001。F={F}, K0={K0}, T={T}")
            variance = 0.0001

        return {'var': variance, 'T': T, 'days': T * 365}

    try:
        groups = df_300.groupby('EXPIRE_DATE_DT')
        if len(groups) < 2:
            raise ValueError("需要至少两个到期日")

        exp_dates = sorted(groups.groups.keys())
        near_group = groups.get_group(exp_dates[0])
        next_group = groups.get_group(exp_dates[1])

        var1 = calculate_var(near_group)
        var2 = calculate_var(next_group)

        NT1, NT2 = var1['days'], var2['days']
        w = (NT2 - 30) / (NT2 - NT1)
        vix_squared = (var1['T'] * var1['var'] * w + var2['T'] * var2['var'] * (1 - w)) * (365 / 30)

        if vix_squared < 0:
            print(f"⚠️ 插值后的方差为负，调整为0.0001。vix_squared={vix_squared}")
            vix_squared = 0.0001

        vix = 100 * np.sqrt(vix_squared)

        if not 5 <= vix <= 80:
            raise ValueError(f"异常VIX值: {vix}")
        return round(vix, 2)

    except Exception as e:
        print(f"⚠️ 方差互换计算失败: {e}")
        try:
            atm_call = df_300[df_300['TYPE'] == 'C'].iloc[np.abs(df_300[df_300['TYPE'] == 'C']['DELTA_VALUE'] - 0.5).argmin()]
            atm_put = df_300[df_300['TYPE'] == 'P'].iloc[np.abs(df_300[df_300['TYPE'] == 'P']['DELTA_VALUE'] + 0.5).argmin()]
            iv_atm = (atm_call['IMPLC_VOLATLTY'] + atm_put['IMPLC_VOLATLTY']) / 2
            return round(iv_atm * 100, 2)
        except:
            return round(df_300['IMPLC_VOLATLTY'].mean() * 100, 2)

# -------------------------------
# 7. 主函数
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
    df_300 = extract_300etf_options(risk_df, price_df_dict, trade_date)
    if df_300 is None or df_300.empty:
        print("❌ 数据合并后为空，程序终止。")
        return

    # 7. ... (后续的IV平均、对比分析、风险敞口、保存等逻辑保持不变)
    # 为简洁省略，与之前相同

    print("\n🎉 全部完成！")

if __name__ == "__main__":
    main()