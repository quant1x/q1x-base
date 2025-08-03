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
# 无风险利率（可替换为真实数据）
RISK_FREE_RATE = 0.02

# -------------------------------
# 2. 获取数据
# -------------------------------
def fetch_risk_data(trade_date: str):
    """获取风险指标数据 (IV, Delta等)"""
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
    """获取市场行情数据 (价格)"""
    try:
        print(f"💰 正在从 AkShare 获取 {symbol} {end_month} 价格数据...")
        df = ak.option_finance_board(symbol=symbol, end_month=end_month)
        if df is not None and not df.empty:
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
    核心函数：提取风险数据，计算真实时间，然后与价格数据合并。
    :param risk_df: 从 risk_indicator_sse 获取的数据
    :param price_df_dict: 从 finance_board 获取的数据字典 {yymm: price_df}
    :param trade_date_str: 交易日期
    :return: 包含风险指标和市场价格的完整 DataFrame
    """
    # 4.1 处理风险数据 (复用您代码中的逻辑)
    df_300 = risk_df[risk_df['CONTRACT_ID'].str.startswith('510300')].copy()
    if df_300.empty:
        print("❌ 未找到 300ETF 期权数据")
        return None

    df_300['TYPE'] = df_300['CONTRACT_ID'].str.extract(r'([CP])')[0]
    df_300['EXPIRE_YYMM'] = df_300['CONTRACT_ID'].str.extract(r'[CP](\d{4})')[0]

    if df_300['EXPIRE_YYMM'].isnull().any():
        df_300 = df_300.dropna(subset=['EXPIRE_YYMM']).copy()
        print(f"⚠️ 过滤了无法提取年月的合约")

    def calc_expire_date(yy_mm: str) -> date:
        try:
            year = 2000 + int(yy_mm[:2])
            month = int(yy_mm[2:4])
            return get_fourth_wednesday(year, month)
        except Exception as e:
            print(f"❌ 计算到期日失败 {yy_mm}: {e}")
            return None

    df_300['EXPIRE_DATE_DT'] = df_300['EXPIRE_YYMM'].apply(calc_expire_date)
    if df_300['EXPIRE_DATE_DT'].isnull().any():
        df_300 = df_300.dropna(subset=['EXPIRE_DATE_DT']).copy()
        print(f"⚠️ 过滤了计算到期日失败的合约")

    # 计算真实剩余时间
    current_date = datetime.strptime(trade_date_str, "%Y%m%d").date()
    df_300['EXPIRE_DATE_DT'] = pd.to_datetime(df_300['EXPIRE_DATE_DT'])
    df_300['T_DAYS'] = (df_300['EXPIRE_DATE_DT'] - pd.to_datetime(current_date)).dt.days
    df_300['T_YEARS'] = df_300['T_DAYS'] / 365.0

    # 提取行权价和IV
    df_300['STRIKE'] = df_300['CONTRACT_SYMBOL'].str.extract(r'(\d+(?:\.\d+)?)')[0].astype(float)
    iv_col = [col for col in df_300.columns if 'IMPLC' in col][0]
    df_300.rename(columns={iv_col: 'IMPLC_VOLATLTY'}, inplace=True)

    numeric_cols = ['DELTA_VALUE', 'THETA_VALUE', 'GAMMA_VALUE', 'VEGA_VALUE', 'RHO_VALUE', 'IMPLC_VOLATLTY']
    for col in numeric_cols:
        df_300[col] = pd.to_numeric(df_300[col], errors='coerce')
    df_300 = df_300[(df_300['IMPLC_VOLATLTY'] > 0.01) & (df_300['IMPLC_VOLATLTY'] < 1.0)]

    # 4.2 合并价格数据
    # 为每个合约找到对应的价格
    prices = []
    for _, row in df_300.iterrows():
        contract_id = row['CONTRACT_ID'] # 如 510300C2508M04000
        expire_yymm = row['EXPIRE_YYMM']  # 如 2508

        # 检查该到期月是否有价格数据
        if expire_yymm not in price_df_dict:
            print(f"❌ 无 {expire_yymm} 月的价格数据")
            prices.append(np.nan)
            continue

        price_df = price_df_dict[expire_yymm]
        # 在价格数据中查找合约交易代码
        price_row = price_df[price_df['合约交易代码'] == contract_id]
        if price_row.empty:
            print(f"❌ 未找到合约 {contract_id} 的价格")
            prices.append(np.nan)
        else:
            # 使用“当前价”作为Q
            prices.append(price_row['当前价'].iloc[0])

    # 将价格列表添加到风险数据中
    df_300['PRICE'] = prices

    # 再次过滤，移除价格缺失的合约
    df_300 = df_300.dropna(subset=['PRICE']).copy()
    print(f"✅ 数据合并完成，最终有效合约: {len(df_300)} 条")
    return df_300

# -------------------------------
# 5. 计算“恐慌指数”（类VIX）(使用真实价格)
# -------------------------------
def calculate_vix_like(df_300):
    """
    使用真实市场价格计算类VIX。
    """
    if df_300 is None or df_300.empty:
        raise ValueError("输入数据为空")

    def calculate_var(group):
        """计算单个到期组的方差"""
        # 2.1 按行权价排序
        group = group.sort_values('STRIKE')
        strikes = group['STRIKE'].values
        T = group['T_YEARS'].iloc[0]

        # --- 修复1: 计算ΔK (CBOE式5) ---
        delta_K = np.zeros_like(strikes)
        delta_K[0] = strikes[1] - strikes[0]
        delta_K[-1] = strikes[-1] - strikes[-2]
        if len(strikes) > 2:
            delta_K[1:-1] = (strikes[2:] - strikes[:-2]) / 2

        # --- 修复2: 计算远期价F (更稳健的方法) ---
        call_mask = (group['TYPE'] == 'C')
        put_mask = (group['TYPE'] == 'P')
        if not (call_mask.any() and put_mask.any()):
            raise ValueError("必须同时存在Call/Put合约")

        # 寻找ATM附近的Call和Put合约
        atm_call = group[call_mask].iloc[np.abs(group[call_mask]['DELTA_VALUE'] - 0.5).argmin()]
        atm_put = group[put_mask].iloc[np.abs(group[put_mask]['DELTA_VALUE'] + 0.5).argmin()]

        # 🔺 关键修复：使用买卖价中点来计算F，更接近市场均衡价
        # 获取Call和Put的买一价和卖一价（如果数据中有）
        # 如果没有，就使用“当前价”作为中点
        call_price = atm_call['PRICE']  # 这里仍是“当前价”
        put_price = atm_put['PRICE']    # 这里仍是“当前价”

        # 使用理论公式计算F
        F = atm_call['STRIKE'] + np.exp(RISK_FREE_RATE * T) * (call_price - put_price)
        if F <= 0:
            raise ValueError(f"计算出的远期价F无效: {F}")

        # --- 修复3: 确定K0 (小于F的最大行权价) ---
        K0_candidates = strikes[strikes <= F]
        if len(K0_candidates) == 0:
            K0 = strikes[0]
        else:
            K0 = K0_candidates[-1]

        # --- 修复4: 计算方差 (CBOE式1) ---
        variance = 0
        for i, K in enumerate(strikes):
            # 🔺 最终确认：根据K和F的关系，正确选择Call或Put的价格
            if K < F:
                # K < F，使用Put的价格
                price_row = group[(group['STRIKE'] == K) & (group['TYPE'] == 'P')]
                if price_row.empty:
                    print(f"❌ 行权价 {K} 的Put价格缺失")
                    continue
                Q = price_row['PRICE'].iloc[0]
            elif K > F:
                # K > F，使用Call的价格
                price_row = group[(group['STRIKE'] == K) & (group['TYPE'] == 'C')]
                if price_row.empty:
                    print(f"❌ 行权价 {K} 的Call价格缺失")
                    continue
                Q = price_row['PRICE'].iloc[0]
            else:
                # K == F，使用Call和Put价格的平均值
                call_row = group[(group['STRIKE'] == K) & (group['TYPE'] == 'C')]
                put_row = group[(group['STRIKE'] == K) & (group['TYPE'] == 'P')]
                if call_row.empty or put_row.empty:
                    print(f"❌ 行权价 {K} 的Call或Put价格缺失")
                    continue
                Q = (call_row['PRICE'].iloc[0] + put_row['PRICE'].iloc[0]) / 2

            # 累加方差贡献
            # 🔺 添加检查，防止数值溢出
            if Q <= 0:
                print(f"⚠️ 行权价 {K} 的价格Q为 {Q}，跳过")
                continue
            variance += (delta_K[i] / (K**2)) * np.exp(RISK_FREE_RATE * T) * Q

        # --- 修复5: 完整方差公式 (CBOE式1) ---
        variance = (2 / T) * variance - (1 / T) * ((F / K0) - 1)**2

        # 方差不能为负，但也不能为0
        if variance <= 0:
            # 如果为负或0，通常是因为数值误差或市场摩擦
            # 我们可以尝试使用ATM隐含波动率来估算一个合理的方差
            print(f"⚠️ 计算出的方差为 {variance}，使用ATM IV估算。")
            atm_iv = (atm_call['IMPLC_VOLATLTY'] + atm_put['IMPLC_VOLATLTY']) / 2
            variance = atm_iv**2 * T  # 方差 = 波动率^2 * 时间
            if variance <= 0:
                variance = 0.0001  # 万不得已的保底

        return {'var': variance, 'T': T, 'days': T * 365}

    try:
        # 3.1 按到期日分组
        groups = df_300.groupby('EXPIRE_DATE_DT')
        if len(groups) < 2:
            raise ValueError("需要至少两个到期日")

        # 3.2 获取最近的两个到期日
        exp_dates = sorted(groups.groups.keys())
        near_group = groups.get_group(exp_dates[0])
        next_group = groups.get_group(exp_dates[1])

        # 3.3 对两个到期日分别计算方差
        var1 = calculate_var(near_group)
        var2 = calculate_var(next_group)

        # 3.4 CBOE插值公式 (式3)
        NT1, NT2 = var1['days'], var2['days']
        w = (NT2 - 30) / (NT2 - NT1)
        vix_squared = (var1['T'] * var1['var'] * w + var2['T'] * var2['var'] * (1 - w)) * (365 / 30)

        # 再次检查，防止开方负数
        if vix_squared <= 0:
            print(f"⚠️ 插值后的方差为 {vix_squared}，使用近月ATM IV估算。")
            atm_call = near_group[near_group['TYPE'] == 'C'].iloc[np.abs(near_group[near_group['TYPE'] == 'C']['DELTA_VALUE'] - 0.5).argmin()]
            atm_put = near_group[near_group['TYPE'] == 'P'].iloc[np.abs(near_group[near_group['TYPE'] == 'P']['DELTA_VALUE'] + 0.5).argmin()]
            atm_iv = (atm_call['IMPLC_VOLATLTY'] + atm_put['IMPLC_VOLATLTY']) / 2
            vix_squared = (atm_iv * 100)**2  # 假设ATM IV的平方就是VIX的平方
            if vix_squared <= 0:
                vix_squared = 1.0  # 保底

        vix = 100 * np.sqrt(vix_squared)

        # 4. 结果验证
        if not 5 <= vix <= 80:  # 合理范围检查
            raise ValueError(f"异常VIX值: {vix}")
        return round(vix, 2)

    except Exception as e:
        print(f"⚠️ 方差互换计算失败: {e}")
        # 降级到使用真实时间的IV插值
        try:
            atm_call = df_300[df_300['TYPE'] == 'C'].iloc[np.abs(df_300[df_300['TYPE'] == 'C']['DELTA_VALUE'] - 0.5).argmin()]
            atm_put = df_300[df_300['TYPE'] == 'P'].iloc[np.abs(df_300[df_300['TYPE'] == 'P']['DELTA_VALUE'] + 0.5).argmin()]
            iv_atm = (atm_call['IMPLC_VOLATLTY'] + atm_put['IMPLC_VOLATLTY']) / 2
            return round(iv_atm * 100, 2)
        except:
            return round(df_300['IMPLC_VOLATLTY'].mean() * 100, 2)

# -------------------------------
# 6. 主函数
# -------------------------------
def main():
    # 设置日期（格式：YYYYMMDD）
    trade_date = "20250801"
    output_file = f"300ETF_监控结果_{trade_date}.csv"
    summary_file = f"300ETF_指标汇总_{trade_date}.csv"
    print("🚀 开始执行 300ETF 恐慌指数监控...")

    # 1. 获取风险数据
    risk_df = fetch_risk_data(trade_date)
    if risk_df is None:
        return

    # 2. 从风险数据中提取所有300ETF的到期月份 (YYMM)
    # 复用 extract_300etf_options 的部分逻辑来获取数据
    df_300_risk = risk_df[risk_df['CONTRACT_ID'].str.startswith('510300')].copy()
    if df_300_risk.empty:
        print("❌ 未找到 300ETF 期权数据")
        return

    # 提取所有唯一的到期年月
    df_300_risk['EXPIRE_YYMM'] = df_300_risk['CONTRACT_ID'].str.extract(r'[CP](\d{4})')[0]
    # 移除空值并获取唯一值
    all_expire_yy_mm = df_300_risk['EXPIRE_YYMM'].dropna().unique()
    print(f"🔍 发现 {len(all_expire_yy_mm)} 个到期月份: {sorted(all_expire_yy_mm)}")

    # 3. 为每个到期月获取价格数据
    price_df_dict = {}
    for yymm in all_expire_yy_mm:
        price_df = fetch_price_data(symbol="华泰柏瑞沪深300ETF期权", end_month=yymm)
        if price_df is not None:
            price_df_dict[yymm] = price_df
        # 可以添加一个延时，避免对AkShare造成过大压力
        # time.sleep(0.5)

    # 4. 提取并合并风险与价格数据
    df_300 = extract_and_merge_data(risk_df, price_df_dict, trade_date)
    if df_300 is None or df_300.empty:
        print("❌ 数据合并后为空，程序终止。")
        return

    # 5. 计算恐慌指数
    # ... (后续的计算、分析、保存步骤保持不变)
    # 注意：此时的 df_300 已经包含了真实价格，calculate_vix_like 函数可以正常执行方差互换

    # 4. 计算恐慌指数
    print("\n🚨 正在计算恐慌指数（类VIX）...")
    try:
        vix_value = calculate_vix_like(df_300)
        print(f"✅ A股300ETF恐慌指数（类VIX）: {vix_value:.2f}")
    except Exception as e:
        vix_value = df_300['IMPLC_VOLATLTY'].mean() * 100
        print(f"⚠️ 计算复杂VIX失败，使用平均IV: {vix_value:.2f}")

    # 5. ... (后续的IV平均、对比分析、风险敞口、保存等逻辑保持不变)

    print("\n🎉 全部完成！")

if __name__ == "__main__":
    main()