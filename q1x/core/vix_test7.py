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

# -------------------------------
# 2. 从 AkShare 获取数据
# -------------------------------
def fetch_risk_data(trade_date: str):
    try:
        print(f"📡 正在从 AkShare 获取 {trade_date} 风险数据...")
        df = ak.option_risk_indicator_sse(date=trade_date)
        if df is not None and not df.empty:
            print(f"✅ 成功获取 {len(df)} 条数据")
            return df
        else:
            print("❌ 未获取到数据，请检查日期是否为交易日")
            return None
    except Exception as e:
        print(f"❌ 获取数据失败: {e}")
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
def extract_300etf_options(df, trade_date_str: str):
    df_300 = df[df['CONTRACT_ID'].str.startswith('510300')].copy()
    if df_300.empty:
        print("❌ 未找到 300ETF 期权数据")
        return None

    # 提取期权类型 (C/P)
    df_300['TYPE'] = df_300['CONTRACT_ID'].str.extract(r'([CP])')[0]

    # --- 关键改进：从合约代码中提取到期年月并计算真实到期日 ---
    df_300['EXPIRE_YYMM'] = df_300['CONTRACT_ID'].str.extract(r'[CP](\d{4})')[0]

    # 过滤掉无法提取年月的行
    if df_300['EXPIRE_YYMM'].isnull().any():
        df_300 = df_300.dropna(subset=['EXPIRE_YYMM']).copy()
        print(f"⚠️ 过滤了无法提取年月的合约")

    # 解析年月并计算真实到期日
    def calc_expire_date(yy_mm: str) -> date:
        try:
            year = 2000 + int(yy_mm[:2])
            month = int(yy_mm[2:4])
            return get_fourth_wednesday(year, month)
        except Exception as e:
            print(f"❌ 计算到期日失败 {yy_mm}: {e}")
            return None

    df_300['EXPIRE_DATE_DT'] = df_300['EXPIRE_YYMM'].apply(calc_expire_date)

    # 再次过滤计算失败的行
    if df_300['EXPIRE_DATE_DT'].isnull().any():
        df_300 = df_300.dropna(subset=['EXPIRE_DATE_DT']).copy()
        print(f"⚠️ 过滤了计算到期日失败的合约")

    # --- 计算真实剩余时间 T (以年为单位) ---
    current_date = datetime.strptime(trade_date_str, "%Y%m%d").date()
    df_300['EXPIRE_DATE_DT'] = pd.to_datetime(df_300['EXPIRE_DATE_DT'])
    df_300['T_DAYS'] = (df_300['EXPIRE_DATE_DT'] - pd.to_datetime(current_date)).dt.days
    df_300['T_YEARS'] = df_300['T_DAYS'] / 365.0

    # 提取行权价和IV
    df_300['STRIKE'] = df_300['CONTRACT_SYMBOL'].str.extract(r'(\d+(?:\.\d+)?)')[0].astype(float)
    iv_col = [col for col in df_300.columns if 'IMPLC' in col][0]
    df_300.rename(columns={iv_col: 'IMPLC_VOLATLTY'}, inplace=True)

    # 转换为数值型并清洗
    numeric_cols = ['DELTA_VALUE', 'THETA_VALUE', 'GAMMA_VALUE', 'VEGA_VALUE', 'RHO_VALUE', 'IMPLC_VOLATLTY']
    for col in numeric_cols:
        df_300[col] = pd.to_numeric(df_300[col], errors='coerce')
    df_300 = df_300[(df_300['IMPLC_VOLATLTY'] > 0.01) & (df_300['IMPLC_VOLATLTY'] < 1.0)]

    print(f"✅ 提取 300ETF 期权: {len(df_300)} 条，已计算真实剩余时间")
    return df_300

# -------------------------------
# 5. 计算“恐慌指数”（类VIX）(使用真实T的IV插值)
# -------------------------------
def calculate_vix_like(df_300):
    """
    使用真实剩余时间T，对近月和次近月的ATM IV进行线性插值，计算30天VIX。
    这是当前数据条件下最合理、最稳定的方法。
    """
    if df_300 is None or df_300.empty:
        raise ValueError("输入数据为空")

    def get_atm_weighted_iv(group):
        """计算一组期权的ATM加权IV"""
        call_mask = (group['TYPE'] == 'C')
        put_mask = (group['TYPE'] == 'P')
        if not (call_mask.any() and put_mask.any()):
            raise ValueError("必须同时存在Call/Put合约")
        # 寻找ATM附近的Call和Put
        atm_call = group[call_mask].iloc[np.abs(group[call_mask]['DELTA_VALUE'] - 0.5).argmin()]
        atm_put = group[put_mask].iloc[np.abs(group[put_mask]['DELTA_VALUE'] + 0.5).argmin()]
        # 返回Call和Put IV的平均值作为ATM IV
        return (atm_call['IMPLC_VOLATLTY'] + atm_put['IMPLC_VOLATLTY']) / 2

    try:
        # 1. 按真实到期日分组
        groups = df_300.groupby('EXPIRE_DATE_DT')
        if len(groups) < 2:
            raise ValueError("需要至少两个到期日")

        # 2. 获取最近的两个到期日
        exp_dates = sorted(groups.groups.keys())
        near_term = groups.get_group(exp_dates[0])
        next_term = groups.get_group(exp_dates[1])

        # 3. 计算两个期限的ATM IV
        iv_near = get_atm_weighted_iv(near_term)
        iv_next = get_atm_weighted_iv(next_term)

        # 4. 获取真实的剩余时间 (以天为单位)
        T1_days = near_term['T_DAYS'].mean()
        T2_days = next_term['T_DAYS'].mean()

        # 5. 使用线性插值计算30天的VIX
        # 公式: VIX = IV_near + (IV_next - IV_near) * (30 - T1) / (T2 - T1)
        if T2_days > T1_days and T2_days >= 30 >= T1_days:
            weight = (30 - T1_days) / (T2_days - T1_days)
            vix = iv_near * (1 - weight) + iv_next * weight
        else:
            # 如果30天不在两个期限之间，直接使用近月IV
            vix = iv_near

        vix_value = vix * 100  # 转为百分比

        # 6. 合理性检查
        if not 5 <= vix_value <= 80:
            raise ValueError(f"异常VIX值: {vix_value}")

        return round(vix_value, 2)

    except Exception as e:
        print(f"⚠️ IV插值计算失败: {e}")
        # 降级到简单平均
        fallback_iv = df_300['IMPLC_VOLATLTY'].mean()
        fallback_value = (fallback_iv * 100) if not np.isnan(fallback_iv) else 20.0
        print(f"📉 降级使用IV平均: {fallback_value:.2f}")
        return round(fallback_value, 2)

# -------------------------------
# 6. 计算组合风险敞口（假设每合约1张）
# -------------------------------
def calculate_risk_exposure(df_300):
    df_300['position'] = 1
    total_delta = (df_300['DELTA_VALUE'] * df_300['position']).sum()
    total_gamma = (df_300['GAMMA_VALUE'] * df_300['position']).sum()
    total_vega = (df_300['VEGA_VALUE'] * df_300['position']).sum()
    total_theta = (df_300['THETA_VALUE'] * df_300['position']).sum()
    return {
        'Delta': round(total_delta, 3),
        'Gamma': round(total_gamma, 3),
        'Vega': round(total_vega, 3),
        'Theta': round(total_theta, 3)
    }

# -------------------------------
# 7. 读取历史 VIX 数据
# -------------------------------
def load_vix_history():
    if os.path.exists(HISTORY_DATA_FILE):
        history_df = pd.read_csv(HISTORY_DATA_FILE)
        if 'VIX_INDEX' in history_df.columns:
            return history_df['VIX_INDEX'].dropna().tolist()
    return []

# -------------------------------
# 8. 保存当前 VIX 到历史记录
# -------------------------------
def save_vix_to_history(trade_date, vix_value, iv_mean, comparison):
    new_row = pd.DataFrame([{'TRADE_DATE': trade_date, 'VIX_INDEX': vix_value, 'IV_MEAN': iv_mean, 'COMPARISON': comparison}])
    if os.path.exists(HISTORY_DATA_FILE):
        history_df = pd.read_csv(HISTORY_DATA_FILE)
        history_df = history_df[history_df['TRADE_DATE'] != trade_date]
        history_df = pd.concat([history_df, new_row], ignore_index=True)
    else:
        history_df = new_row
    history_df.to_csv(HISTORY_DATA_FILE, index=False, encoding='utf-8-sig')

# -------------------------------
# 9. 判断 VIX 历史分位数
# -------------------------------
def analyze_vix_quantile(vix_value, history_vix_list):
    if len(history_vix_list) < 5:
        print("📊 历史数据不足5条")
        return "数据不足"
    low = np.quantile(history_vix_list, HISTORICAL_QUANTILE_LOW)
    high = np.quantile(history_vix_list, HISTORICAL_QUANTILE_HIGH)
    position = "低位" if vix_value < low else "高位" if vix_value > high else "中位"
    suggestion = "可考虑买入期权" if vix_value < low else "适合卖出期权" if vix_value > high else "市场平稳"
    print(f"📊 当前VIX: {vix_value:.2f} → {position} (历史{len(history_vix_list)}条)")
    print(f"💡 建议: {suggestion}")
    return position

# -------------------------------
# 10. 主函数
# -------------------------------
def main():
    trade_date = "20250801"  # 修改为您需要的日期
    output_file = f"300ETF_监控结果_{trade_date}.csv"
    summary_file = f"300ETF_指标汇总_{trade_date}.csv"
    print("🚀 开始执行 300ETF 恐慌指数监控...")

    # 1. 获取数据
    df = fetch_risk_data(trade_date)
    if df is None:
        return

    # 2. 提取 300ETF 期权
    df_300 = extract_300etf_options(df, trade_date)
    if df_300 is None or df_300.empty:
        print("❌ 提取 300ETF 期权后数据为空，程序终止。")
        return

    # 3. 计算恐慌指数
    print("\n🚨 正在计算恐慌指数（类VIX）...")
    try:
        vix_value = calculate_vix_like(df_300)
        print(f"✅ A股300ETF恐慌指数（类VIX）: {vix_value:.2f}")
    except Exception as e:
        vix_value = df_300['IMPLC_VOLATLTY'].mean() * 100
        print(f"⚠️ 计算复杂VIX失败，使用平均IV: {vix_value:.2f}")

    # 4. 计算 IV 平均值
    iv_mean = df_300['IMPLC_VOLATLTY'].mean() * 100
    print(f"📊 IV平均值: {iv_mean:.2f}")

    # 5. VIX vs IV平均 对比分析
    print("\n🔍 VIX vs IV平均 对比分析:")
    if iv_mean == 0:
        print("❌ IV平均为0，无法比较")
        comparison = "N/A"
    else:
        diff = (vix_value - iv_mean) / iv_mean
        if diff < -VIX_THRESHOLD_LOW:
            print(f"📉 VIX 显著低于 IV平均")
            comparison = "VIX显著低于IV"
        elif diff > VIX_THRESHOLD_HIGH:
            print(f"📈 VIX 显著高于 IV平均")
            comparison = "VIX显著高于IV"
        else:
            print(f"🟰 VIX 与 IV平均 基本一致")
            comparison = "VIX与IV基本一致"

    # 6. 历史分位数判断
    print("\n📈 历史分位数分析:")
    history_vix_list = load_vix_history()
    quantile_position = analyze_vix_quantile(vix_value, history_vix_list)

    # 7. 计算风险敞口
    print("\n🛡️  正在计算风险敞口...")
    risk = calculate_risk_exposure(df_300)
    for k, v in risk.items():
        print(f"   {k}: {v}")

    # 8. 保存到 CSV
    df_300['VIX_INDEX'] = vix_value
    df_300['IV_MEAN'] = iv_mean
    df_300.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"💾 300ETF 期权数据已保存至 {output_file}")

    # 9. 保存当前 VIX 到历史记录
    save_vix_to_history(trade_date, vix_value, iv_mean, comparison)
    print("\n🎉 全部完成！")

if __name__ == "__main__":
    main()