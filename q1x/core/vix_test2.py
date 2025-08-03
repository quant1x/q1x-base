import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import os

# -------------------------------
# 1. 常量定义
# -------------------------------
VIX_THRESHOLD_LOW = 0.05      # VIX 相对于 IV 均值的“显著偏低”阈值（5%）
VIX_THRESHOLD_HIGH = 0.05     # VIX 相对于 IV 均值的“显著偏高”阈值（5%）
HISTORY_DATA_FILE = "vix_history_300etf.csv"  # 历史数据存储文件
HISTORICAL_QUANTILE_LOW = 0.2  # 历史低位分位数（20%）
HISTORICAL_QUANTILE_HIGH = 0.8 # 历史高位分位数（80%）

# -------------------------------
# 2. 从 AkShare 获取数据
# -------------------------------
def fetch_risk_data(trade_date: str):
    """
    从 AkShare 获取上交所期权风险数据
    :param trade_date: 日期，格式 YYYYMMDD，如 "20250731"
    :return: DataFrame
    """
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
    """
    计算指定年月的第四个星期三。
    上交所ETF期权的到期日为合约月的第四个星期三。
    :param year: 年份 (int)
    :param month: 月份 (int)
    :return: 到期日 (datetime.date)
    """
    first_day = datetime(year, month, 1)
    # 找到第一个周三
    first_wed = first_day + timedelta(days=(2 - first_day.weekday()) % 7)
    # 第四个周三 = 第一个周三 + 3周
    fourth_wed = first_wed + timedelta(weeks=3)

    # 检查跨月情况（如春节假期顺延）
    if fourth_wed.month != month:
        print(f"⚠️ {year}年{month}月合约将跨月交割至{fourth_wed.month}月")
    return fourth_wed.date()

# -------------------------------
# 4. 提取 300ETF 期权数据 (使用真实剩余时间)
# -------------------------------
def extract_300etf_options(df, trade_date_str: str):
    """
    提取沪深300ETF期权数据，并解析关键字段，包括计算真实剩余时间。
    :param df: 从 AkShare 获取的原始风险数据 DataFrame
    :param trade_date_str: 交易日期字符串，格式 "YYYYMMDD"
    :return: 处理后的 300ETF 期权数据 DataFrame，包含真实剩余时间 T
    """
    # 筛选 300ETF 期权：CONTRACT_ID 以 510300 开头
    df_300 = df[df['CONTRACT_ID'].str.startswith('510300')].copy()

    # === 保留您原有的所有代码 ===
    # 您的原始行权价提取、类型解析等代码全部保留
    df_300['TYPE'] = df_300['CONTRACT_ID'].str.extract(r'([CP])')[0]
    df_300['STRIKE'] = df_300['CONTRACT_SYMBOL'].str.extract(r'(\d+(?:\.\d+)?)')[0].astype(float)

    # === 仅新增日期处理（插入到原有代码中间）===
    # 解析年月（如2508 -> 2025年8月）
    df_300['expiry_yy'] = df_300['CONTRACT_ID'].str[7:9].astype(int) + 2000
    df_300['expiry_mm'] = df_300['CONTRACT_ID'].str[9:11].astype(int)

    # 计算到期日（确保返回Timestamp类型）
    df_300['EXPIRE_DATE'] = pd.to_datetime(df_300.apply(
        lambda x: get_fourth_wednesday(x['expiry_yy'], x['expiry_mm']), axis=1))

    # 计算剩余时间（保持与您原有逻辑兼容）
    trade_date = pd.to_datetime(trade_date_str)
    df_300['T_DAYS'] = (df_300['EXPIRE_DATE'] - trade_date).dt.days
    df_300['T_YEARS'] = df_300['T_DAYS'] / 365.25

    # === 继续保留您原有的IV处理和其他逻辑 ===
    iv_col = [col for col in df_300.columns if 'IMPLC' in col][0]
    df_300.rename(columns={iv_col: 'IMPLC_VOLATLTY'}, inplace=True)
    df_300 = df_300[(df_300['IMPLC_VOLATLTY'] > 0.01) & (df_300['IMPLC_VOLATLTY'] < 1.0)]
    return df_300

# -------------------------------
# 5. 计算“恐慌指数”（类VIX）(使用真实T)
# -------------------------------
def calculate_vix_like(df_300):
    """
    计算类VIX指数（基于近月+次近月期权，使用真实剩余时间）。
    注意：这是一个简化版，旨在反映市场对短期波动的预期，而非精确复制CBOE VIX。
    """
    if df_300 is None or df_300.empty:
        print("❌ 输入数据为空")
        return np.nan

    # 按真实到期日分组
    exp_groups = df_300.groupby('EXPIRE_DATE_DT')
    if len(exp_groups) < 2:
        print("⚠️ 数据不足两个到期日，使用所有数据平均")
        return df_300['IMPLC_VOLATLTY'].mean() * 100

    # 获取最近两个到期日
    all_expirations = sorted(exp_groups.groups.keys())  # 排序得到时间顺序
    near_exp_date = all_expirations[0]  # 最近的到期日
    next_exp_date = all_expirations[1]  # 次近的到期日

    near_term = exp_groups.get_group(near_exp_date)
    next_term = exp_groups.get_group(next_exp_date)

    def get_atm_weighted_iv(group):
        """
        计算一组期权（一个到期日）的平值附近加权隐含波动率。
        """
        # 找 ATM 附近合约（Call Delta ~0.5，Put Delta ~-0.5）
        # 取 Delta 绝对值最接近目标值的前3个合约
        if 'C' in group['TYPE'].values:
            atm_call = group[group['TYPE'] == 'C'].iloc[
                (group[group['TYPE'] == 'C']['DELTA_VALUE'] - 0.5).abs().argsort()[:3]
            ]
        else:
            atm_call = pd.DataFrame()
        if 'P' in group['TYPE'].values:
            atm_put = group[group['TYPE'] == 'P'].iloc[
                (group[group['TYPE'] == 'P']['DELTA_VALUE'] + 0.5).abs().argsort()[:3]
            ]
        else:
            atm_put = pd.DataFrame()
        # 合并看涨和看跌的ATM合约
        combined = pd.concat([atm_call, atm_put])
        # 如果没有找到合适的ATM合约，则使用该组所有合约的平均IV
        if len(combined) == 0:
            return group['IMPLC_VOLATLTY'].mean()
        # 返回ATM附近合约的平均IV
        return combined['IMPLC_VOLATLTY'].mean()

    # 计算近月和次近月的ATM加权IV
    iv_near = get_atm_weighted_iv(near_term)
    iv_next = get_atm_weighted_iv(next_term)

    # --- 使用真实剩余时间 ---
    # 使用组内所有合约的平均剩余时间（年化）
    T1 = near_term['T_YEARS'].mean()  # 近月合约的真实剩余时间
    T2 = next_term['T_YEARS'].mean()  # 次近月合约的真实剩余时间

    # --- 修正 VIX 公式 ---
    # 原始公式有误。这里采用一个更合理、更稳定的简化逻辑：线性插值。
    TARGET_T = 30 / 365.0  # 目标：30天（年化）
    if T2 > T1 and T2 > TARGET_T and T1 < TARGET_T:
        # 线性插值：vix = iv_near + (iv_next - iv_near) * (TARGET_T - T1) / (T2 - T1)
        weight = (TARGET_T - T1) / (T2 - T1)
        vix = iv_near * (1 - weight) + iv_next * weight
    else:
        # 如果目标时间不在范围内，直接使用近月IV
        vix = iv_near

    return vix * 100  # 转为百分比

# -------------------------------
# 6. 计算组合风险敞口（假设每合约1张）
# -------------------------------
def calculate_risk_exposure(df_300):
    """
    计算假设持仓下的总风险敞口（Delta, Gamma, Vega, Theta）。
    当前假设每种期权都持有1张。
    :param df_300: 期权数据 DataFrame
    :return: 包含各项风险敞口的字典
    """
    df_300['position'] = 1  # 假设每张合约持有1份，可替换为真实持仓
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
    """读取历史 VIX 数据列表"""
    if os.path.exists(HISTORY_DATA_FILE):
        history_df = pd.read_csv(HISTORY_DATA_FILE)
        if 'VIX_INDEX' in history_df.columns:
            return history_df['VIX_INDEX'].dropna().tolist()
    return []

# -------------------------------
# 8. 保存当前 VIX 到历史记录
# -------------------------------
def save_vix_to_history(trade_date, vix_value, iv_mean, comparison):
    """将当前 VIX 保存到历史文件"""
    new_row = {
        'TRADE_DATE': trade_date,
        'VIX_INDEX': vix_value,
        'IV_MEAN': iv_mean,
        'COMPARISON': comparison
    }
    new_df = pd.DataFrame([new_row])
    if os.path.exists(HISTORY_DATA_FILE):
        history_df = pd.read_csv(HISTORY_DATA_FILE)
        # 避免重复日期
        if trade_date in history_df['TRADE_DATE'].values:
            history_df = history_df[history_df['TRADE_DATE'] != trade_date]
        history_df = pd.concat([history_df, new_df], ignore_index=True)
    else:
        history_df = new_df
    history_df.to_csv(HISTORY_DATA_FILE, index=False, encoding='utf-8-sig')
    print(f"💾 当前 VIX 已保存至历史记录: {HISTORY_DATA_FILE}")

# -------------------------------
# 9. 判断 VIX 历史分位数
# -------------------------------
def analyze_vix_quantile(vix_value, history_vix_list):
    """判断当前 VIX 在历史数据中的分位数位置"""
    if len(history_vix_list) < 5:
        print("📊 历史数据不足5条，暂不判断分位数")
        return "数据不足"
    low_threshold = np.quantile(history_vix_list, HISTORICAL_QUANTILE_LOW)
    high_threshold = np.quantile(history_vix_list, HISTORICAL_QUANTILE_HIGH)
    if vix_value < low_threshold:
        position = "低位"
        suggestion = "可考虑买入期权布局波动"
    elif vix_value > high_threshold:
        position = "高位"
        suggestion = "适合卖出期权，收割溢价"
    else:
        position = "中位"
        suggestion = "市场预期平稳，正常操作"
    print(f"📊 历史分位数判断:")
    print(f"   历史数据量: {len(history_vix_list)}")
    print(f"   低位阈值({HISTORICAL_QUANTILE_LOW*100:.0f}%): {low_threshold:.2f}")
    print(f"   高位阈值({HISTORICAL_QUANTILE_HIGH*100:.0f}%): {high_threshold:.2f}")
    print(f"   当前VIX: {vix_value:.2f} → 处于 **{position}**")
    print(f"💡 建议: {suggestion}")
    return position

# -------------------------------
# 10. 主函数
# -------------------------------
def main():
    # 设置日期（格式：YYYYMMDD）
    trade_date = "20250801"  # 可改为 datetime.now().strftime("%Y%m%d")
    output_file = f"300ETF_监控结果_{trade_date}.csv"
    summary_file = f"300ETF_指标汇总_{trade_date}.csv"
    print("🚀 开始执行 300ETF 恐慌指数监控...")

    # 1. 获取数据
    df = fetch_risk_data(trade_date)
    if df is None:
        return

    # 2. 提取 300ETF 期权
    # 注意：这里将 trade_date 传入，以便计算真实剩余时间
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
    print(f"📊 所有期权隐含波动率（IV）平均值: {iv_mean:.2f}")

    # 5. VIX vs IV平均 对比分析（带阈值）
    print("\n🔍 VIX vs IV平均 对比分析（带阈值）:")
    if iv_mean == 0:
        print("❌ IV平均为0，无法比较")
        comparison = "N/A"
    else:
        diff = (vix_value - iv_mean) / iv_mean  # 相对差异
        if diff < -VIX_THRESHOLD_LOW:
            print(f"📉 VIX ({vix_value:.2f}) 比 IV平均 ({iv_mean:.2f}) 低 {abs(diff):.1%}")
            print("👉 市场对未来30天的波动预期显著低于当前整体情绪，可能正在降温。")
            print("💡 适合卖出期权，收割时间价值。")
            comparison = "VIX显著低于IV"
        elif diff > VIX_THRESHOLD_HIGH:
            print(f"📈 VIX ({vix_value:.2f}) 比 IV平均 ({iv_mean:.2f}) 高 {diff:.1%}")
            print("👉 市场对未来波动的担忧显著高于当前平均水平，可能预期事件发生。")
            print("💡 警惕短期波动加剧，可考虑对冲或买入保险。")
            comparison = "VIX显著高于IV"
        else:
            print(f"🟰 VIX ({vix_value:.2f}) 与 IV平均 ({iv_mean:.2f}) 基本一致（差异在{VIX_THRESHOLD_LOW:.0%}内）")
            print("👉 市场预期平稳，无明显方向性信号。")
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

    # 8. 添加指标列
    df_300['VIX_INDEX'] = vix_value
    df_300['IV_MEAN'] = iv_mean
    df_300['VIX_IV_COMPARISON'] = comparison
    df_300['HISTORICAL_QUANTILE'] = quantile_position
    df_300['RISK_DELTA'] = risk['Delta']
    df_300['RISK_GAMMA'] = risk['Gamma']
    df_300['RISK_VEGA'] = risk['Vega']
    df_300['RISK_THETA'] = risk['Theta']

    # 9. 保存到 CSV
    df_300.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"💾 300ETF 期权数据已保存至 {output_file}")

    # 10. 保存汇总指标
    summary = pd.DataFrame([{
        'TRADE_DATE': df_300['TRADE_DATE'].iloc[0],
        'INDICATOR': '300ETF监控指标',
        'VIX_INDEX': f"{vix_value:.2f}",
        'IV_MEAN': f"{iv_mean:.2f}",
        'VIX_vs_IV': comparison,
        'HISTORICAL_QUANTILE': quantile_position,
        'AVG_IV': f"{df_300['IMPLC_VOLATLTY'].mean()*100:.2f}",
        'CALL_IV': f"{df_300[df_300['TYPE']=='C']['IMPLC_VOLATLTY'].mean()*100:.2f}",
        'PUT_IV': f"{df_300[df_300['TYPE']=='P']['IMPLC_VOLATLTY'].mean()*100:.2f}",
        'Delta': risk['Delta'],
        'Gamma': risk['Gamma'],
        'Vega': risk['Vega'],
        'Theta': risk['Theta']
    }])
    summary.to_csv(summary_file, index=False, encoding='utf-8-sig')
    print(f"📊 汇总指标已保存至 {summary_file}")

    # 11. 保存当前 VIX 到历史记录
    save_vix_to_history(trade_date, vix_value, iv_mean, comparison)
    print("\n🎉 全部完成！")

# -------------------------------
# 11. 运行
# -------------------------------
if __name__ == "__main__":
    main()