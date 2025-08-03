import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
from typing import Dict, List, Optional, Tuple

# -------------------------------
# 1. 配置常量（使用配置文件）
# -------------------------------
class Config:
    def __init__(self):
        self.VIX_THRESHOLD_LOW = 0.05      # VIX 相对于 IV 均值的"显著偏低"阈值
        self.VIX_THRESHOLD_HIGH = 0.05     # VIX 相对于 IV 均值的"显著偏高"阈值
        self.HISTORY_DATA_FILE = "data/vix_history_300etf.csv"  # 历史数据存储路径
        self.HISTORICAL_QUANTILE_LOW = 0.2  # 历史低位分位数
        self.HISTORICAL_QUANTILE_HIGH = 0.8 # 历史高位分位数
        self.STRIKE_RANGE = 0.05           # 执行价筛选范围(±5%)
        self.MIN_HISTORY_DAYS = 30         # 最小历史数据天数
        self.OUTPUT_DIR = "output"         # 输出目录

        # 确保目录存在
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(self.HISTORY_DATA_FILE), exist_ok=True)

CONFIG = Config()

# -------------------------------
# 2. 数据获取与处理
# -------------------------------
class OptionDataProcessor:
    @staticmethod
    def fetch_risk_data(trade_date: str) -> Optional[pd.DataFrame]:
        """从 AkShare 获取上交所期权风险数据"""
        try:
            print(f"📡 正在从 AkShare 获取 {trade_date} 风险数据...")
            df = ak.option_risk_indicator_sse(date=trade_date)

            if df is not None and not df.empty:
                print(f"✅ 成功获取 {len(df)} 条数据")
                return df

            print("❌ 未获取到数据，请检查日期是否为交易日")
            return None

        except Exception as e:
            print(f"❌ 获取数据失败: {e}")
            return None

    @staticmethod
    def preprocess_300etf_options(df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """预处理300ETF期权数据"""
        if df.empty:
            return None

        # 筛选300ETF期权
        df_300 = df[df['CONTRACT_ID'].str.startswith('510300')].copy()

        if df_300.empty:
            print("❌ 未找到300ETF期权数据")
            return None

        # 解析合约信息
        df_300['TYPE'] = df_300['CONTRACT_ID'].str.extract(r'([CP])')[0]
        df_300['EXPIRE_CODE'] = df_300['CONTRACT_ID'].str.extract(r'(M\d{3,4}[A-Z]?|A\d{3,4}[A-Z]?)')[0]
        df_300['STRIKE'] = df_300['CONTRACT_SYMBOL'].str.extract(r'(\d+(?:\.\d+)?)')[0].astype(float)

        # 处理隐含波动率列
        iv_col = next((col for col in df_300.columns if 'IMPLC' in col), None)
        if iv_col:
            df_300.rename(columns={iv_col: 'IMPLC_VOLATLTY'}, inplace=True)

        # 数值转换
        numeric_cols = ['DELTA_VALUE', 'THETA_VALUE', 'GAMMA_VALUE',
                        'VEGA_VALUE', 'RHO_VALUE', 'IMPLC_VOLATLTY']
        for col in numeric_cols:
            if col in df_300.columns:
                df_300[col] = pd.to_numeric(df_300[col], errors='coerce')

        # 过滤异常值
        df_300 = df_300[
            (df_300['IMPLC_VOLATLTY'] > 0.01) &
            (df_300['IMPLC_VOLATLTY'] < 1.0)
            ].dropna(subset=['IMPLC_VOLATLTY'])

        print(f"✅ 预处理300ETF期权数据: {len(df_300)}条")
        return df_300

# -------------------------------
# 3. VIX计算核心
# -------------------------------
class VIXCalculator:
    @staticmethod
    def calculate_time_to_expiry(expiry_code: str, trade_date: str) -> float:
        """计算到期时间(年化)"""
        # 这里简化处理，实际应解析到期日
        # 示例: M09 -> 9月到期，A12 -> 12月到期
        month_map = {
            'M01': 1, 'M02': 2, 'M03': 3, 'M04': 4, 'M05': 5, 'M06': 6,
            'M07': 7, 'M08': 8, 'M09': 9, 'M10': 10, 'M11': 11, 'M12': 12,
            'A01': 1, 'A02': 2, 'A03': 3, 'A04': 4, 'A05': 5, 'A06': 6,
            'A07': 7, 'A08': 8, 'A09': 9, 'A10': 10, 'A11': 11, 'A12': 12
        }

        try:
            month = month_map.get(expiry_code[:3], 1)
            year = int(trade_date[:4])
            expiry_date = datetime(year, month, 1) + timedelta(days=31)
            trade_date_dt = datetime.strptime(trade_date, "%Y%m%d")
            days_to_expiry = (expiry_date - trade_date_dt).days
            return max(days_to_expiry, 1) / 365.0
        except:
            return 30 / 365.0  # 默认30天

    @staticmethod
    def get_atm_iv(group: pd.DataFrame) -> float:
        """获取ATM附近的隐含波动率"""
        if group.empty:
            return 0.0

        # 处理认购期权
        calls = group[group['TYPE'] == 'C'].copy()
        if not calls.empty:
            calls['delta_diff'] = (calls['DELTA_VALUE'] - 0.5).abs()
            atm_calls = calls.nsmallest(3, 'delta_diff')
        else:
            atm_calls = pd.DataFrame()

        # 处理认沽期权
        puts = group[group['TYPE'] == 'P'].copy()
        if not puts.empty:
            puts['delta_diff'] = (puts['DELTA_VALUE'] + 0.5).abs()
            atm_puts = puts.nsmallest(3, 'delta_diff')
        else:
            atm_puts = pd.DataFrame()

        # 合并结果
        combined = pd.concat([atm_calls, atm_puts])

        if not combined.empty:
            return combined['IMPLC_VOLATLTY'].mean()
        elif not group.empty:
            return group['IMPLC_VOLATLTY'].mean()
        else:
            return 0.0

    @staticmethod
    def calculate_vix(df_300: pd.DataFrame, trade_date: str) -> Tuple[float, Dict]:
        """计算VIX指数"""
        if df_300.empty:
            return 0.0, {}

        # 1. 按到期日分组
        exp_groups = df_300.groupby('EXPIRE_CODE')
        if len(exp_groups) < 2:
            print("⚠️ 数据不足两个到期日，使用所有数据平均")
            iv_mean = df_300['IMPLC_VOLATLTY'].mean()
            return iv_mean * 100, {'method': 'simple_average'}

        # 2. 获取最近两个到期日
        sorted_expiries = sorted(exp_groups.groups.keys())
        near_exp, next_exp = sorted_expiries[:2]

        # 3. 计算时间和IV
        T1 = VIXCalculator.calculate_time_to_expiry(near_exp, trade_date)
        T2 = VIXCalculator.calculate_time_to_expiry(next_exp, trade_date)

        near_iv = VIXCalculator.get_atm_iv(exp_groups.get_group(near_exp))
        next_iv = VIXCalculator.get_atm_iv(exp_groups.get_group(next_exp))

        # 4. VIX核心公式
        try:
            vix = np.sqrt(
                (T1 * near_iv**2 * ((T2 - 1/365) / (T2 - T1))) +
                (T2 * next_iv**2 * ((1/365 - T1) / (T2 - T1)))
            ) * 100

            details = {
                'method': 'standard',
                'near_expiry': near_exp,
                'next_expiry': next_exp,
                'T1': T1,
                'T2': T2,
                'near_iv': near_iv,
                'next_iv': next_iv
            }
            return vix, details
        except Exception as e:
            print(f"⚠️ VIX计算异常: {e}")
            # 计算失败时回退到加权平均
            weighted_iv = (near_iv * T1 + next_iv * T2) / (T1 + T2)
            return weighted_iv * 100, {'method': 'weighted_average'}

    @staticmethod
    def calculate_iv_stats(df_300: pd.DataFrame) -> Dict:
        """计算IV统计指标"""
        if df_300.empty:
            return {}

        call_iv = df_300[df_300['TYPE'] == 'C']['IMPLC_VOLATLTY']
        put_iv = df_300[df_300['TYPE'] == 'P']['IMPLC_VOLATLTY']

        return {
            'iv_mean': df_300['IMPLC_VOLATLTY'].mean() * 100,
            'iv_median': df_300['IMPLC_VOLATLTY'].median() * 100,
            'call_iv_mean': call_iv.mean() * 100 if not call_iv.empty else 0,
            'put_iv_mean': put_iv.mean() * 100 if not put_iv.empty else 0,
            'skew': (call_iv.mean() - put_iv.mean()) * 100 if (not call_iv.empty and not put_iv.empty) else 0,
            'iv_range': (df_300['IMPLC_VOLATLTY'].max() - df_300['IMPLC_VOLATLTY'].min()) * 100
        }

# -------------------------------
# 4. 风险分析
# -------------------------------
class RiskAnalyzer:
    @staticmethod
    def calculate_greeks_exposure(df_300: pd.DataFrame) -> Dict:
        """计算希腊值风险敞口"""
        if df_300.empty:
            return {}

        df_300['position'] = 1  # 假设每合约1张

        return {
            'delta': round((df_300['DELTA_VALUE'] * df_300['position']).sum(), 3),
            'gamma': round((df_300['GAMMA_VALUE'] * df_300['position']).sum(), 3),
            'vega': round((df_300['VEGA_VALUE'] * df_300['position']).sum(), 3),
            'theta': round((df_300['THETA_VALUE'] * df_300['position']).sum(), 3)
        }

    @staticmethod
    def analyze_vix_position(vix_value: float, history_vix: List[float]) -> Dict:
        """分析VIX历史位置"""
        if len(history_vix) < CONFIG.MIN_HISTORY_DAYS:
            return {
                'position': 'insufficient_data',
                'quantile': None,
                'suggestion': '需要更多历史数据'
            }

        current_quantile = np.mean(np.array(history_vix) < vix_value)
        low_threshold = np.quantile(history_vix, CONFIG.HISTORICAL_QUANTILE_LOW)
        high_threshold = np.quantile(history_vix, CONFIG.HISTORICAL_QUANTILE_HIGH)

        if vix_value < low_threshold:
            position = 'low'
            suggestion = '可考虑买入期权布局波动'
        elif vix_value > high_threshold:
            position = 'high'
            suggestion = '适合卖出期权，收割溢价'
        else:
            position = 'medium'
            suggestion = '市场预期平稳，正常操作'

        return {
            'position': position,
            'quantile': round(current_quantile, 3),
            'low_threshold': round(low_threshold, 2),
            'high_threshold': round(high_threshold, 2),
            'suggestion': suggestion
        }

# -------------------------------
# 5. 数据存储与报告
# -------------------------------
class DataManager:
    @staticmethod
    def load_history() -> List[float]:
        """加载历史VIX数据"""
        if os.path.exists(CONFIG.HISTORY_DATA_FILE):
            try:
                history_df = pd.read_csv(CONFIG.HISTORY_DATA_FILE)
                if 'vix' in history_df.columns:
                    return history_df['vix'].dropna().tolist()
            except:
                pass
        return []

    @staticmethod
    def save_results(trade_date: str, results: Dict) -> None:
        """保存结果到文件"""
        # 保存历史数据
        history_df = pd.DataFrame([{
            'date': trade_date,
            'vix': results['vix'],
            'iv_mean': results['iv_stats']['iv_mean'],
            'position': results['vix_analysis']['position']
        }])

        if os.path.exists(CONFIG.HISTORY_DATA_FILE):
            existing_df = pd.read_csv(CONFIG.HISTORY_DATA_FILE)
            history_df = pd.concat([existing_df, history_df]).drop_duplicates('date')

        history_df.to_csv(CONFIG.HISTORY_DATA_FILE, index=False)

        # 保存详细报告
        report_path = os.path.join(CONFIG.OUTPUT_DIR, f"300etf_report_{trade_date}.json")
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"💾 数据已保存至 {report_path}")

# -------------------------------
# 6. 主流程
# -------------------------------
class VIXMonitor:
    def __init__(self):
        self.processor = OptionDataProcessor()
        self.calculator = VIXCalculator()
        self.analyzer = RiskAnalyzer()
        self.data_manager = DataManager()

    def run(self, trade_date: str = None) -> Dict:
        """运行监控流程"""
        if not trade_date:
            trade_date = datetime.now().strftime("%Y%m%d")

        print(f"\n🚀 开始300ETF VIX监控 {trade_date}")

        # 1. 获取数据
        df = self.processor.fetch_risk_data(trade_date)
        if df is None:
            return {}

        # 2. 预处理
        df_300 = self.processor.preprocess_300etf_options(df)
        if df_300 is None:
            return {}

        # 3. 计算VIX和IV统计
        vix_value, vix_details = self.calculator.calculate_vix(df_300, trade_date)
        iv_stats = self.calculator.calculate_iv_stats(df_300)

        # 4. 风险分析
        greeks = self.analyzer.calculate_greeks_exposure(df_300)
        history_vix = self.data_manager.load_history()
        vix_analysis = self.analyzer.analyze_vix_position(vix_value, history_vix)

        # 5. 生成报告
        report = {
            'date': trade_date,
            'vix': round(vix_value, 2),
            'vix_details': vix_details,
            'iv_stats': iv_stats,
            'greeks': greeks,
            'vix_analysis': vix_analysis,
            'data_points': len(df_300)
        }

        # 6. 保存结果
        self.data_manager.save_results(trade_date, report)

        # 7. 打印摘要
        self.print_summary(report)

        return report

    @staticmethod
    def print_summary(report: Dict) -> None:
        """打印摘要报告"""
        print("\n📊 300ETF VIX监控摘要")
        print(f"📅 日期: {report['date']}")
        print(f"📈 VIX指数: {report['vix']:.2f} (计算方法: {report['vix_details']['method']})")
        print(f"📊 IV统计: 均值={report['iv_stats']['iv_mean']:.2f} 偏斜={report['iv_stats']['skew']:.2f}")

        if report['vix_analysis']['position'] != 'insufficient_data':
            print(f"📉 历史分位: {report['vix_analysis']['quantile']:.0%} ({report['vix_analysis']['position']})")
            print(f"💡 建议: {report['vix_analysis']['suggestion']}")

        print("\n🛡️ 风险敞口:")
        print(f"  Delta: {report['greeks'].get('delta', 0):.2f}")
        print(f"  Gamma: {report['greeks'].get('gamma', 0):.2f}")
        print(f"  Vega: {report['greeks'].get('vega', 0):.2f}")
        print(f"  Theta: {report['greeks'].get('theta', 0):.2f}")

# -------------------------------
# 7. 执行
# -------------------------------
if __name__ == "__main__":
    monitor = VIXMonitor()

    # 可以指定日期或使用当天
    monitor.run("20250801")
    # monitor.run()