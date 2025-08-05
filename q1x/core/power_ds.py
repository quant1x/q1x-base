# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import talib
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

from q1x.base import cache


class AShareControlDegreeAnalyzer:
    """
    A股专用控盘度分析框架
    功能特点：
    1. 完全适配A股市场特性
    2. 使用cache.klines()接口获取数据
    3. 支持动态阈值和自适应参数
    4. 包含完整的信号生成和回测引擎
    """

    def __init__(self):
        self.results: Dict[str, Dict] = {}
        self.signals: pd.DataFrame = pd.DataFrame()
        self.backtest_results: Dict = {}
        self.market_status: str = 'neutral'  # bull/bear/neutral

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """A股数据预处理"""
        df = df.copy()
        # 确保包含A股必要字段
        required_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
        assert all(col in df.columns for col in required_cols), f"数据缺少必要列: {required_cols}"

        # 处理停牌和异常值
        df['volume'] = df['volume'].replace(0, np.nan).fillna(method='ffill')
        df['amount'] = df['amount'].replace(0, np.nan).fillna(method='ffill')

        # 计算A股常用指标
        df['returns'] = df['close'].pct_change(fill_method=None)
        df['avg_price'] = df['amount'] / (df['volume'] * 100 + 1e-6)  # 计算均价(考虑A股单位)
        df['turnover'] = df['volume'] / (df['volume'].rolling(250, min_periods=20).mean() + 1e-6)
        return df.dropna(subset=['close', 'volume'])

    def _calculate_chip_indicators(self, df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
        """A股筹码指标计算"""
        results = pd.DataFrame(index=df.index)

        # 成本集中度 (适应A股高波动特性)
        price_range = df['high'].rolling(window).max() - df['low'].rolling(window).min()
        results['chip_concentration'] = 1 / (price_range / df['close'].rolling(window).mean() + 1e-6)

        # 获利比例 (基于均价计算)
        results['profit_ratio'] = (df['close'] - df['avg_price'].rolling(window).mean()) / (
                df['avg_price'].rolling(window).std() + 1e-6)

        # 大单活跃度 (A股主力常用手法)
        large_vol = df['volume'] > df['volume'].rolling(20).mean() * 1.5
        results['large_activity'] = (df['close'] - df['open']) * large_vol

        # 综合筹码分数
        results['chip_score'] = (
                0.4 * results['chip_concentration'].clip(-3, 3) +
                0.3 * results['profit_ratio'].clip(-3, 3) +
                0.3 * results['large_activity'].clip(-3, 3))
        return results

    def _calculate_vp_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """A股量价指标"""
        results = pd.DataFrame(index=df.index)

        # 量价背离 (适应A股涨跌停限制)
        price_up = df['close'] > df['open']
        vol_ratio = df['volume'] / df['volume'].shift(1)

        results['vp_divergence'] = np.where(
            price_up & (vol_ratio < 0.8), 1,  # 价升量缩
            np.where(~price_up & (vol_ratio > 1.2), -1, 0)  # 价跌量增
        )

        # 控盘系数 (考虑A股换手率特性)
        results['control_ratio'] = df['close'].pct_change(5) / (
                df['turnover'].rolling(5).mean() + 1e-6)

        return results

    def _composite_control_degree(self, df: pd.DataFrame,
                                  chip_score: pd.Series,
                                  vp_score: pd.Series) -> pd.Series:
        """综合控盘度计算 (A股优化版)"""
        # 加入流动性调整因子 (小盘股效应)
        market_cap_factor = np.log1p(df['amount'].rolling(20).mean()) / 10

        # 动态权重调整
        volatility = df['close'].pct_change().rolling(20).std()
        chip_weight = 0.5 - 0.3 * np.tanh(volatility * 100)  # 波动大时降低筹码权重
        vp_weight = 1 - chip_weight

        composite = (
                            chip_weight * chip_score +
                            vp_weight * vp_score
                    ) * market_cap_factor

        # 归一化到0-100
        return 50 + 50 * np.tanh(composite)

    def analyze_single_stock(self, code: str,
                             start_date: str = None,
                             end_date: str = None,
                             window: int = 60) -> Optional[pd.DataFrame]:
        """分析单只A股"""
        try:
            # 获取数据 (使用您的cache接口)
            klines = cache.klines(code, start_date=start_date, end_date=end_date)
            df = cache.convert_klines_trading(klines, period='d')
            if len(df) < 100:
                print(f"[{code}] 数据不足{len(df)}条，至少需要100个交易日数据")
                return None

            # 数据预处理
            df = self._preprocess_data(df)

            # 计算指标
            chip_score = self._calculate_chip_indicators(df, window)
            vp_score = self._calculate_vp_indicators(df)

            # 综合控盘度
            control_degree = self._composite_control_degree(df, chip_score['chip_score'], vp_score['vp_divergence'])

            # 动态阈值 (基于A股历史分位数)
            roll = control_degree.rolling(window)
            overbought = roll.quantile(0.85).fillna(method='bfill')  # A股阈值调低
            oversold = roll.quantile(0.15).fillna(method='bfill')

            # 生成信号
            signals = pd.DataFrame(index=df.index)
            signals['code'] = code
            signals['close'] = df['close']
            signals['control_degree'] = control_degree
            signals['overbought'] = overbought
            signals['oversold'] = oversold

            # 基础信号
            signals['buy_signal'] = (control_degree.shift(1) < oversold.shift(1)) & (
                    control_degree > oversold)
            signals['sell_signal'] = (control_degree.shift(1) > overbought.shift(1)) & (
                    control_degree < overbought)

            # 存储结果
            self.results[code] = {
                'data': df,
                'signals': signals,
                'control_degree': control_degree
            }

            return signals

        except Exception as e:
            print(f"[{code}] 分析失败: {str(e)}")
            return None

    def analyze_batch(self, stock_list: List[str],
                      start_date: str = '20100101',
                      end_date: str = None,
                      **kwargs) -> pd.DataFrame:
        """批量分析A股"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')

        all_signals = []
        for code in tqdm(stock_list, desc="分析A股"):
            try:
                signals = self.analyze_single_stock(
                    code=code,
                    start_date=start_date,
                    end_date=end_date,
                    window=kwargs.get('window', 60))

                if signals is not None:
                    all_signals.append(signals.reset_index())
            except Exception as e:
                print(f"[{code}] 处理失败: {str(e)}")
                continue

        if all_signals:
            self.signals = pd.concat(all_signals, ignore_index=True)
            if 'date' in self.signals.columns:
                self.signals['date'] = pd.to_datetime(self.signals['date'])
                self.signals.set_index('date', inplace=True)
        return self.signals

    def a_share_backtest(self,
                         initial_capital: float = 1000000,
                         position_ratio: float = 0.2,
                         stop_loss: float = 0.07,
                         take_profit: float = 0.15,
                         t_cost: float = 0.002) -> Dict:
        """A股专用回测引擎 (考虑涨跌停和交易成本)"""
        if self.signals.empty:
            raise ValueError("没有可回测的信号")

        trades = []
        portfolio = {
            'cash': initial_capital,
            'positions': {},
            'daily_value': [],
            'current_date': None
        }

        # 按日期排序处理
        dates = sorted(self.signals.index.unique())
        for date in dates:
            portfolio['current_date'] = date
            daily_data = self.signals.loc[date]
            if isinstance(daily_data, pd.Series):
                daily_data = pd.DataFrame([daily_data])

            # 处理持仓
            for code in list(portfolio['positions'].keys()):
                pos = portfolio['positions'][code]
                stock_data = self.results[code]['data']

                # 检查是否触发止损止盈
                current_close = stock_data.loc[date, 'close']
                ret = (current_close - pos['entry_price']) / pos['entry_price']

                if ret <= -stop_loss or ret >= take_profit:
                    # 平仓
                    exit_reason = '止损' if ret <= -stop_loss else '止盈'
                    portfolio['cash'] += pos['shares'] * current_close * (1 - t_cost)

                    trades.append({
                        '代码': code,
                        '买入日期': pos['entry_date'],
                        '卖出日期': date,
                        '持有天数': (date - pos['entry_date']).days,
                        '买入价': pos['entry_price'],
                        '卖出价': current_close,
                        '收益率': ret,
                        '退出原因': exit_reason
                    })
                    portfolio['positions'].pop(code)

            # 执行买入信号
            buy_signals = daily_data[daily_data['buy_signal']]
            for _, signal in buy_signals.iterrows():
                code = signal['code']

                # 检查是否已在持仓中
                if code in portfolio['positions']:
                    continue

                # 计算可买数量
                available_cash = portfolio['cash'] * position_ratio
                price = signal['close']
                shares = int(available_cash // (price * 100)) * 100  # A股100股整数倍

                if shares <= 0:
                    continue

                # 买入操作 (考虑交易成本)
                cost = shares * price * (1 + t_cost)
                if cost > portfolio['cash']:
                    continue

                portfolio['cash'] -= cost
                portfolio['positions'][code] = {
                    'entry_date': date,
                    'entry_price': price,
                    'shares': shares
                }

            # 计算当日净值
            position_value = sum(
                pos['shares'] * self.results[code]['data'].loc[date, 'close']
                for code, pos in portfolio['positions'].items()
            )
            portfolio['daily_value'].append({
                'date': date,
                'total': portfolio['cash'] + position_value,
                'cash': portfolio['cash'],
                'positions': position_value,
                'holdings': len(portfolio['positions'])
            })

        # 整理回测结果
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        daily_df = pd.DataFrame(portfolio['daily_value']).set_index('date')

        # 计算绩效指标
        if not trades_df.empty:
            win_rate = trades_df['收益率'] > 0
            avg_return = trades_df['收益率'].mean()
            annualized = (daily_df['total'].iloc[-1] / initial_capital) ** (252/len(daily_df)) - 1
            max_dd = (daily_df['total'] / daily_df['total'].cummax() - 1).min()
        else:
            win_rate = avg_return = annualized = max_dd = np.nan

        self.backtest_results = {
            '初始资金': initial_capital,
            '最终净值': daily_df['total'].iloc[-1],
            '总收益率': (daily_df['total'].iloc[-1] - initial_capital) / initial_capital,
            '年化收益率': annualized,
            '胜率': win_rate.mean() if not trades_df.empty else 0,
            '平均收益率': avg_return,
            '最大回撤': max_dd,
            '交易次数': len(trades_df),
            '交易明细': trades_df,
            '每日净值': daily_df
        }
        return self.backtest_results

    def plot_a_share_result(self, code: str):
        """A股分析结果可视化"""
        if code not in self.results:
            print(f"[{code}] 没有分析结果")
            return

        result = self.results[code]
        df = result['data']
        signals = result['signals']

        plt.figure(figsize=(14, 8))
        ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
        ax2 = plt.subplot2grid((4, 1), (3, 0), sharex=ax1)

        # K线图
        ax1.plot(df.index, df['close'], label='价格', color='#1f77b4', linewidth=1.5)

        # 标记买卖信号
        buy_dates = signals[signals['buy_signal']].index
        sell_dates = signals[signals['sell_signal']].index
        ax1.scatter(buy_dates, df.loc[buy_dates, 'close'],
                    color='red', marker='^', s=100, label='买入信号')
        ax1.scatter(sell_dates, df.loc[sell_dates, 'close'],
                    color='green', marker='v', s=100, label='卖出信号')

        ax1.set_title(f'A股 {code} 控盘度分析', fontsize=14)
        ax1.legend(loc='upper left')
        ax1.grid(True, linestyle='--', alpha=0.5)

        # 控盘度曲线
        ax2.plot(signals.index, signals['control_degree'],
                 label='控盘度', color='#ff7f0e')
        ax2.fill_between(signals.index, signals['overbought'],
                         signals['oversold'], color='gray', alpha=0.2)
        ax2.axhline(70, color='red', linestyle='--', alpha=0.5)
        ax2.axhline(30, color='green', linestyle='--', alpha=0.5)
        ax2.legend(loc='upper left')
        ax2.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.show()

    def export_a_share_results(self, output_dir: str = 'a_share_results'):
        """导出A股分析结果"""
        os.makedirs(output_dir, exist_ok=True)

        # 导出信号
        if not self.signals.empty:
            self.signals.to_csv(
                os.path.join(output_dir, '交易信号.csv'),
                encoding='gbk',  # 适应A股常用编码
                float_format='%.4f')

        # 导出回测结果
        if self.backtest_results:
            # 汇总统计
            summary = pd.Series({
                '初始资金': self.backtest_results['初始资金'],
                '最终净值': self.backtest_results['最终净值'],
                '总收益率': f"{self.backtest_results['总收益率']:.2%}",
                '年化收益率': f"{self.backtest_results['年化收益率']:.2%}",
                '胜率': f"{self.backtest_results['胜率']:.2%}",
                '最大回撤': f"{self.backtest_results['最大回撤']:.2%}",
                '交易次数': self.backtest_results['交易次数']
            })
            summary.to_csv(
                os.path.join(output_dir, '回测汇总.csv'),
                encoding='gbk', header=False)

            # 交易记录
            if not self.backtest_results['交易明细'].empty:
                self.backtest_results['交易明细'].to_csv(
                    os.path.join(output_dir, '交易明细.csv'),
                    index=False, encoding='gbk')

            # 每日净值
            if not self.backtest_results['每日净值'].empty:
                self.backtest_results['每日净值'].to_csv(
                    os.path.join(output_dir, '每日净值.csv'),
                    encoding='gbk', float_format='%.2f')

        print(f"✅ A股分析结果已导出到: {os.path.abspath(output_dir)}")

# 使用示例
if __name__ == "__main__":
    # 初始化分析器
    analyzer = AShareControlDegreeAnalyzer()

    # 定义要分析的A股代码列表
    a_share_stocks = [
        '600519',  # 贵州茅台
        '000858',  # 五粮液
        '601318',  # 中国平安
        '600036',  # 招商银行
    ]

    # 批量分析 (使用您的cache.klines接口)
    print("开始分析A股数据...")
    signals = analyzer.analyze_batch(
        stock_list=a_share_stocks,
        start_date='20200101',
        end_date='20231231',
        window=60
    )

    # 运行回测
    print("运行回测...")
    backtest_results = analyzer.a_share_backtest(
        initial_capital=1000000,
        position_ratio=0.3,
        stop_loss=0.08,
        take_profit=0.12
    )

    # 可视化茅台的分析结果
    analyzer.plot_a_share_result('600519')

    # 导出结果 (GBK编码适应A股环境)
    analyzer.export_a_share_results()

    # 打印回测摘要
    print("\n回测结果摘要:")
    print(f"初始资金: {backtest_results['初始资金']:,.2f}")
    print(f"最终净值: {backtest_results['最终净值']:,.2f}")
    print(f"总收益率: {backtest_results['总收益率']:.2%}")
    print(f"年化收益率: {backtest_results['年化收益率']:.2%}")
    print(f"胜率: {backtest_results['胜率']:.2%}")
    print(f"最大回撤: {backtest_results['最大回撤']:.2%}")