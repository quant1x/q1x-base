# control_degree_analyzer.py
import pandas as pd
import numpy as np
import talib
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from typing import Dict, List, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
import traceback

from q1x.base import cache


class SignalType(Enum):
    NEUTRAL = 0
    BUY = 1
    SELL = -1

@dataclass
class BacktestResult:
    total_trades: int
    win_rate: float
    avg_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float
    trades: pd.DataFrame

class EnhancedControlDegreeFramework:
    DEFAULT_PARAMS = {
        'window': 60,
        'overbought_quantile': 0.95,
        'oversold_quantile': 0.05,
        'min_interval': 10,
        'weights': {
            'chip_score': 0.3,
            'vp_score': 0.3,
            'main_capital_score': 0.2,
            'tech_score': 0.2
        }
    }

    def __init__(self, params: Optional[Dict] = None):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.results: Dict[str, Dict] = {}
        self.signals = pd.DataFrame()
        self.backtest_result: Optional[BacktestResult] = None

    def _safe_zscore(self, x: Union[pd.Series, np.ndarray]) -> float:
        """计算安全的Z-Score，兼容NumPy数组和Pandas Series"""
        if isinstance(x, pd.Series):
            x_values = x.values
        else:
            x_values = x

        if len(x_values) < 2:
            return 0.0
        mean = np.mean(x_values)
        std = np.std(x_values)
        return (x_values[-1] - mean) / (std + 1e-6)

    def _calculate_chip_metrics(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """计算筹码相关指标"""
        results = pd.DataFrame(index=df.index)

        # 确保数据列存在
        if not all(col in df.columns for col in ['high', 'low', 'close', 'volume']):
            raise ValueError("DataFrame缺少必要列: high, low, close, volume")

        # 1. 筹码集中度
        df['avg_price'] = (df['high'] + df['low'] + df['close']) / 3
        cost_mean = df['avg_price'].rolling(window, min_periods=1).mean()
        results['profit_ratio'] = (df['close'] - cost_mean) / (cost_mean + 1e-6)

        vol_ma_250 = df['volume'].rolling(250, min_periods=20).mean()
        turnover = df['volume'] / (vol_ma_250 + 1e-6)
        turnover_volatility = turnover.rolling(window, min_periods=1).std()
        results['chip_concentration'] = 1 / (turnover_volatility + 1e-6)

        price_range = (df['high'].rolling(window, min_periods=1).max() -
                       df['low'].rolling(window, min_periods=1).min() + 1e-6)
        chip_position = (df['close'] - df['low'].rolling(window, min_periods=1).min()) / price_range
        results['chip_position_adj'] = 1 - abs(chip_position - 0.5)

        results['chip_score'] = (
                (results['profit_ratio'] + 1).clip(lower=0.1) * 0.4 +
                results['chip_concentration'].clip(upper=10) * 0.3 +
                results['chip_position_adj'] * 0.3
        )

        return results

    def _calculate_vp_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算量价关系指标"""
        results = pd.DataFrame(index=df.index)

        # 2. 量价关系
        short_window, long_window = 5, 20
        price_ma_short = df['close'].rolling(short_window, min_periods=1).mean()
        price_ma_long = df['close'].rolling(long_window, min_periods=1).mean()
        price_trend = price_ma_short / (price_ma_long + 1e-6)

        vol_ma_short = df['volume'].rolling(short_window, min_periods=1).mean()
        vol_ma_long = df['volume'].rolling(long_window, min_periods=1).mean()
        volume_trend = vol_ma_short / (vol_ma_long + 1e-6)

        ret_5 = df['close'].pct_change(periods=5)
        turnover = df['volume'] / df['volume'].rolling(250, min_periods=20).mean()
        turnover_5 = turnover.rolling(5, min_periods=1).mean()
        control_coef = ret_5 / (turnover_5 + 1e-6)

        results['vp_score'] = np.where(
            (price_trend > 1) & (volume_trend < 1),
            control_coef,
            -abs(control_coef)
        )

        return results

    def _calculate_main_capital_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算主力资金指标"""
        results = pd.DataFrame(index=df.index)

        # 3. 主力资金活跃度
        vol_ma_20 = df['volume'].rolling(20, min_periods=1).mean()
        is_large = df['volume'] > vol_ma_20 * 0.8
        large_flow = np.where(
            df['close'] >= df['open'],
            df['volume'] * is_large,
            -df['volume'] * is_large
        )
        results['main_capital_score'] = pd.Series(large_flow).rolling(5, min_periods=1).sum()

        return results

    def _calculate_tech_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        results = pd.DataFrame(index=df.index)

        # 4. 技术形态
        upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20)
        boll_width = (upper - lower) / (middle + 1e-6)
        long_positive = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-6)

        results['tech_score'] = (
                (1 / (boll_width + 1e-6)) * 0.5 +
                long_positive.fillna(0) * 0.5
        )

        return results

    def _generate_signals(self,
                          df: pd.DataFrame,
                          control_degree: pd.Series,
                          oversold: pd.Series,
                          overbought: pd.Series,
                          min_interval: int) -> pd.DataFrame:
        """生成交易信号"""
        signal = pd.DataFrame(index=df.index)
        signal['prev_cd'] = control_degree.shift(1)

        # 信号判断
        signal['buy_signal'] = (signal['prev_cd'] <= oversold) & (control_degree > oversold)
        signal['sell_signal'] = (signal['prev_cd'] >= overbought) & (control_degree < overbought)

        # 冷却期处理
        signal['final_signal'] = SignalType.NEUTRAL.value
        last_signal_pos = None

        for i in range(1, len(signal)):
            if signal['buy_signal'].iloc[i]:
                if last_signal_pos is None or (i - last_signal_pos) >= min_interval:
                    signal.iloc[i, signal.columns.get_loc('final_signal')] = SignalType.BUY.value
                    last_signal_pos = i
            elif signal['sell_signal'].iloc[i]:
                if last_signal_pos is None or (i - last_signal_pos) >= min_interval:
                    signal.iloc[i, signal.columns.get_loc('final_signal')] = SignalType.SELL.value
                    last_signal_pos = i

        return signal

    def analyze_single_stock(self,
                             code: str,
                             name: Optional[str] = None,
                             data: Optional[pd.DataFrame] = None,
                             **kwargs) -> Optional[pd.DataFrame]:
        """
        分析单只股票

        Args:
            code: 股票代码
            name: 股票名称(可选)
            data: 股票数据(可选)
            **kwargs: 可覆盖实例参数

        Returns:
            包含信号的DataFrame或None(如果数据不足)
        """
        params = {**self.params, **kwargs}

        if data is None or len(data) < 100:
            print(f"[{code}] 数据不足，跳过")
            return None

        df = data.copy()
        results = pd.DataFrame(index=df.index)

        # 1. 计算各维度指标
        chip_results = self._calculate_chip_metrics(df, params['window'])
        vp_results = self._calculate_vp_metrics(df)
        main_results = self._calculate_main_capital_metrics(df)
        tech_results = self._calculate_tech_metrics(df)

        results = pd.concat([results, chip_results, vp_results, main_results, tech_results], axis=1)

        # 2. 计算综合控盘度
        final_score = pd.Series(0.0, index=df.index)
        for col, weight in params['weights'].items():
            if col not in results:
                continue
            raw_series = results[col].replace([np.inf, -np.inf], np.nan).fillna(0)
            z_series = raw_series.rolling(window=60, min_periods=10).apply(self._safe_zscore, raw=True)
            final_score += z_series * weight

        control_degree_raw = 50 + 50 * np.tanh(final_score)
        control_degree = control_degree_raw.fillna(50).clip(0, 100)

        # 3. 动态阈值
        roll = control_degree.rolling(params['window'], min_periods=20)
        oversold = roll.quantile(params['oversold_quantile']).bfill().fillna(40)
        overbought = roll.quantile(params['overbought_quantile']).bfill().fillna(80)

        # 4. 生成信号
        signal = self._generate_signals(df, control_degree, oversold, overbought, params['min_interval'])

        # 5. 整理结果
        signal['code'] = code
        signal['name'] = name or cache.stock_name(code) or code
        signal['close'] = df['close']
        signal['control_degree'] = control_degree
        signal['overbought'] = overbought.values
        signal['oversold'] = oversold.values

        # 保存结果
        self.results[code] = {
            'data': df,
            'control_degree': control_degree,
            'full_results': signal.copy()
        }

        return signal

    def _process_stock(self, item: Union[str, Tuple[str, str]]) -> Optional[pd.DataFrame]:
        """处理单只股票的辅助函数(用于并行处理)"""
        if isinstance(item, (list, tuple)):
            code, name = item
        else:
            code, name = item, None

        try:
            klines = cache.klines(code)
            df = cache.convert_klines_trading(klines, period='d')

            # 确保转换为DataFrame并设置索引
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)

            df.index = pd.to_datetime(df.index)
            if len(df) > 400:
                df = df.tail(400)

            # 检查必要列是否存在
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                print(f"[{code}] 数据缺少必要列: {required_cols}")
                return None

            return self.analyze_single_stock(code, name=name, data=df)
        except Exception as e:
            print(f"[{code}] 分析失败: {str(e)}")
            traceback.print_exc()
            return None

    def analyze_batch(self,
                      stock_list: List[Union[str, Tuple[str, str]]],
                      parallel: bool = True,
                      **kwargs) -> pd.DataFrame:
        """
        批量分析股票

        Args:
            stock_list: 股票列表(代码或(代码,名称)元组)
            parallel: 是否使用并行处理
            **kwargs: 可覆盖实例参数

        Returns:
            包含所有股票信号的DataFrame
        """
        params = {**self.params, **kwargs}
        signals = []

        if parallel:
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(self._process_stock, item) for item in stock_list]

                for future in tqdm(as_completed(futures), total=len(stock_list), desc="Analyzing Stocks"):
                    try:
                        sig = future.result()
                        if sig is not None:
                            signals.append(sig.reset_index())
                    except Exception as e:
                        print(f"处理股票时出错: {str(e)}")
                        traceback.print_exc()
        else:
            for item in tqdm(stock_list, desc="Analyzing Stocks"):
                try:
                    sig = self._process_stock(item)
                    if sig is not None:
                        signals.append(sig.reset_index())
                except Exception as e:
                    print(f"处理股票时出错: {str(e)}")
                    traceback.print_exc()

        if signals:
            self.signals = pd.concat(signals, ignore_index=True)
            self.signals['date'] = pd.to_datetime(self.signals['index'])
            self.signals.set_index('date', inplace=True)
        else:
            print("⚠️ 警告: 没有生成任何有效信号")

        return self.signals

    def backtest(self,
                 holding_days: int = 10,
                 transaction_cost: float = 0.001) -> BacktestResult:
        """
        执行回测

        Args:
            holding_days: 持有天数
            transaction_cost: 交易成本(百分比)

        Returns:
            BacktestResult对象
        """
        if self.signals.empty:
            raise ValueError("No signals to backtest")

        trades = []
        equity_curve = []
        prev_equity = 1.0

        for code in self.signals['code'].unique():
            stock_data = self.signals[self.signals['code'] == code].copy()
            buy_signals = stock_data[stock_data['final_signal'] == SignalType.BUY.value]

            for idx, row in buy_signals.iterrows():
                future = stock_data.loc[idx:].head(holding_days + 1)
                if len(future) < holding_days + 1:
                    continue

                entry_price = row['close']
                exit_price = future.iloc[-1]['close']
                gross_ret = (exit_price - entry_price) / entry_price
                net_ret = gross_ret - 2 * transaction_cost  # 买卖各一次交易成本

                trades.append({
                    'code': code,
                    'name': row['name'],
                    'entry_date': idx,
                    'exit_date': future.index[-1],
                    'holding_days': holding_days,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'gross_return': gross_ret,
                    'net_return': net_ret
                })

                # 更新权益曲线
                prev_equity *= (1 + net_ret)
                equity_curve.append({
                    'date': future.index[-1],
                    'equity': prev_equity
                })

        if not trades:
            return BacktestResult(
                total_trades=0,
                win_rate=np.nan,
                avg_return=np.nan,
                annualized_return=np.nan,
                max_drawdown=np.nan,
                sharpe_ratio=np.nan,
                trades=pd.DataFrame()
            )

        trades_df = pd.DataFrame(trades)
        equity_df = pd.DataFrame(equity_curve).set_index('date')

        # 计算各项指标
        win_rate = (trades_df['net_return'] > 0).mean()
        avg_return = trades_df['net_return'].mean()
        annualized = (1 + avg_return) ** (252 / holding_days) - 1

        # 计算最大回撤
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak']
        max_drawdown = equity_df['drawdown'].min()

        # 计算夏普比率
        daily_returns = equity_df['equity'].pct_change().dropna()
        sharpe_ratio = np.sqrt(252) * daily_returns.mean() / (daily_returns.std() + 1e-6)

        self.backtest_result = BacktestResult(
            total_trades=len(trades_df),
            win_rate=win_rate,
            avg_return=avg_return,
            annualized_return=annualized,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            trades=trades_df
        )

        return self.backtest_result

    def plot_stock(self, code: str, figsize: Tuple[int, int] = (14, 8)) -> None:
        """
        绘制股票分析结果

        Args:
            code: 股票代码
            figsize: 图表大小
        """
        if code not in self.results:
            print(f"[{code}] 没有分析结果")
            return

        result = self.results[code]
        df = result['data']
        signal = result['full_results']

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw={'height_ratios': [2, 1]})

        # 价格图表
        ax1.plot(df.index, df['close'], label='Price', color='black', linewidth=1.2)
        buys = signal[signal['final_signal'] == SignalType.BUY.value]
        sells = signal[signal['final_signal'] == SignalType.SELL.value]
        ax1.scatter(buys.index, df.loc[buys.index]['close'], color='green', label='Buy', marker='^', s=100, zorder=5)
        ax1.scatter(sells.index, df.loc[sells.index]['close'], color='red', label='Sell', marker='v', s=100, zorder=5)
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, axis='y', alpha=0.3)

        # 控盘度图表
        ax2.plot(signal.index, signal['control_degree'], label='Control Degree', color='blue')
        ax2.plot(signal.index, signal['overbought'], color='red', linestyle='--', label='Overbought')
        ax2.plot(signal.index, signal['oversold'], color='green', linestyle='--', label='Oversold')
        ax2.fill_between(signal.index, signal['control_degree'], signal['overbought'],
                         where=(signal['control_degree'] > signal['overbought']),
                         color='red', alpha=0.2)
        ax2.fill_between(signal.index, signal['control_degree'], signal['oversold'],
                         where=(signal['control_degree'] < signal['oversold']),
                         color='green', alpha=0.2)
        ax2.set_ylabel('Control Degree')
        ax2.legend()
        ax2.grid(True, axis='y', alpha=0.3)

        plt.suptitle(f'{code} - Control Degree Analysis (Dynamic Threshold)', fontsize=14)
        plt.xlabel('Date')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def export_results(self, output_dir: str = 'output') -> None:
        """
        导出分析结果

        Args:
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)

        # 导出信号
        self.signals.to_csv(f'{output_dir}/all_signals.csv', encoding='utf-8-sig')

        # 导出回测结果
        if self.backtest_result is not None:
            summary = {
                'total_trades': self.backtest_result.total_trades,
                'win_rate': self.backtest_result.win_rate,
                'avg_return': self.backtest_result.avg_return,
                'annualized_return': self.backtest_result.annualized_return,
                'max_drawdown': self.backtest_result.max_drawdown,
                'sharpe_ratio': self.backtest_result.sharpe_ratio
            }

            pd.Series(summary).to_json(f'{output_dir}/backtest_summary.json', force_ascii=False)
            self.backtest_result.trades.to_csv(f'{output_dir}/trades.csv', index=False, encoding='utf-8-sig')

        print(f"✅ 结果已导出到目录: {output_dir}")

if __name__ == "__main__":
    # 示例用法
    stock_list = ['sh000001', 'sz000158']

    try:
        framework = EnhancedControlDegreeFramework()
        signals = framework.analyze_batch(
            stock_list=stock_list,
            parallel=True
        )

        if not signals.empty:
            bt_result = framework.backtest(holding_days=10, transaction_cost=0.0005)
            print("📊 回测结果:", bt_result)

            for code in framework.results.keys():
                framework.plot_stock(code)

            framework.export_results()
        else:
            print("❌ 错误: 未能生成任何交易信号，请检查输入数据")
    except Exception as e:
        print(f"主程序出错: {str(e)}")
        traceback.print_exc()