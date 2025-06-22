import pandas as pd
import numpy as np
import empyrical as ep
import pyfolio as pf
from matplotlib import pyplot as plt

from config import *
from utils import timer

class Backtester:
    def __init__(self, factors: pd.DataFrame, returns: pd.DataFrame, prices: pd.DataFrame):
        self.factors = factors
        self.returns = returns
        self.prices = prices
        self.portfolio_results = {}

    @timer
    def quantile_backtest(self, factor_name: str, quantiles=QUANTILES) -> pd.DataFrame:
        """分位数回测"""
        # 按因子值分组
        factor = self.factors[factor_name]
        labels = [f'Q{i+1}' for i in range(quantiles)]
        groups = pd.qcut(factor, q=quantiles, labels=labels)

        # 计算每组收益率
        group_returns = self.returns.groupby(groups).mean().T
        group_returns.index = pd.to_datetime(group_returns.index)

        # 计算累计收益
        cum_returns = (1 + group_returns).cumprod()

        # 保存结果
        self.portfolio_results[factor_name] = {
            'group_returns': group_returns,
            'cum_returns': cum_returns
        }

        return group_returns

    @timer
    def long_short_backtest(self, factor_name: str) -> pd.Series:
        """多空组合回测"""
        if factor_name not in self.portfolio_results:
            self.quantile_backtest(factor_name)

        group_returns = self.portfolio_results[factor_name]['group_returns']
        long_short = group_returns['Q5'] - group_returns['Q1']

        # 计算绩效指标
        perf_stats = self.calculate_performance(long_short)

        self.portfolio_results[factor_name]['long_short'] = {
            'returns': long_short,
            'performance': perf_stats
        }

        return long_short

    @timer
    def calculate_performance(self, returns: pd.Series) -> pd.Series:
        """计算绩效指标"""
        stats = {
            'Annual Return': ep.annual_return(returns),
            'Cumulative Returns': ep.cum_returns_final(returns),
            'Annual Volatility': ep.annual_volatility(returns),
            'Sharpe Ratio': ep.sharpe_ratio(returns),
            'Max Drawdown': ep.max_drawdown(returns),
            'Calmar Ratio': ep.calmar_ratio(returns),
            'Sortino Ratio': ep.sortino_ratio(returns),
            'Omega Ratio': ep.omega_ratio(returns),
            'Tail Ratio': ep.tail_ratio(returns)
        }

        return pd.Series(stats)

    @timer
    def visualize_results(self, factor_name: str):
        """可视化回测结果"""
        if factor_name not in self.portfolio_results:
            self.quantile_backtest(factor_name)
            self.long_short_backtest(factor_name)

        # 绘制累计收益曲线
        cum_returns = self.portfolio_results[factor_name]['cum_returns']
        long_short = self.portfolio_results[factor_name]['long_short']['returns']

        plt.figure(figsize=(15, 7))
        cum_returns.plot()
        (1 + long_short).cumprod().plot(label='Long-Short', ls='--', color='black')
        plt.title(f'{factor_name} Factor Portfolio Performance')
        plt.ylabel('Cumulative Returns')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(RESULT_DIR/f'{factor_name}_backtest.png')
        plt.close()

        # 使用Pyfolio生成详细报告
        returns = self.portfolio_results[factor_name]['long_short']['returns']
        pf.create_returns_tear_sheet(returns, benchmark_rets=self.returns.mean(axis=1))
        plt.savefig(RESULT_DIR/f'{factor_name}_tear_sheet.png')
        plt.close()

    @timer
    def backtest_all_factors(self):
        """回测所有因子"""
        for factor in self.factors.columns:
            self.quantile_backtest(factor)
            self.long_short_backtest(factor)
            self.visualize_results(factor)

        return self.portfolio_results

    def save_backtest_results(self, path=RESULT_DIR):
        """保存回测结果"""
        for factor, results in self.portfolio_results.items():
            # 保存绩效指标
            if 'performance' in results['long_short']:
                results['long_short']['performance'].to_csv(path/f'{factor}_performance.csv')

            # 保存收益数据
            results['group_returns'].to_csv(path/f'{factor}_group_returns.csv')
            results['cum_returns'].to_csv(path/f'{factor}_cum_returns.csv')

            if 'long_short' in results:
                results['long_short']['returns'].to_csv(path/f'{factor}_long_short.csv')