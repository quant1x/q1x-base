import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t
from typing import Dict
from utils import timer, save_object  # 确保从您的utils模块导入timer装饰器
from config import *

class FactorEvaluator:
    def __init__(self, factors: Dict[str, pd.DataFrame], returns: pd.Series):
        self.factors = factors
        self.returns = returns
        self.evaluation_results = {}

    def _safe_spearman(self, x, y):
        """安全计算Spearman相关系数"""
        try:
            # 强制转换为 numpy 一维数组
            x = np.asarray(x).flatten()
            y = np.asarray(y).flatten()

            # 移除 NaN 值
            mask = ~np.isnan(x) & ~np.isnan(y)
            if mask.sum() < 3:
                return float('nan')

            x_valid = x[mask]
            y_valid = y[mask]

            # 检查是否为常量数组（会导致 ConstantInputWarning）
            if (x_valid == x_valid[0]).all() or (y_valid == y_valid[0]).all():
                return float('nan')

            # 正常计算 Spearman 相关系数
            return float(spearmanr(x_valid, y_valid)[0])
        except Exception:
            return float('nan')

    @timer
    def calculate_ic(self, factor_type: str) -> pd.DataFrame:
        """计算信息系数"""
        factor_data = self.factors[factor_type]
        ic_df = pd.DataFrame(index=factor_data.columns,
                             columns=['IC', 'IR', 'IC_pvalue'])

        # 对齐数据
        aligned_returns = self.returns.reindex(factor_data.index)

        # 强制转换为 Series（如果 returns 是多列，则取第一列）
        if isinstance(aligned_returns, pd.DataFrame):
            aligned_returns = aligned_returns.iloc[:, 0]

        for factor in factor_data.columns:
            # 整体IC
            valid_mask = factor_data[factor].notna() & aligned_returns.notna()
            ic = self._safe_spearman(
                factor_data[factor][valid_mask],
                aligned_returns[valid_mask]
            )

            # 滚动IC
            rolling_ic = factor_data[factor].rolling(20).apply(
                lambda x: self._safe_spearman(x, aligned_returns.loc[x.index]),
                raw=False
            )

            # 计算IR
            ir = rolling_ic.mean() / rolling_ic.std() if rolling_ic.notna().sum() > 0 else np.nan

            # IC显著性
            n = valid_mask.sum()
            if n >= 3 and not np.isnan(ic):
                t_stat = ic * np.sqrt(n - 2) / np.sqrt(1 - ic**2)
                p_value = 2 * (1 - t.cdf(abs(t_stat), df=n-2))
            else:
                p_value = np.nan

            ic_df.loc[factor] = [ic, ir, p_value]

        self.evaluation_results[f'{factor_type}_ic'] = ic_df
        return ic_df

    @timer
    def evaluate_all_factors(self, prices: pd.DataFrame = None) -> Dict[str, pd.DataFrame]:
        """评估所有因子"""
        for factor_type in self.factors.keys():
            self.calculate_ic(factor_type)
        return self.evaluation_results

    def save_evaluation_results(self, path=RESULT_DIR):
        # 保存共识因子
        consensus_factors = self.evaluate_all_factors()
        save_object(consensus_factors, path/'consensus_factors.pkl')