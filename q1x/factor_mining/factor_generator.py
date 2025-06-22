import pandas as pd
import numpy as np
import talib
from itertools import combinations
from typing import Dict, List
from config import *
from utils import timer, save_object

class FactorGenerator:
    def __init__(self, data: Dict[str, pd.DataFrame]):
        self.data = data
        self.factors = {}

    @timer
    def generate_technical_factors(self) -> pd.DataFrame:
        """生成技术指标因子"""
        price_data = self.data['price']
        factors = pd.DataFrame(index=price_data.index)

        # 确保使用列名明确指定价格序列
        close_prices = price_data['close'].values  # 转换为numpy数组
        high_prices = self.data['other_data']['high'].values
        low_prices = self.data['other_data']['low'].values

        # 移动平均类
        for window in TECHNICAL_WINDOWS:
            factors[f'MA_{window}'] = price_data['close'].rolling(window).mean()
            factors[f'STD_{window}'] = price_data['close'].rolling(window).std()
            factors[f'RET_{window}'] = price_data['close'].pct_change(window)

        # 动量类指标（使用.values转换）
        factors['RSI_14'] = talib.RSI(close_prices, timeperiod=14)
        factors['MACD'], factors['MACD_signal'], _ = talib.MACD(close_prices)
        factors['ADX_14'] = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)

        # 波动率指标
        factors['ATR_14'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)

        # 确保结果存入字典并返回
        self.factors['technical'] = factors.dropna()
        return self.factors['technical']

    @timer
    def generate_cross_section_factors(self) -> pd.DataFrame:
        """生成横截面因子"""
        price_data = self.data['price']
        returns = self.data['returns']
        factors = pd.DataFrame(index=price_data.index)

        # 横截面排名
        for window in [5, 10, 20]:
            factors[f'rank_{window}'] = price_data.rolling(window).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1])

            # 相对强弱
            factors['rel_strength'] = price_data / price_data.rolling(20).mean()

            # 波动率排名
            factors['vol_rank'] = returns.rolling(20).std().rank(axis=1, pct=True)

            factors = factors.dropna()
            self.factors['cross_section'] = factors
        self.factors['cross_section'] = factors.dropna()
        return self.factors['cross_section']

    @timer
    def generate_interaction_factors(self) -> pd.DataFrame:
        """生成因子交互项"""
        if not self.factors:
            self.generate_technical_factors()
            self.generate_cross_section_factors()

        tech_factors = self.factors['technical']
        cross_factors = self.factors['cross_section']

        # 技术因子之间的交互
        tech_interactions = pd.DataFrame(index=tech_factors.index)
        interaction_columns = []
        for i, (f1, f2) in enumerate(combinations(tech_factors.columns, 2)):
            interaction_columns.append(pd.Series(tech_factors[f1] * tech_factors[f2], name=f'tech_int_{i}'))
        tech_interactions = pd.concat(interaction_columns, axis=1)

        # 横截面因子与技术因子的交互
        cross_interactions = pd.DataFrame(index=tech_factors.index)
        for tech_col in tech_factors.columns:
            for cross_col in cross_factors.columns:
                cross_interactions[f'cross_int_{tech_col[:5]}_{cross_col[:5]}'] = (
                        tech_factors[tech_col] * cross_factors[cross_col]
                )

        interactions = pd.concat([tech_interactions, cross_interactions], axis=1)
        self.factors['interactions'] = interactions.dropna()
        return self.factors['interactions']

    @timer
    def generate_all_factors(self) -> Dict[str, pd.DataFrame]:
        """生成所有因子"""
        self.generate_technical_factors()
        self.generate_cross_section_factors()
        self.generate_interaction_factors()
        return self.factors

    def save_factors(self, path=FACTOR_DIR):
        """保存所有因子"""
        if not self.factors:
            self.generate_all_factors()

        for factor_type, factor_data in self.factors.items():
            save_object(factor_data, path/f'{factor_type}_factors.pkl')