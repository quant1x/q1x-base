from typing import Dict

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from config import *
from utils import timer

class DataProcessor:
    def __init__(self, data_path=DATA_DIR/DATA_FILE):
        self.data_path = data_path
        self.raw_data = None
        self.processed_data = None

    @timer
    def load_data(self) -> pd.DataFrame:
        """加载原始数据"""
        self.raw_data = pd.read_csv(self.data_path, parse_dates=['date'], index_col='date')
        self.raw_data = self.raw_data.loc[START_DATE:END_DATE]
        self.raw_data.columns = self.raw_data.columns.str.strip().str.lower()
        return self.raw_data

    @timer
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """数据清洗"""
        # 前向填充缺失值
        data = data.ffill().bfill()

        # 去除极端值
        for col in data.columns:
            if data[col].dtype in [np.float64, np.int64]:
                q_low = data[col].quantile(0.01)
                q_high = data[col].quantile(0.99)
                data[col] = data[col].clip(lower=q_low, upper=q_high)

        return data

    @timer
    def normalize_data(self, data: pd.DataFrame, method='standard') -> pd.DataFrame:
        """数据标准化"""
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError("Unsupported normalization method")

        normalized = pd.DataFrame(
            scaler.fit_transform(data),
            index=data.index,
            columns=data.columns
        )
        return normalized

    @timer
    def calculate_returns(self, price_data: pd.DataFrame, periods=1) -> pd.DataFrame:
        """计算收益率"""
        returns = price_data.pct_change(periods=periods).dropna()
        return returns

    @timer
    def process_pipeline(self) -> Dict[str, pd.DataFrame]:
        """完整数据处理流程"""
        # 加载数据
        raw_data = self.load_data()
        raw_data = raw_data[['open', 'high', 'low', 'close', 'volume','amount']]

        # 分离价格数据和其他数据
        price_cols = [col for col in raw_data.columns if 'price' in col.lower() or 'close' in col.lower()]
        other_cols = [col for col in raw_data.columns if col not in price_cols]

        price_data = raw_data[price_cols]
        other_data = raw_data[other_cols]

        # 数据清洗
        cleaned_price = self.clean_data(price_data)
        cleaned_other = self.clean_data(other_data)

        # 标准化
        norm_price = self.normalize_data(cleaned_price)
        norm_other = self.normalize_data(cleaned_other)

        # 计算收益率
        returns = self.calculate_returns(cleaned_price)

        # 保存处理后的数据
        self.processed_data = {
            'price': cleaned_price,
            'normalized_price': norm_price,
            'other_data': cleaned_other,
            'normalized_other': norm_other,
            'returns': returns
        }

        return self.processed_data

    def save_processed_data(self, path=DATA_DIR/'processed_data.parquet'):
        """保存处理后的数据"""
        if self.processed_data is None:
            self.process_pipeline()

        # 合并所有数据
        all_data = pd.concat([
            self.processed_data['price'],
            self.processed_data['normalized_price'],
            self.processed_data['other_data'],
            self.processed_data['normalized_other'],
            self.processed_data['returns']
        ], axis=1, keys=['price', 'norm_price', 'other', 'norm_other', 'returns'])

        all_data.to_parquet(path)