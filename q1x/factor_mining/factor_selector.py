from typing import Tuple, Dict

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from config import *
from utils import timer, save_object

class FactorSelector:
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y
        self.selected_factors = {}

    @timer
    def select_by_univariate(self, k=SELECT_K) -> pd.Index:
        """基于单变量测试选择因子"""
        selector = SelectKBest(score_func=f_regression, k=k)
        selector.fit(self.X, self.y)
        selected = self.X.columns[selector.get_support()]
        self.selected_factors['univariate'] = selected
        return selected

    @timer
    def select_by_mutual_info(self, k=SELECT_K) -> pd.Index:
        """基于互信息选择因子"""
        selector = SelectKBest(score_func=mutual_info_regression, k=k)
        selector.fit(self.X, self.y)
        selected = self.X.columns[selector.get_support()]
        self.selected_factors['mutual_info'] = selected
        return selected

    @timer
    def select_by_pca(self, n_components=PCA_COMPONENTS) -> Tuple[pd.DataFrame, np.ndarray]:
        """使用PCA降维"""
        pca = PCA(n_components=n_components)
        pca_factors = pca.fit_transform(self.X)
        columns = [f'PC_{i+1}' for i in range(pca_factors.shape[1])]
        pca_df = pd.DataFrame(pca_factors, index=self.X.index, columns=columns)

        self.selected_factors['pca'] = {
            'factors': pca_df,
            'explained_variance': pca.explained_variance_ratio_
        }
        return pca_df, pca.explained_variance_ratio_

    @timer
    def select_by_correlation(self, threshold=CORR_THRESHOLD) -> pd.DataFrame:
        """基于相关性去除冗余因子"""
        corr_matrix = self.X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        selected = self.X.drop(to_drop, axis=1)

        self.selected_factors['correlation'] = selected.columns
        return selected

    @timer
    def select_by_importance(self, n_estimators=100) -> pd.Index:
        """基于随机森林特征重要性选择因子"""
        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        rf.fit(self.X, self.y)

        importance = pd.Series(rf.feature_importances_, index=self.X.columns)
        selected = importance.nlargest(SELECT_K).index

        self.selected_factors['rf_importance'] = selected
        return selected

    @timer
    def select_all_methods(self) -> Dict[str, pd.Index]:
        """使用所有方法选择因子"""
        self.select_by_univariate()
        self.select_by_mutual_info()
        self.select_by_pca()
        self.select_by_correlation()
        self.select_by_importance()
        return self.selected_factors

    def get_consensus_factors(self, min_votes=3) -> pd.Index:
        """获取共识因子(被多数方法选中的因子)"""
        if not self.selected_factors:
            self.select_all_methods()

        # 收集所有被选中的因子
        all_selections = []
        for method, factors in self.selected_factors.items():
            if method != 'pca':  # PCA产生的是新因子，不参与投票
                if isinstance(factors, pd.Index):
                    all_selections.extend(factors.tolist())
                elif isinstance(factors, pd.DataFrame):
                    all_selections.extend(factors.columns.tolist())

        # 计算每个因子被选中的次数
        factor_votes = pd.Series(all_selections).value_counts()
        consensus = factor_votes[factor_votes >= min_votes].index

        return consensus

    def save_selected_factors(self, path=MODEL_DIR):
        """保存选择的因子"""
        if not self.selected_factors:
            self.select_all_methods()

        save_object(self.selected_factors, path/'selected_factors.pkl')

        # 保存共识因子
        consensus = self.get_consensus_factors()
        save_object(consensus, path/'consensus_factors.pkl')