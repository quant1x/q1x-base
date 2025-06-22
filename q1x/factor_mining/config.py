import os
from pathlib import Path

# 路径配置
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
FACTOR_DIR = BASE_DIR / 'factors'
MODEL_DIR = BASE_DIR / 'models'
RESULT_DIR = BASE_DIR / 'results'

# 数据配置
DATA_FILE = 'stock_data.csv'
START_DATE = '2010-01-01'
END_DATE = '2023-12-31'

# 因子配置
TECHNICAL_WINDOWS = [5, 10, 20, 60]
QUANTILES = 5

# 模型配置
SELECT_K = 10
PCA_COMPONENTS = 0.95
CORR_THRESHOLD = 0.8

# 回测配置
INITIAL_CAPITAL = 1000000
COMMISSION = 0.0005