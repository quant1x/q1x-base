import pandas as pd
import numpy as np
import joblib
import pickle
from pathlib import Path
from typing import Dict, Union
from config import *

def save_object(obj: object, path: Union[str, Path]) -> None:
    """保存对象到文件"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == '.pkl':
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
    elif path.suffix == '.joblib':
        joblib.dump(obj, path)
    else:
        raise ValueError("Unsupported file format")

def load_object(path: Union[str, Path]) -> object:
    """从文件加载对象"""
    path = Path(path)
    if path.suffix == '.pkl':
        with open(path, 'rb') as f:
            return pickle.load(f)
    elif path.suffix == '.joblib':
        return joblib.load(path)
    else:
        raise ValueError("Unsupported file format")

def timer(func):
    """计时装饰器"""
    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} executed in {end-start:.2f} seconds")
        return result
    return wrapper