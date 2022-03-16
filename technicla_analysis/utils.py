import numpy as np
import modin.pandas as pd
from modin.config import Engine
Engine.put("ray")


def get_diff(data:pd.Series, window:int = 14) -> pd.Series:
    diff = data.rolling(window = window , center = False).apply(np.diff)
    diff[diff.isna()] = 0
    return diff

def get_diff_up(diff:pd.Series, window:int = 2) -> pd.Series:
    diff_up = diff.copy()
    diff_up[diff_up < 0] = 0
    return diff_up.rolling(window = window, center = False).sum()

def get_diff_down(diff:pd.Series, window:int = 2) -> pd.Series:
    diff_down = diff.copy()
    diff_down[diff_down > 0] = 0
    return diff_down.rolling(window = window, center = False).sum()

def get_low(diff:pd.Series, window:int = 2) -> pd.Series:
    temp = diff.copy()
    return temp.rolling(window = window, center = False).min()

def get_high(diff:pd.Series, window:int = 2) -> pd.Series:
    temp = diff.copy()
    return temp.rolling(window = window, center = False).max()