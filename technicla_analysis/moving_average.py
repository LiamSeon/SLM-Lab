import numpy as np
import modin.pandas as pd
from modin.config import Engine
Engine.put("ray")

"""
- https://github.com/fmilthaler/FinQuant/blob/master/finquant/moving_average.py
- https://github.com/modin-project/modin 
"""

def sma(data:pd.Series, span:int) -> pd.Series:
    return data.rolling(window=span, center=False).mean()

def ema(data:pd.Series, span:int, alpha:int) -> pd.Series:
    return data.ewm(span=span, adjust=False, min_periods=span, alpha = alpha).mean()

def sma_std(data:pd.Series, span:int) -> pd.Series:
    return data.rolling(window=span, center=False).std()

def ema_std(data:pd.Series, span:int, alpha:int) -> pd.Series:
    return data.ewm(span=span, adjust=False, min_periods=span, alpha = alpha).std()

def macd(data:pd.Series, alpha:int, dif_span1:int = 12, dif_span2:int = 26, dea_span:int = 9) -> pd.Series:
    dif = (ema(data, span = dif_span1, alpha = alpha) - ema(data, span = dif_span2, alpha = alpha))/ data
    dea = ema(dif, span = dea_span, alpha = alpha) / data
    return 2 * (dif - dea)
