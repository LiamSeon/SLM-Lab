import numpy as np
import modin.pandas as pd
from modin.config import Engine
from ..technical_analysis.utils import *
from ..technical_analysis.moving_average import *
Engine.put("ray")

def RSI(data:pd.Series, window:int=14) -> pd.Series:
    diff = get_diff(data, window)
    up = get_diff_up(diff, 2)
    down = get_diff_down(diff, 2)
    rsi = up / up + down
    return rsi

def RMI(data:pd.Series, window:int=14, window2:int=5) -> pd.Series:
    diff = get_diff(data, window)
    up = get_diff_up(diff, window2)
    down = get_diff_down(diff, window2)
    rmi = up / up + down
    return rmi

def fast_stochastic_oscillator(price:pd.Seires, window:int = 15) -> pd.Series:
    low = get_low(price, window =  window)
    high = get_high(price, window = window)
    return (price - low) / (high - low)

def slow_stochastic_oscillator(price:pd.Series, window:int = 20, pk:int = 5,  pd:int  = 3):
    fast_k = fast_stochastic_oscillator(price, window)
    slow_k = sma(fast_k, pk)
    return sma(slow_k, pd)


