import numpy as np
import modin.pandas as pd
from modin.config import Engine
from ..preprocessing.utils import *
Engine.put("ray")


def bollinger_band(price, length = 20, num_stdev = 2):
    mean_price = price.rolling(length).mean()
    stdev = price.rolling(length).std()
    upband = mean_price + num_stdev*stdev
    dwnband = mean_price - num_stdev*stdev
    return upband,  dwnband
