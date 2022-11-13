import pandas as pd
import numpy as np
import datetime as datetime
import matplotlib.pyplot as plt
# from pandas_datareader import data as pdr
# import yfinance
import pandas_ta as ta

# Volume Weighted Average Price
def vwap(price, volume, period = 14):
    Vwap = []
    for i in range(len(price)):
        if i < period:
            Vwap.append(np.nan)
        else:
            low_range = i - period
            pvol = price[low_range:(i+1)]*volume[low_range:(i+1)]
            ind = np.sum(pvol)/np.sum(volume[low_range:(i+1)])
            Vwap.append(ind)
    return Vwap
    
# Function to classify up or down movement of price data, up = True, down = False

def up_down(price):
    up_down = [True]
    for i in range(1,len(price)):
        if price[i] > price[i-1]:
            up_down.append(True)
        else:
            up_down.append(False)
    return up_down

# Relative Strenght index

def rsi(price, period):
    delta = price.diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
    ema_up = up.ewm(com=period, adjust=False).mean()
    ema_down = down.ewm(com=period, adjust=False).mean()
    rs = 100 - (100/(1 + (ema_up/ema_down)))
    return rs

# Moving average

def MA(price, period):
    MA = []
    for i in range(len(price)):
        if i < period:
            MA.append(np.nan)
        else:
            mean = np.mean(price[(i-period):(i+1)])
            MA.append(mean)
    return MA

# Stocastic Oscillator

def Stochastic_Oscillator(price, period = 14):
    K = []
    for i in range(len(price)):
        if i < period:
            K.append(np.nan)
        else:
            low_range = i - period
            L = np.min(price[low_range:(i+1)])
            H = np.max(price[low_range:(i+1)])
            C = price[i]
            ind = ((C-L)/(H-L))*100
            K.append(ind)
    return K       

#Exponential Moving Average 
def exp_moving_average(price, period):
    ema= price.ewm(span=period, adjust=False).mean()
    return ema 

#MACD
def macd(df, price):
    res= df.ta.macd(close=price, fast=12, slow=26, signal=9, append=True)
    df.append(res)
    return res

#Bollinger Bands
def Bollinger(price, ma_period = 14, n_std = 2):
    UB = []
    LB = []
    for i in range(len(price)):
        if i < ma_period:
            UB.append(np.nan)
            LB.append(np.nan)
        else:
            lr = i - ma_period
            hr = i +1
            ub = MA(price, ma_period)[-1] + n_std*np.std(price[lr:hr])
            lb = MA(price, ma_period)[-1] - n_std*np.std(price[lr:hr])
            UB.append(ub)
            LB.append(lb)
    return UB,LB 
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
print('FInished!')