import pandas as pd
import numpy as np
import datetime as datetime
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance
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

def ROI(df,n):
    m = len(df)
    arr = []
    for i in range(0,n):
        arr.append('N')
    for j in range(n,m):
        roi= (df.Close[j] - df.Close[j-n])/df.Close[j-n] #Equation for ROI
        arr.append(roi)
    return arr

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

def ud_pred(pred, actual):
    up_down = [1]
    for i in range(1,len(pred)):
        if pred[i] > actual[i-1]:
            up_down.append(1)
        else:
            up_down.append(0)
    return up_down

def ud(predict):
    up_down = [1]
    for i in range(1,len(predict)):
        if predict[i] > predict[i-1]:
            up_down.append(1)
        else:
            up_down.append(0)
    return up_down

# Scale back predictions
def scale_back(pred, close):
    predict = [close[0]]
    for i in range(1,len(pred)):
        predict.append(pred[i]+list(close)[i-1])
    return predict

def scale_back_pct(pred, close):
    close = list(close)
    predict = [close[0]]
    for i in range(1,len(pred)):
        predict.append(pred[i]*list(close)[i-1]+list(close)[i-1])
    return predict

def long_equity(prediction, close, commission = 0.00):
    long = [close[0]]
    for i in range(1,len(prediction)-1):
        if prediction[i] == 1:
            long.append(long[i-1] + close[i] - close[i-1] - close[i-1]*commission)
        else:
            long.append(long[i-1] + 0)
    return long

def short_equity(prediction, close, commission = 0.00):
    short = [close[0]]
    for i in range(1,len(prediction)-1):
        if prediction[i] == 0:
            short.append(short[i-1] + close[i-1] - close[i] - close[i-1]*commission)
        else:
            short.append(short[i-1] + 0)
    return short  

def total_equity(prediction, close, commission = 0.00):
    total = [close[0]]
    for i in range(1,len(prediction)-1):
        if prediction[i] == 1:
            total.append(total[i-1] + close[i] - close[i-1] - close[i-1]*commission)
        else:
            total.append(total[i-1] + close[i-1] - close[i] - close[i-1]*commission)
    return total

def percentage_returns(price):
    perc = [0]
    for i in range(1,len(price)):
        perc.append(perc[i-1] + ((price[i]-price[i-1])/price[0])*100)
    return perc

def percentage_returns_for_dd(price):
    perc = [0]
    for i in range(1,len(price)):
        perc.append(perc[i-1] + ((price[i]-price[i-1])/price[i-1])*100)
    return perc

def max_drawdown(equity):
    dd = 0
    drawdown = [0]
    for i in range(1,len(equity)):
        if equity[i] < equity[i-1]:
            dd += equity[i-1] - equity[i]
            drawdown.append(dd)
        else:
            drawdown.append(dd)
            dd = 0
    for i in range(1, len(drawdown)):
        if drawdown[i] != 0:
            drawdown[i] = drawdown[i] + drawdown[i-1]
        else:
            drawdown[i] = 0
    Drawdown = [i for i in drawdown if i != 0]
    return [max(Drawdown), np.mean(Drawdown), min(Drawdown)]

def number_of_trades(equity, updown_pred):
    count = 0
    for i in range(1,len(equity)-1):
        if (equity[i] != equity[i-1] and updown_pred[i] != updown_pred[i-1]):
            count += 1
    total_profit = float(equity[len(equity)-1] - equity[0])
    average_profit = total_profit/count
    return count 

def number_of_winning_long_trades(equity, updown_pred):
    long_results = []
    profit_loss = 0
    for i in range(1,len(equity)-1):
        if (equity[i] != equity[i-1] and updown_pred[i] == updown_pred[i-1] and updown_pred[i] == 1):
            profit_loss += equity[i] - equity[i-1]
        if (equity[i] != equity[i-1] and updown_pred[i] != updown_pred[i-1]):
            long_results.append(profit_loss)
            profit_loss = 0
    wins = [i for i in long_results if i > 0]
    losses = [i for i in long_results if i < 0]
    average_profit = np.mean(wins)
    best_profit = np.max(wins)
    worst_loss = np.min(losses)
    average_loss = np.mean(losses)
    profit_loss_ratio = np.abs(average_profit/average_loss)
    return [len(wins), average_profit, average_loss, profit_loss_ratio, best_profit, worst_loss]

def number_of_winning_short_trades(equity, updown_pred):
    long_results = []
    profit_loss = 0
    for i in range(1,len(equity)-1):
        if (equity[i] != equity[i-1] and updown_pred[i] == updown_pred[i-1] and updown_pred[i] == 0):
            profit_loss += equity[i] - equity[i-1]
        if (equity[i] != equity[i-1] and updown_pred[i] != updown_pred[i-1]):
            long_results.append(profit_loss)
            profit_loss = 0
    wins = [i for i in long_results if i > 0]
    losses = [i for i in long_results if i < 0]
    average_profit = np.mean(wins)
    best_profit = np.max(wins)
    worst_loss = np.min(losses)
    average_loss = np.mean(losses)
    profit_loss_ratio = np.abs(average_profit/average_loss)
    return [len(wins), average_profit, average_loss, profit_loss_ratio, best_profit, worst_loss]  

def long_short_market_time(updown_pred):
    long_days = 0
    short_days = 0
    for i in updown_pred:
        if i > 0:
            long_days += 1
        else:
            short_days += 1
    return [long_days, short_days]
    