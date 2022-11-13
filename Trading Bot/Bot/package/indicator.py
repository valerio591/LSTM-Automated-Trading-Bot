import numpy as np
from sklearn.preprocessing import MinMaxScaler

# calculate momentum for each day
# 5-day momentum
def momentum(df):
    n = len(df)
    arr = []
    for i in range(0,5):
        arr.append('N')
    for j in range(5,n):
        momentum = df.Close[j] - df.Close[j-5] #Equation for momentum
        arr.append(momentum)
    return arr

#ROI function
def ROI(df,n):
    m = len(df)
    arr = []
    for i in range(0,n):
        arr.append('N')
    for j in range(n,m):
        roi= (df.Close[j] - df.Close[j-n])/df.Close[j-n] #Equation for ROI
        arr.append(roi)
    return arr

# calculate RSI for each day
def RSI(df,period):
    # get average of upwards of last 14 days: Ct - Ct-1
    # get average of downwards of last 14 days: Ct-1 - Ct
    n = len(df)
    arr = []
    for i in range(0,period):
        arr.append('N')
    for j in range(period,n):
        total_upwards = 0
        total_downwards = 0
        # this will find average of upwards
        for k in range(j,j-period,-1):
            if(df.Close[k-1] > df.Close[k]):
                total_downwards = total_downwards + (df.Close[k-1] - df.Close[k])    
        avg_down = total_downwards / period
        for l in range(j,j-period,-1):
            if(df.Close[l] > df.Close[l-1]):
                total_upwards = total_upwards + (df.Close[l] - df.Close[l-1])
        avg_up = total_upwards / period
        RS = avg_up / avg_down
        RSI  = 100 - (100/(1+RS))
        arr.append(RSI)
    return arr

# calculate EMA for each day
# formula: EMA = (2/(n+1))*ClosePrice + (1-(2/(n+1)))*previousEMA
def EMA(df, n):
    m = len(df)
    arr = []
    arr.append('N')
    prevEMA = df.Close[0]
    for i in range(1,m):
        close = df.Close[i]
        EMA = ((2/(n+1))*close) + ((1-(2/(n+1)))*prevEMA)
        arr.append(EMA)
        prevEMA = EMA
    return arr

#Function to Classify each day as a 1 or a 0
def clas(df, value):
    n = len(df)
    arr = []
    for i in range(0,len(df)-1):
        if (100*((df.Close[i+1]-df.Open[i+1])/df.Open[i+1]))>=value:
            arr.append(1)
        else:
            arr.append(0)
    arr.append('N')
    return arr

#MACD
# Moving Average of EMA(n) - EMA(m2) for each row
# where n = 12 and m2 = 26
def MACD(df):
    n = 12
    m2 = 26
    arr = []
    arr.append('N')
    ema_12 = EMA(df,n)
    ema_26 = EMA(df,m2)
    m = len(df)
    for i in range(1,m):
        arr.append(ema_12[i] - ema_26[i])
    return arr

#SRSI: Stochastic RSI
#SRSI = (RSI_today - min(RSI_past_n)) / (max(RSI_past_n) - min(RSI_past_n))
def SRSI(df,n):
    m = len(df)
    arr = []
    list_RSI = RSI(df,n)
    for i in range(0,n):
        arr.append('N')
    for j in range(n,n+n):
        last_n = list_RSI[n:j]
        if(not(last_n == []) and not(max(last_n) == min(last_n))):
            SRSI = (list_RSI[j] - min(last_n)) / (max(last_n)- min(last_n))
            if SRSI > 1:
                arr.append(1)
            else:
                arr.append(SRSI)
        else:
            arr.append(0)
    for j in range(n+n,m):
        last_n = list_RSI[2*n:j]
        if(not(last_n == []) and not(max(last_n) == min(last_n))):
            SRSI = (list_RSI[j] - min(last_n)) / (max(last_n)- min(last_n))
            if SRSI > 1:
                arr.append(1)
            else:
                arr.append(SRSI)
        else:
            arr.append(0)
    return arr

# calculate Williams %R oscillator for each day
def Williams(df,n):
    m = len(df)
    arr = []
    for i in range(0,n-1):
        arr.append('N')
    for j in range(n-1,m):
        maximum = max(df.High[(j-n+1):j+1])
        minimum = min(df.Low[(j-n+1):j+1])
        val = (-100)*(maximum-df.Close[j])/(maximum-minimum)
        arr.append(val)
    return arr

# True Range
# TR = MAX(high[today] - close[yesterday]) - MIN(low[today] - close[yesterday])
def TR(df,n):
    high = df.High[n]
    low = df.Low[n]
    close = df.Close[n-1]
    l_max = list()
    l_max.append(high)
    l_max.append(close)
    l_min = list()
    l_min.append(low)
    l_min.append(close)
    return (max(l_max) - min(l_min))

# Average True Range
# Same as EMA except use TR in lieu of close (prevEMA = TR(dataframe,14days))
def ATR(df,n):
    m = len(df)
    arr = []
    prevEMA = TR(df,n+1)
    for i in range(0,n):
        arr.append('N')
    for j in range(n,m):
        TR_ = TR(df,j)
        EMA = ((2/(n+1))*TR_) + ((1-(2/(n+1)))*prevEMA)
        arr.append(EMA)
        prevEMA = EMA
    return arr

def CCI(df,n):
    m = len(df)
    arr = []
    tparr = []
    for i in range(0,n-1):
        arr.append('N')
        tp = (df.High[i]+df.Low[i]+df.Close[i])/3
        tparr.append(tp)
    for j in range(n-1,m):
        tp = (df.High[j]+df.Low[j]+df.Close[j])/3
        tparr.append(tp) 
        tps = np.array(tparr[(j-n+1):(j+1)])
        val = (tp-tps.mean())/(0.015*tps.std())
        arr.append(val)
    return arr

def normalize(data):
    df = data
    for column in df:
        df[column]=((df[column]-df[column].mean())/df[column].std())
    return df





