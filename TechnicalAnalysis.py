import pandas as pd
import numpy as np

def SimpleMovingAverage(series, name, period, minPeriods = 0):
    if not name:
        name = str(period) + "Day_SimpleMovingAverage"
    return pd.Series(series.rolling(window=period,min_periods=minPeriods).mean(), index=[name])

def ExponentialMovingAverage(series, name, period, minPeriods = 0):
    if not name:
        name = str(period) + "Day_ExponentialMovingAverage"
    return pd.Series(series.ewm(span=period, min_periods=0).mean(), index=[name])

def DoubleExponentialMovingAverage(series, name, period):
    if not name:
        name = str(period) + "Day_DoubleExponentialMovingAverage"

    #DEMA = ( 2 * EMA(n)) - (EMA(EMA(n)) ), where n= period
    emaOfEma = pd.Series(series.ewm(period).mean())
    return pd.Series((2*series) - emaOfEma, index=[name])

def TripleExponentialMovingAverage(series, name, period):
    if not name:
        name = str(period) + "Day_TripleExponentialMovingAverage"

    #TEMA = 3 * EMA - 3 * EMA(EMA) + EMA(EMA(EMA))
    emaEma = pd.Series(series.ewm(period).mean())
    emaEmaEma = pd.Series(emaEma.ewm(period).mean())
    
    return pd.Series((3*series) - (3*emaEma) + emaEmaEma, index=[name])

def TriangularMovingAverage(series, name, period, minPeriods = 0):
    
    if not name:
        name = str(period) + "Day_TriangularMovingAverage"
    sma = SimpleMovingAverage(series, name, period, minPeriods)
    return SimpleMovingAverage(sma, name, period, minPeriods)

def CompareMovingAverage(shortMovingAverage, longMovingAverage, nameBuy, nameSell):
    buy = []
    sell = []
    buy.append(0)
    sell.append(0)
    for i in range (1, longMovingAverage.shape[0]-1, 1):
        if shortMovingAverage[i-1] <= longMovingAverage[i-1] and shortMovingAverage[i] > longMovingAverage[i]:
            buy.append(1)
            sell.append(0)
        elif shortMovingAverage[i-1] >= longMovingAverage[i-1] and shortMovingAverage[i] < longMovingAverage[i]:
            buy.append(0)
            sell.append(1)
        else:
            buy.append(0)
            sell.append(0)

    return pd.Series(np.array(buy), index=[nameSell]), pd.Series(np.array(sell), index=[nameBuy])

def GenerateAllMovingAverages(series, period):
    sma = SimpleMovingAverage(series, "", period)
    ema = ExponentialMovingAverage(series, "", period)
    dema = DoubleExponentialMovingAverage(series, "", period)
    tema = TripleExponentialMovingAverage(series, "", period)
    tma = TriangularMovingAverage(series, "", period)

    return sma, ema, dema, tema, tma

def MovingAverageConvergenceDivergence(series, name, fastPeriod = 12, slowPeriod = 26, averagePeriod = 9):
    if not name:
        name = str(fastPeriod) + "Day_" + str(slowPeriod) + "Day_MovingAverageConvergenceDivergence"
    slow = ExponentialMovingAverage(series, slowPeriod, slowPeriod)
    fast = ExponentialMovingAverage(series, fastPeriod, fastPeriod)
    diff = pd.Series(fast - slow)
    average = ExponentialMovingAverage(diff, "", averagePeriod)
    return pd.Series(fast - slow, average, index=[name, name + "Average"])
    
def DMPlusMinus(High, Low):
    DMPlus = []
    DMMinus = []
    DMPlus.append(0)
    DMMinus.append(0)
    
    for i in range(1, len(High), 1):
        highDiff = High[i] - High[i-1]
        lowDiff = Low[i] - Low[i-1]
        
        if (highDiff > lowDiff):
            if highDiff > 0:
                DMPlus.append(highDiff)
            else:
                DMPlus.append(0)
            DMMinus.append(0)
        elif (lowDiff > highDiff):
            if lowDiff > 0:
                DMMinus.append(lowDiff)
            else:
                DMMinus.append(0)
            DMPlus.append(0)
        else:
            DMPlus.append(0)
            DMMinus.append(0)
    return pd.Series(np.array(DMPlus)), pd.Series(np.array(DMMinus))

def TrueRange(high, low):
    return pd.Series(high-low)

def AverageTrueRange(trueRange, period):
    return SimpleMovingAverage(trueRange, "", period)

def PeriodDMPlusMinus(high, low, period):
    DmPlus, DmMinus = DMPlusMinus(high, low)
    return pd.Series(DmPlus.rolling(period).sum()), pd.Series(DmMinus.rolling(period).sum())

def PeriodTrueRange(high, low, period):
    trueRange = TrueRange(high, low)
    return pd.Series(trueRange.rolling(period).sum())

def PeriodDI(high, low, period):
    periodTrueRange = PeriodTrueRange(high, low, period)
    periodDmPlus, periodDmMinus = PeriodDMPlusMinus(high, low, period)
    return pd.Series(100*(periodDmPlus/periodTrueRange)), pd.Series(100*(periodDmMinus/periodTrueRange))

def DIDiffSum(high, low, period):
    diPlus, diMinus = PeriodDI(high, low, period)
    return pd.Series(abs(diPlus-diMinus)), pd.Series(diPlus + diMinus)

def DX(high, low, period):
    diDiff, diSum = DIDiffSum(high, low, period)
    return pd.Series(100*(diDiff/diSum))

def ADX(high, low, period):
    dx = DX(high, low, period)
    return pd.Series(dx.rolling(period).mean())

def ADXTrend(ADX):
    return pd.Series(ADX >= 25)

def DIBullBear(DIPlus, DIMinus):
    return pd.Series(DIPlus > DIMinus), pd.Series(DIMinus > DIPlus)

def MinMax(data, period):
    return pd.Series(data.rolling(period).max()), pd.Series(data.rolling(period).min())

def NewHighLow(data, mins, maxs):
    return pd.Series(data > maxs), pd.Series(data < mins)

def BBRange(UB, LB):
    return pd.Series(UB-LB)

def BBRangeDiff(shortRange, longRange):
    return pd.Series(shortRange, longRange)

def StdDev(data, periods):
    return pd.Series(pd.rolling_std(data, window = periods))

def BollingerBands(stdDev, ma, numberOfStdDev = 2):
    return pd.Series((stdDev * numberOfStdDev) + ma), pd.Series(ma - (stdDev * numberOfStdDev))

def RelativeStrengthIndex(Close, period):
    diff = Close.diff()
    which_dn = diff < 0

    up, dn = diff, diff*0
    up[which_dn], dn[which_dn] = 0, -up[which_dn]

    emaup = up.ewm(period).mean()
    emadn = dn.ewm(period).mean()

    rsi = 100 * emaup/(emaup + emadn)

    return pd.Series(rsi)

def RSIOverboughtOversold(rsi, overboughtLevel = 70, oversoldLevel = 30):
    return pd.Series(rsi > overboughtLevel), pd.Series(rsi < oversoldLevel)

def MoneyFlowIndex(high, low, close, volume, n, fillna):
    df = pd.DataFrame([high, low, close, volume]).T
    df.columns = ['High', 'Low', 'Close', 'Volume']
    df['Up_or_Down'] = 0
    df.loc[(df['Close'] > df['Close'].shift(1)), 'Up_or_Down'] = 1
    df.loc[(df['Close'] < df['Close'].shift(1)), 'Up_or_Down'] = 2

    # 1 typical price
    tp = (df['High'] + df['Low'] + df['Close']) / 3.0

    # 2 money flow
    mf = tp * df['Volume']

    # 3 positive and negative money flow with n periods
    df['1p_Positive_Money_Flow'] = 0.0
    df.loc[df['Up_or_Down'] == 1, '1p_Positive_Money_Flow'] = mf
    n_positive_mf = df['1p_Positive_Money_Flow'].rolling(n).sum()

    df['1p_Negative_Money_Flow'] = 0.0
    df.loc[df['Up_or_Down'] == 2, '1p_Negative_Money_Flow'] = mf
    n_negative_mf = df['1p_Negative_Money_Flow'].rolling(n).sum()

    # 4 money flow index
    mr = n_positive_mf / n_negative_mf
    mr = (100 - (100 / (1 + mr)))

    return pd.Series(mr)

def MFIOverboughtOversold(mfi, overboughtLevel = 80, oversoldLevel = 20):
    return pd.Series(mfi > overboughtLevel), pd.Series(mfi < oversoldLevel)

def TrueStrengthIndex(short, long, close):
    m = close - close.shift(1)
    m1 = m.ewm(long).mean().ewm(short).mean()
    m2 = abs(m).ewm(long).mean().ewm(short).mean()
    tsi = m1/m2
    tsi *= 100
    return pd.Series(tsi)