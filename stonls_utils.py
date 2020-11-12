import numpy as np
import pandas as pd
import tensorflow as tf

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    Xtrain = []
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=10, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.shuffle(shuffle_buffer)
    for X in ds:
      Xtrain.append(X.numpy())
    return  np.asarray(Xtrain)

    
def indicators(dataset):
  # Create 7 and 21 days Moving Average
    dataset['ma7'] = dataset['Close'].rolling(window=7).mean()
    dataset['ma21'] = dataset['Close'].rolling(window=21).mean()
    
    # Create MACD
    dataset['12ema'] = dataset.Close.ewm(span=12, adjust=False).mean()
    dataset['26ema'] = dataset.Close.ewm(span=26, adjust=False).mean()
    dataset['MACD'] = (dataset['12ema']-dataset['26ema'])

    # Create Bollinger Bands
    dataset['20std'] = dataset['Close'].rolling(window=20).std()
    dataset['upper_band'] = dataset['ma21'] + (dataset['20std']*2)
    dataset['lower_band'] = dataset['ma21'] - (dataset['20std']*2)
    
    # Create Exponential moving average
    dataset['ema'] = dataset['Close'].ewm(com=0.5).mean()
    
    return dataset

def get_sar(s, af=0.02, amax=0.2):
    high, low = s.High, s.Low

    # Starting values
    sig0, xpt0, af0 = True, high[0], af
    sar = [low[0] - (high - low).std()]

    for i in range(1, len(s)):
        sig1, xpt1, af1 = sig0, xpt0, af0

        lmin = min(low[i - 1], low[i])
        lmax = max(high[i - 1], high[i])

        if sig1:
            sig0 = low[i] > sar[-1]
            xpt0 = max(lmax, xpt1)
        else:
            sig0 = high[i] >= sar[-1]
            xpt0 = min(lmin, xpt1)

        if sig0 == sig1:
            sari = sar[-1] + (xpt1 - sar[-1])*af1
            af0 = min(amax, af1 + af)

            if sig0:
                af0 = af0 if xpt0 > xpt1 else af1
                sari = min(sari, lmin)
            else:
                af0 = af0 if xpt0 < xpt1 else af1
                sari = max(sari, lmax)
        else:
            af0 = af
            sari = xpt0

        sar.append(sari)

    return pd.Series(sar, index=s.index)

def rsiFunc(prices, n=14):
    deltas = np.diff(prices)
    seed = deltas[:n+1]
    up = seed[seed>=0].sum()/n
    down = -seed[seed<0].sum()/n
    rs = up/down
    rsi = np.zeros(prices.shape)
    rsi[:n] = 100. - 100./(1.+rs)

    for i in range(n, len(prices)):
        delta = deltas[i-1] # cause the diff is 1 shorter
        if delta>0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta
        up = (up*(n-1) + upval)/n
        down = (down*(n-1) + downval)/n
        rs = up/down
        rsi[i] = 100. - 100./(1.+rs)
    return rsi

def preprocess(df):
  dataset = df.filter(['High','Low','Open','Close','Volume'])
  dataset.reset_index()
  dataset_indicators = indicators(dataset)
  sar=get_sar(dataset)
  dataset_indicators['sar']=sar
  rsi= rsiFunc(dataset['Close'])
  dataset_indicators['rsi']=rsi
  return dataset_indicators
