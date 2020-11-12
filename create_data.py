import yfinance as yf
import argparse
import pandas as pd
import datetime
from datetime import date
from stonls_utils import preprocess, windowed_dataset
from model.py import predictor

arg_parser = argparse.ArgumentParser("Enter the tickers")

arg_parser.add_argument("--first", type=object, help="first value", default='')
arg_parser.add_argument("--second", type=object, help="second value", default='')
arg_parser.add_argument("--third", type=object, help="third value", default='')
args = arg_parser.parse_args()    
tickers = [args.first, args.second, args.third]

def creation():
    inference =[]
    try:
        start = date.today()
        d = datetime.timedelta(days = 100)
        a = start-d
        start = start.strftime('%Y-%m-%d')
        a = a.strftime('%Y-%m-%d')
        for i in tickers:
            data = yf.download(i, start=a, end=start)
            train = preprocess(data)
            train = windowed_dataset(train.values, 60,32,100)
            model = predictor('------------')
            inference.append(i,(model.predict(train, batch_size = 32)))
    except:
        print('The ticker {} does exist'.format(i))
    finally:
        return inference

    
