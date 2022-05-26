data_root = "../../data/"
#stats stuff
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

# ML stuff
import numpy as np
from numpy.fft import *
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
import pandas as pd
import lightgbm as lgb


# DL stuff
from torch.autograd import Variable
from fastprogress import master_bar, progress_bar
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# plotting
import matplotlib.pyplot as plt
import seaborn as sns



# basic stuff
import datetime
import io
import os
from os.path import join
from collections import Counter
from tqdm import tqdm
import gc


# set index as datetime
def date_index_nasdaq(nasdaq):
    nasdaq_c = nasdaq.copy()
    dates = pd.to_datetime(nasdaq_c.Date)
    nasdaq_c.set_index(dates, inplace=True)
    # set date as index
    nasdaq_c.drop("Date", axis=1, inplace=True)
    # ここでFBとかTESLAとかに合わせている
    nasdaq_c = nasdaq_c["2012-05-18":]
    return nasdaq_c

################### PREPARE STOCK FROOM DATAFRAME DIRECTLY TAKEN FROM CSV FILE #######################
# for prepare_stock
def date_range_df(start, end, column_name = "Time"):
    date_range = pd.date_range(start, end)
    df = pd.DataFrame(date_range, columns = [column_name])
    df.set_index(column_name, inplace=True)
    return df

# merging with date range df
def prepare_stock(nasdaq, start, end, stock_name="AAPL", drop=True):
    nasdaq = nasdaq.loc[nasdaq["Name"]==stock_name]
    dates = date_range_df(start, end)
    new_nasdaq = dates.merge(nasdaq, how="left", left_index=True, right_index=True)
    if drop:
        new_nasdaq.dropna(inplace=True)
    return new_nasdaq
######################################################################################################

# create log_Volatility, log_Volume, log_Adj_Close and drop Adj_Close if not included in features
def get_features(df, features):
    #rename Adj Close
    df.rename(columns={"Adj Close":"Adj_Close"}, inplace=True) 
    df["log_Volatility"] = np.log(df.High - df.Low + 1)
    df["log_Volume"] = np.log(df.Volume + 1) 
    df["log_Adj_Close"] = np.log(df["Adj_Close"] + 1)
    # df["day_of_week"] = np.array(list(map(lambda date: date.weekday(), df.index)))

    if 'Adj_Close' not in features:
        df.drop(columns=["Adj_Close"], inplace=True)

    df.drop(columns = ["Low", "High", "Close", "Open", "Name", "Volume"], inplace=True)

    return df

# this will return feature engineered stock dataframe
def get_stock(nasdaq, features, stock_name="AAPL"):
    nasdaq_c = date_index_nasdaq(nasdaq)
    stock = prepare_stock(nasdaq_c, nasdaq_c.index[0], nasdaq_c.index[-1], stock_name)
    stock = get_features(stock, features)
    stock.fillna("ffill", inplace=True)
    return stock


# get features with sliding window
def sliding_windows_mutli_features(data, seq_length, target_cols_ids):
    x = []
    y = []

    for i in range((data.shape[0])-seq_length-1):
        #change here after finishing feature engineering process
        _x = data[i:(i+seq_length), :] 
        _y = data[i+seq_length, target_cols_ids] ## column 1 contains the labbel(log_Adj_Close)
        x.append(_x)
        y.append(_y)

    return np.array(x), np.array(y)

# sliding windows for one feature
def sliding_windows_single_feature(X, y, seq_length):
    x = []
    Y = []

    for i in range((X.shape[0])-seq_length-1):
        #change here after finishing feature engineering process
        _x = X[i:(i+seq_length), :] 
        _y = y[i+seq_length] ## column 1 contains the labbel(log_Adj_Close)
        x.append(_x)
        Y.append(_y)

    return np.array(x), np.array(Y)

# get predictor and target variable in np arrays
def get_Xy(df, window_size):
    log_adj_close_cols_ids = []
    volatility_cols_ids = []
    volume_cols_ids = []
    weekday_col_id = []
    count = 0
    for col in df.columns:
        # print(col)
        if col[1] == "Adj_Close":
            df.drop(col, axis=1, inplace=True)
            count -= 1
        if col[1] == "log_Adj_Close":
            log_adj_close_cols_ids.append(count)
        if col[1] == "log_Volume":
            volume_cols_ids.append(count)
        if col[1] == "log_Volatility":
            volatility_cols_ids.append(count)
        if col[0] == "weekday":
            weekday_col_id.append(count)
        count += 1
    df = df.to_numpy()
    x, y = sliding_windows_mutli_features(df, window_size, log_adj_close_cols_ids)

    # x.shape, y.shape
    return x, y

def get_train_test(x, y, train_ratio):
    train_size = int(len(y)*train_ratio)
    test_size = len(y) - train_size

    dataX = Variable(torch.Tensor(np.array(x)))
    dataY = Variable(torch.Tensor(np.array(y)))

    trainX = Variable(torch.Tensor(np.array(x[0:train_size])))
    trainY = Variable(torch.Tensor(np.array(y[0:train_size])))

    testX = Variable(torch.Tensor(np.array(x[train_size:len(x)])))
    testY = Variable(torch.Tensor(np.array(y[train_size:len(y)])))

    return trainX, trainY, testX, testY


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def binary_y(y_np, criterion):
    for i in range(len(y_np)):
        if y_np[i] > criterion:
            y_np[i] = int(1)
        else:
            y_np[i] = int(0)
    return y_np


# get binary class for stocks 
def get_bc_per_stock_Xy(nasdaq, features, stock_name, train_ratio):
    stock = get_stock(nasdaq, features, stock_name)

    stock_new = stock.copy()
    stock_log_adj = stock["log_Adj_Close"]
    stock_log_adj_diff = stock_log_adj.shift(-1) - stock_log_adj
    stock_new["log_Adj_Close"] = stock_log_adj_diff # tomorrow - today
    X = stock_new
    y = stock_new["log_Adj_Close"].iloc[1:-1] #　そのまま, getting rid of nan although ind 0 is not nan for y
    X["log_Adj_Close"] = stock_new["log_Adj_Close"].shift(1) # today - yesterday
    X = X.iloc[1:-1] # getting rid of nans
    new_index = y.index

    scaler = MinMaxScaler(feature_range=(-1, 1))
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y.to_numpy().reshape(-1,1))

    # to 1 and 0 two class classification
    X[:, 2] = binary_y(X[:, 2], c)
    y = binary_y(y, c)

    X, y = sliding_windows_single_feature(X, y, 50)

    train_size = int(X.shape[0]*train_ratio)
    
    X_train, X_test = X[:train_size, :, :], X[train_size:, :, :]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, X_test, y_train, y_test
