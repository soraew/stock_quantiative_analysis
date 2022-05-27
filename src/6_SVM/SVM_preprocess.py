data_root = "../../data/"
#stats stuff
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.api import SimpleExpSmoothing

# ML stuff
import numpy as np
from numpy.fft import *
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn import metrics
from sklearn.linear_model import Lasso
from sklearn.svm import SVC
import pandas as pd
import lightgbm as lgb


# DL stuff
from torch.autograd import Variable
from fastprogress import master_bar, progress_bar
import torch
import torch.nn as nn
from torch.utils.data import Dataset


# plotting
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (20, 8)



# basic stuff
import datetime
import io
import os
from os.path import join
import re
from collections import Counter
from tqdm import tqdm


########################################################################
################## this cell is dedicated to kaggle ####################
########################################################################

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

############## REINDEX FUNCTION AND PREPARE_STOCK FUNCTION ARE PRETTY MUCH SAME, HOWEVER, I PREFER THE PRIOR ##################
# for ARIMA or some shit    
def reindex(df):
    return df.reindex(pd.date_range(df.index[0], df.index[-1])).fillna(method="ffill")

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
#############################################################################################################################

def get_features(df, features):
    #rename Adj Close
    
    df.rename(columns={"Adj Close":"Adj_Close"}, inplace=True) 
    df["log_Volatility"] = np.log(df.High - df.Low + 1)
    df["log_Volume"] = np.log(df.Volume + 1) 
    df["log_Adj_Close"] = np.log(df["Adj_Close"] + 1)
    # df["day_of_week"] = np.array(list(map(lambda date: date.weekday(), df.index)))

    if 'Adj_Close' not in features:
        df.drop(columns=["Adj_Close"], inplace=True)
    # nasdaq["log_Adj_Close_diff"] = nasdaq["log_Adj_Close"].diff()

    df.drop(columns = ["Low", "High", "Close", "Open", "Name", "Volume"], inplace=True)
    # nasdaq = nasdaq[features]

    # nasdaq.dropna(inplace = True)
    return df

# this will return feature engineered stock dataframe
def get_stock(nasdaq, features, stock_name="AAPL"):
    nasdaq_c = date_index_nasdaq(nasdaq)
    stock = prepare_stock(nasdaq_c, nasdaq_c.index[0], nasdaq_c.index[-1], stock_name)
    stock = get_features(stock, features)
    stock.fillna("ffill", inplace=True)
    return stock

# plot heatmap for top stocks
def plot_attribute(nasdaq, using,feature="log_Adj_Close"):
    stocks = pd.DataFrame()
    for name in using:
        stocks[name] = get_stock(nasdaq, name)[feature]
    stocks.dropna(inplace=True)
    stocks.plot()
    plt.show()

####### In the 2 functions below, we are adding weekday and making dataframe/nparray to feed into LSTM however ###########
####### prob we could have done this in like get_stock or something #######
# the main difference between the two is , the prior is just adding weekday at the end,
# whereas the latter function is adding it to every stock
def get_train_df(nasdaq, using, features):
    df_features_arr = reindex(get_stock(nasdaq, using[0])).to_numpy().T
    for name in using[1:]:
        adding = reindex(get_stock(nasdaq, name)).to_numpy().T
        df_features_arr = np.concatenate([df_features_arr, adding])
    df_features_arr = df_features_arr.T

    ## df_features = pd.DataFrame(data=df_features_arr, columns=pd.MultiIndex.from_tuples(zip(col_one, col_two)))
    
    # making columns
    # features must not include weekday here
    if "weekday" in features:
        features.remove("weekday")
    col_one = []
    for element in using:
        for i in range(len(features)):
            col_one.append(element)
    col_two = list(features)*len(using)
    # print(len(col_one), len(col_two))

    # scaling 
    scaler = MinMaxScaler((-1, 1))
    scaled = scaler.fit_transform(df_features_arr)
    df_features = pd.DataFrame(data=scaled, columns=pd.MultiIndex.from_tuples(zip(col_one, col_two)))

    df_features.index = pd.date_range("2012-05-18", "2021-09-10")

    day_of_week = np.array(list(map(lambda date: date.weekday(), df_features.index)))
    day_of_week = day_of_week.reshape(-1, 1)
    day_of_week = pd.Series(data=scaler.fit_transform(day_of_week).reshape(-1,), index = df_features.index)
    df_features["weekday"] = day_of_week
    if "weekday" not in features:
        features.append("weekday")

    return df_features, features


# for feeding into network
# this doesn't work because of too many dimensions
def get_train_arr(nasdaq, using, features):
    df_features_arr = []
    for name in using:
        arr = reindex(get_stock(nasdaq, name)).to_numpy()
        # scaling for each column, for each stock_df in nasdaq
        scaler = MinMaxScaler(feature_range=(-1, 1))
        arr_scaled = scaler.fit_transform(arr)    

        # adding day of week
        day_of_week = np.array(list(map(lambda date: date.weekday(), pd.date_range("2012-05-18", "2021-09-10"))))
        day_of_week = day_of_week.reshape(-1, 1)
        day_of_week = scaler.fit_transform(day_of_week)
      
        arr_scaled = np.concatenate([arr_scaled, day_of_week], axis=1)

        df_features_arr.append(arr_scaled)


    df_features_arr = np.array(df_features_arr)
    if "weekday" not in features:
        features.append("weekday")
    df_features_arr = df_features_arr.reshape(-1, len(features), 7)

    return df_features_arr, features


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

def random_search_lgbm(X, y, params_space):
    params_log = {}
    iteration = 40
    cv = TimeSeriesSplit()
    for i in tqdm(range(iteration)):
        params = {}
        for key in params_space.keys():
            param_list = params_space[key]
            length = len(param_list)
            idx =np.random.randint(0,length) 
            params.update({key:param_list[idx]})
            # fit model to data
        
        model = lgb.LGBMRegressor(**params)
        for train_idx, test_idx, in cv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test).squeeze()
            coef_score = np.corrcoef(y_pred, y_test)[0][1]
            rmse_score = np.sqrt(mean_squared_error(y_pred, y_test))
        params_log.update({i:[coef_score, rmse_score, params]})

    sorted_by_coef=sorted(params_log.items(), key = lambda item: item[1][0], reverse=True)
    sorted_by_rmse=sorted(params_log.items(), key = lambda item: item[1][1])
    
    return sorted_by_coef, sorted_by_rmse


def binary_y(y_np, criterion):
    for i in range(len(y_np)):
        if y_np[i] > criterion:
            y_np[i] = int(1)
        else:
            y_np[i] = int(0)
        
    return y_np


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

def get_bc_per_stock_Xy(nasdaq, features, stock_name, window_size, train_ratio, val_ratio):
    stock = get_stock(nasdaq, features, stock_name)

    stock_new = stock.copy()
    stock_log_adj = stock["log_Adj_Close"]
    stock_log_adj_diff = stock_log_adj.shift(-1) - stock_log_adj
    stock_new["log_Adj_Close"] = stock_log_adj_diff # tomorrow - today
    X = stock_new
    y = stock_new["log_Adj_Close"].iloc[1:-1] #　そのまま, getting rid of nan although ind 0 is not nan for y
    X["log_Adj_Close"] = stock_new["log_Adj_Close"].shift(1) # today - yesterday
    X = X.iloc[1:-1] # getting rid of nans
    # new_index = y.index


    scaler = MinMaxScaler(feature_range=(-1, 1))
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y.to_numpy().reshape(-1,1))

    # to 1 and 0 two class classification



    # X[:, 2] = binary_y(X[:, 2], c) ######CHANGED FOR GBT_3##########
    y = binary_y(y, c)

    X, y = sliding_windows_single_feature(X, y, window_size)

    print("len >> ",len(X))

    train_size = int(X.shape[0]*train_ratio)
    val_size = int(X.shape[0]*val_ratio)
    X_train, X_val, X_test = X[:train_size, :, :], X[train_size:train_size + val_size, :, :], X[train_size + val_size:, :, :]
    y_train, y_val, y_test = y[:train_size], y[train_size:train_size + val_size], y[train_size + val_size:]
    
    return X_train, X_test, X_val, y_val, y_train, y_test
