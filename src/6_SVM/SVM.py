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

#optuna
import optuna
from optuna.integration import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import warnings 
warnings.filterwarnings("ignore")

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


#################### LOAD DATA ######################
################ DEFINE CONSTANTS ###################
##### we will only predict for one stock here #######

nasdaq = pd.read_csv(data_root + "NASDAQ_100_Data_From_2010.csv", sep="\t")

c = 0.14 # this is for scaled apple stock so that the two classes have roughly the same amount of data points
window_size = 50
# train_ratio = 0.8316
train_ratio = 0.75 
val_ratio = 0.15

stock_name = "AAPL"
DROPOUT = 0.2

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
# the order of the features is extremely important for latter indexing so make sure
features = ['log_Volatility', 'log_Volume', 'log_Adj_Close']


# adding week, month and year for letting model search through time
stock = get_stock(nasdaq, features, stock_name=stock_name)
stock_new = stock.copy()
stock_log_adj = stock["log_Adj_Close"]
stock_log_adj_diff = stock_log_adj.shift(-1) - stock_log_adj
stock_new["log_Adj_Close"] = stock_log_adj_diff # tomorrow - today

X = stock_new.copy()


y = X["log_Adj_Close"].shift(-1)[:-2] # smoothing y here for alignment issues
X = X.iloc[:-2]


day_of_week = np.array(list(map(lambda date: date.day_of_week, X.index)))
day_of_month = np.array(list(map(lambda date: date.days_in_month, X.index)))
day_of_year = np.array(list(map(lambda date: date.day_of_year, X.index)))

X["day_of_week"] = day_of_week
X["day_of_month"] = day_of_month
X["day_of_year"] = day_of_year


scaler = MinMaxScaler(feature_range=(-1, 1))
X = scaler.fit_transform(X)
y = scaler.fit_transform(y.to_numpy().reshape(-1, 1))

# for binary classification!!
c = 0.14
y = binary_y(y, c)

#################### FOR SVC !!!!!!! ########################
scaler_ = StandardScaler()
X = scaler_.fit_transform(X)
y = scaler.fit_transform(y)


train_size = int(X.shape[0]*train_ratio)
val_size = int(X.shape[0]*val_ratio)

# print(train_size + val_size)
X_train, X_val, X_test = X[:train_size, :], X[train_size : train_size + val_size, :], X[train_size + val_size:, :]
y_train, y_val, y_test = y[:train_size], y[train_size : train_size + val_size], y[train_size + val_size:]


#################### tuning with Optuna #####################
def objective(trial):

    classifier_name = trial.suggest_categorical('classifier', ['SVC'])
    if classifier_name == 'SVC':
        kernel=trial.suggest_categorical('kernel',['rbf'])#,'poly','linear','sigmoid'])
        svc_c = trial.suggest_loguniform('C', 0.01, 1e2)
        # gamma=trial.suggest_categorical('gamma',['auto','scale'])
        classifier_obj = SVC(C=svc_c, kernel=kernel)#, gamma=gamma)

    classifier_obj.fit(X_train, y_train.squeeze())
    preds = classifier_obj.predict(X_val)
    # score = np.corrcoef(preds, y_val.squeeze())[0][1]
    score = accuracy_score(y_val,preds)

    return score


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
print(study.best_trial)
