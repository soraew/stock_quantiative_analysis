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


#relative import preprocessing
from GBDT_preprocess import *



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


################################
# TUNE HYPERPARAMS WITH OPTUNA #
################################
lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_val = lgb.Dataset(X_val, label=y_val)

def objective(trial):
    params = {
        'objective': 'cross_entropy',
        'metric': 'auc',
        'boosting': 'gbdt',
        'learning_rate': 0.05,
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        'num_leaves': trial.suggest_int('num_leaves', 2, 512),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 0, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'seed': 0,
        'verbosity': -1,
    }

    gbm = lgb.train(params, lgb_train, valid_sets=lgb_val,\
                                verbose_eval=False, num_boost_round=1000, early_stopping_rounds=100)
            
    preds = gbm.predict(X_val)

    # score = np.corrcoef(preds, y_val)[0][1]
    score = accuracy_score(y_val, preds)
    # score = roc_auc_score(preds, y_val.squeeze())
    print("SCORE :>>",score)

    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=1)


model = LGBMClassifier(study.best_params)
model.fit(X_train, y_train)
preds = model.predict(X_test)
print(f"corr:{np.corrcoef(y_test, preds)[0][1]}, acc:{accuracy_score(y_test, preds)}")


# plot results
plt.xlim(50, 150)
plt.plot(y_test)
plt.plot(preds)
plt.legend(["gt", "preds"])













