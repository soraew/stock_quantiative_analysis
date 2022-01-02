#!/Users/soraward/opt/miniconda3/bin/python3 
data_root = "../archive/"
# ML stuff
import numpy as np
import torch
from sklearn.linear_model import Lasso
import pandas as pd


from PIL import Image
# plotting
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
mpl.rc('image', cmap='gray')

# basic stuff
import datetime
import requests
import io
from collections import Counter






# read data
# for now, reading only unitl ebay
nasdaq = pd.read_csv(data_root + "NASDAQ_100_Data_From_2010.csv", sep="\t", nrows=100598)

# resetting index to datetime
nasdaq.Date = pd.to_datetime(nasdaq.Date)
nasdaq.set_index("Date", inplace = True)

# line plot stock price
def line_plot_(param, stock_list, df):
    title = param 
    plt.figure()
    plt.title(title)
    df_grouped = df.groupby(["Name"])
    for stock in stock_list:
        df_grouped.get_group(stock)[param].plot(kind = "line")
    plt.legend()
    plt.show()


# stock lists
stock_list = ['AMGN','AAPL','MSFT','GOOG','AMZN','FB','TSLA','CTSH','JD','EBAY','AMD','SBUX','NVDA','ZM']
stock_list2 = ['CHKP','CHTR','CMCSA','COST','CPRT','CRWD','CSCO','CSX','CTAS','CTSH','DLTR','DOCU','DXCM','EA']
stock_list3 = ['EXC','FAST','FB','FISV','FOX','FOXA','GILD','GOOG','GOOGL','HON','IDXX','ILMN','INCY','INTC','INTU']
first_names = list(set([name for name in nasdaq.Name]))

# for easier understanding
nasdaq["log_Volume"] = np.log(np.array(nasdaq["Volume"]+1e-9))

# reuters data に合わせる　
# nasdaq_reu = nasdaq.loc["2018-03-20":"2020-07-18"]s





