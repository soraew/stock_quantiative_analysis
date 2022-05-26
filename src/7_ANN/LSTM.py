
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


# basic stuff
import datetime
import io
import os
from os.path import join
from collections import Counter
from tqdm import tqdm
import gc

from LSTMpreprocess import *



#################### LOAD DATA ######################
################ DEFINE CONSTANTS ###################
##### we will only predict for one stock here #######

nasdaq = pd.read_csv(data_root + "NASDAQ_100_Data_From_2010.csv", sep="\t")
c = 0.14 # this is for scaled apple stock 
window_size = 50
train_ratio = 0.80
batch_size = 64 
stock_name = "AAPL"
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
# the order of the features is extremely important for latter indexing so make sure
features = ['log_Volatility', 'log_Volume', 'log_Adj_Close']
dirname = f"{stock_name}_bc"
check_mkdir(dirname)

# return scaled numpy array of train, test sets of one stock
X_train, X_test, y_train, y_test = get_bc_per_stock_Xy(nasdaq, features, stock_name, train_ratio)



#lstm original
class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers, DROPOUT):
        super(LSTM, self).__init__()
        
        self.num_classes = num_classes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(p=DROPOUT)
        
        if num_layers > 1:
            self.lstm = nn.LSTM(\
                input_size=input_size, 
                hidden_size=hidden_size,
                num_layers=num_layers, 
                batch_first=True,
                # added dropout because we are layering it
                dropout = 0.20
                )
        else:
            self.lstm = nn.LSTM(\
                input_size=input_size, 
                hidden_size=hidden_size,
                num_layers=num_layers, 
                batch_first=True,
                )
            
        
        # Linear(in_features, out_features)
        self.fc = nn.Linear(hidden_size*num_layers, num_classes) 
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size).to(device))
        
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size).to(device))
        
        # Propagate input through LSTM
        output, (h_out, _) = self.lstm(x, (h_0, c_0))
        
        
        h_out = h_out.view(-1, self.hidden_size*self.num_layers)
        
        out = self.fc(h_out)                                         
        out = self.dropout(out)
        
        out = self.sigmoid(out)
       
        return out
    
def init_weights(model):
    for name, param in model.named_parameters():
        nn.init.uniform_(param.data, -0.88, 0.08)


class PearsonLoss(nn.Module):
    def forward(self, x, y):
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        
class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()
        
    def forward(self, yhat, y):
        return self.bce(yhat, y)



######################## START TRAINING ###########################
batch_start_idxs = []
for idx in range(1, X_train.shape[0]+1):
    if idx % batch_size == 0:
        batch_start_idxs.append(idx-1)


# rewrite so that we use batch start 
def train_lstm(num_classes, input_size, hidden_size, num_layers,batch_size, X_train, y_train, X_test, y_test, batch_start_idxs, dirname, epochs, lr, DROPOUT):
    best_val_loss = 100 
    
    lstm = LSTM(num_classes, input_size, hidden_size, num_layers, DROPOUT)
    lstm.to(device)

    # lstm.apply(init_weights) #is this necessary?
    trainX, trainY = Variable(torch.Tensor(X_train)), Variable(torch.Tensor(y_train))
    testX, testY = Variable(torch.Tensor(X_test)), Variable(torch.Tensor(y_test))
    
    num_epochs = epochs
    learning_rate = lr

    # Set Criterion Optimizer and scheduler
    criterion = torch.nn.BCELoss().to(device) 
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100, factor=0.5, min_lr=1e-7, eps=1e-08)

    #optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)


    optimizer.zero_grad() # moved this out of loop 
    vall_losses = []
    train_losses = []
    preds = []

    # Train model
    batch_end_idx = 0
    for epoch in progress_bar(range(num_epochs)):
        optimizer.zero_grad()
        lstm.train()

        loss_per_batch = []
        for batch_start_idx in batch_start_idxs:
            if X_train.shape[0] - batch_start_idx < batch_size:
                batch_end_idx = X_train.shape[0]
            else:
                batch_end_idx = batch_start_idx + batch_size
            trainX_batch, trainY_batch = trainX[batch_start_idx:batch_end_idx, : , :], trainY[batch_start_idx:batch_end_idx]


            outputs= lstm(trainX_batch.to(device))
            torch.nn.utils.clip_grad_norm_(lstm.parameters(),1)

            # obtain loss func
            loss = criterion(outputs, trainY_batch.to(device))
            loss_per_batch.append(loss.detach().cpu().item())
            loss.backward()

            scheduler.step(loss)
            optimizer.step()
            
            # deleting some things because of memory 
            del(trainX_batch)
            del(trainY_batch)
            gc.collect()

        loss_per_batch_mean = np.mean(np.array(loss_per_batch))
        train_losses.append(loss_per_batch_mean)

        #evaluate on test
        lstm.eval()
        valid = lstm(testX.to(device))
        vall_loss = criterion(valid, testY.to(device))
        vall_losses.append(vall_loss.detach().cpu().item())

        scheduler.step(vall_loss)



        if (vall_loss.cpu().item() < best_val_loss) and epoch % 5 == 0:
            torch.save(lstm.state_dict(), join(dirname, f'best_model_{epoch}.pt'))
            print("saved model epoch:",epoch,"val loss is:",vall_loss.cpu().item(), "train loss is:", loss_per_batch_mean)
            best_val_loss = vall_loss.cpu().item()
            preds.append(valid.detach().cpu().numpy())

        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, loss: {loss_per_batch_mean}, valid loss:{vall_loss.cpu().item()}")
            
    
    
    torch.save({f"vall_losses":vall_losses, f"train_losses":train_losses}, join(dirname, "losses.pt"))
    torch.save(preds, join(dirname, "preds.pt"))
    


# network params
# hidden_sizes = [4, 8, 16, 32, 64, 128, 256, 512]
hidden_sizes = [200]
input_size = 3 # number of features
num_layers = 2 
num_classes = 1
DROPOUT = 0.25

# training params
epochs = 50
lr = 1e-3

for hidden_size in hidden_sizes:
    if hidden_size > 200:
        lr = 5e-4
    dirname_per_h = dirname + f"{hidden_size}{num_layers}"
    check_mkdir(dirname_per_h)
    train_lstm(num_classes, input_size, hidden_size, num_layers, batch_size,\
               X_train, y_train, X_test, y_test, \
               batch_start_idxs, dirname_per_h, \
               epochs, lr, DROPOUT)


