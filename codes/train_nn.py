# -*- coding: utf-8 -*-
"""
Author: Wenxuan Ma @ RUC
mawenxuan@ruc.edu.cn
"""

import os
import copy
import math
import random
import numpy as np
import pandas as pd

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

import warnings
warnings.filterwarnings("ignore")

#==============================================================================

class model_nn(nn.Module):
    def __init__(self, in_dim, hiddens):
        super(model_nn, self).__init__()
        
        self.dims = [in_dim] + list(hiddens) + [1]
        
        self.linears = nn.ModuleList()
        for i in range(1, len(self.dims)):
            self.linears.append(nn.Linear(self.dims[i-1], self.dims[i]))
        
        self.activation = nn.ReLU()
        # self.dropout = nn.Dropout()
    
    def forward(self, X):
        for i in range(len(self.linears) - 1):
            X = self.activation(self.linears[i](X))
            # X = self.dropout(X)
        X = self.linears[-1](X)
        return X

#==============================================================================

criterion = nn.MSELoss()

def load_array(data_arrays, batch_size, is_train = True):
    dataset = TensorDataset(*data_arrays)
    return DataLoader(dataset, batch_size, shuffle = is_train)

def initialize(model, weight_seed):
    torch.manual_seed(weight_seed)
    for linear in model.linears:
        nn.init.xavier_normal_(linear.weight)
        nn.init.constant_(linear.bias, 0.0)

#==============================================================================

def fit_nn(train_data, valid_data, var, y, hidden_dims, num_epochs, batch_size, lr, set_seed):
    
    model = model_nn(len(var), hidden_dims)
    
    x_train, y_train = train_data[var], train_data[y]
    x_valid, y_valid = valid_data[var], valid_data[y]
    
    x_train = torch.tensor(x_train.values, dtype = torch.float32)
    y_train = torch.tensor(y_train.values, dtype = torch.float32).reshape(-1, 1)
    x_valid = torch.tensor(x_valid.values, dtype = torch.float32)
    y_valid = torch.tensor(y_valid.values, dtype = torch.float32).reshape(-1, 1)
    
    initialize(model, set_seed)
    optimizer = optim.Adam(model.parameters(), lr = lr)
    train_iter = load_array((x_train, y_train), batch_size = batch_size)
    
    best_valid = 1e8
    epoch_valids = []
    
    for epoch in range(num_epochs):
        batch_valids = []
        for x_batch, y_batch in train_iter:
            model.train()
            out = model(x_batch)
            loss = criterion(out, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            model.eval()
            valid_loss = criterion(model(x_valid), y_valid).detach().numpy()
            batch_valids.append(valid_loss)
            if valid_loss < best_valid:
                best_valid = valid_loss
                best_dict = copy.deepcopy(model.state_dict())
        
        #print("epoch {}:".format(epoch),np.mean(batch_valids))
        epoch_valids.append(np.mean(batch_valids))
        if len(epoch_valids) >= 5 and epoch_valids[-1] > epoch_valids[-5] * 1.05:
            break
        if len(epoch_valids) >= 25 and np.mean(epoch_valids[-5:]) > np.mean(epoch_valids[-25:-5]):
            break
    
    model.load_state_dict(best_dict)
    model.eval()
    Y_hat = model(x_train).detach().numpy()
    residual = y_train.detach().numpy().reshape(-1) - Y_hat.reshape(-1)
    
    return model, residual

