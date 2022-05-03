import os
import math
import pandas as pd
import numpy as np

from sklearn import linear_model
from sklearn.linear_model import Ridge, RidgeCV
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from train_nn import *
import usnrt

import warnings
warnings.filterwarnings("ignore")

#==============================================================================
# Performance Evaluation

def RMSE(pred, real):
    pred = np.array(pred).reshape(-1)
    real = np.array(real).reshape(-1)
    return np.sqrt(((pred - real) ** 2).mean())


def OOS(pred, real):
    pred = np.array(pred).reshape(-1)
    real = np.array(real).reshape(-1)
    s1 = ((pred - real) ** 2).sum()
    s2 = (real ** 2).sum()
    return s1, s2, 1 - s1/s2

#==============================================================================
# Models

def pred(train_data, valid_data, test_data, var, y, year):
#==============================================================================
## OLS

    lr = linear_model.LinearRegression()
    lr.fit(train_data[var], train_data[y])
    pred_lr = lr.predict(test_data[var])
    
    rmse = RMSE(pred_lr, test_data[y])
    s1, s2, oos = OOS(pred_lr, test_data[y])

    if not os.path.isfile("result_"+str(year)+".csv"):
        with open("result_"+str(year)+".csv", 'a') as file:
            file.write('method,year,rmse,s1,s2,oos\n')
    with open("result_"+str(year)+".csv", 'a') as file:
        file.write("{},{},{},{},{},{}\n".format('OLS', year, rmse, s1, s2, oos))

#==============================================================================
## Ridge

    alphas = [0.1, 1, 10, 100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    ridge = RidgeCV(alphas = alphas)
    ridge.fit(train_data[var], train_data[y])
    pred_ridge = ridge.predict(test_data[var])

    print('alpha: %.2f' % ridge.alpha_)
    rmse = RMSE(pred_ridge, test_data[y])
    s1, s2, oos = OOS(pred_ridge, test_data[y])

    with open("result_"+str(year)+".csv", 'a') as file:
        file.write("{},{},{},{},{},{}\n".format('Ridge', year, rmse, s1, s2, oos))

#==============================================================================
## CART

    Nmin = math.floor(len(train_data) / 10)

    cart = tree.DecisionTreeRegressor(min_samples_leaf = Nmin)
    cart.fit(train_data[var], train_data[y])
    pred_cart = cart.predict(test_data[var])
    
    rmse = RMSE(pred_cart, test_data[y])
    s1, s2, oos = OOS(pred_cart, test_data[y])

    with open("result_"+str(year)+".csv", 'a') as file:
        file.write("{},{},{},{},{},{}\n".format('CART', year, rmse, s1, s2, oos))
#==============================================================================
## RF

    T = 100

    rf = RandomForestRegressor(n_estimators = T, min_samples_leaf = Nmin)
    rf.fit(train_data[var], train_data[y])
    pred_rf = rf.predict(test_data[var])
    
    rmse = RMSE(pred_rf, test_data[y])
    s1, s2, oos = OOS(pred_rf, test_data[y])

    with open("result_"+str(year)+".csv", 'a') as file:
        file.write("{},{},{},{},{},{}\n".format('RF', year, rmse, s1, s2, oos))

#==============================================================================
## GBDT

    gbdt = GradientBoostingRegressor(n_estimators = T, min_samples_leaf = Nmin)
    gbdt.fit(train_data[var], train_data[y])
    pred_gbdt = gbdt.predict(test_data[var])
    
    rmse = RMSE(pred_gbdt, test_data[y])
    s1, s2, oos = OOS(pred_gbdt, test_data[y])

    with open("result_"+str(year)+".csv", 'a') as file:
        file.write("{},{},{},{},{},{}\n".format('GBDT', year, rmse, s1, s2, oos))

#==============================================================================
## XGBoost

    xgboost = XGBRegressor(n_estimators = T)
    xgboost.fit(train_data[var], train_data[y])
    pred_xgboost = xgboost.predict(test_data[var])
    
    rmse = RMSE(pred_xgboost, test_data[y])
    s1, s2, oos = OOS(pred_xgboost, test_data[y])

    with open("result_"+str(year)+".csv", 'a') as file:
        file.write("{},{},{},{},{},{}\n".format('XGBoost', year, rmse, s1, s2, oos))

#============================================================================
## Neural Network

    x_test, y_test = test_data[var], test_data[y]
    x_test= torch.tensor(x_test.values, dtype = torch.float32)
    y_test = torch.tensor(y_test.values, dtype = torch.float32).reshape(-1, 1)

    num_epochs = 1000
    batch_size = 128
    lr = 0.01
    set_seed = 42

#============================================================================
### NN1

    hidden_dims = [32]

    model = fit_nn(train_data, valid_data, var, y, hidden_dims = hidden_dims, num_epochs = num_epochs, 
    batch_size = batch_size, lr = lr, set_seed = set_seed)
    pred_nn = model(x_test).detach().numpy()

    rmse = RMSE(pred_nn, test_data[y])
    s1, s2, oos = OOS(pred_nn, test_data[y])

    with open("result_"+str(year)+".csv", 'a') as file:
        file.write("{},{},{},{},{},{}\n".format('NN1', year, rmse, s1, s2, oos))

#============================================================================
### NN2

    hidden_dims = [32, 16]

    model = fit_nn(train_data, valid_data, var, y, hidden_dims = hidden_dims, num_epochs = num_epochs, 
    batch_size = batch_size, lr = lr, set_seed = set_seed)
    pred_nn = model(x_test).detach().numpy()

    rmse = RMSE(pred_nn, test_data[y])
    s1, s2, oos = OOS(pred_nn, test_data[y])

    with open("result_"+str(year)+".csv", 'a') as file:
        file.write("{},{},{},{},{},{}\n".format('NN2', year, rmse, s1, s2, oos))

#============================================================================
### NN3

    hidden_dims = [32, 16, 8]

    model = fit_nn(train_data, valid_data, var, y, hidden_dims = hidden_dims, num_epochs = num_epochs, 
    batch_size = batch_size, lr = lr, set_seed = set_seed)
    pred_nn = model(x_test).detach().numpy()

    rmse = RMSE(pred_nn, test_data[y])
    s1, s2, oos = OOS(pred_nn, test_data[y])

    with open("result_"+str(year)+".csv", 'a') as file:
        file.write("{},{},{},{},{},{}\n".format('NN3', year, rmse, s1, s2, oos))

#============================================================================
### NN4

    hidden_dims = [32, 16, 8, 4]

    model = fit_nn(train_data, valid_data, var, y, hidden_dims = hidden_dims, num_epochs = num_epochs, 
    batch_size = batch_size, lr = lr, set_seed = set_seed)
    pred_nn = model(x_test).detach().numpy()

    rmse = RMSE(pred_nn, test_data[y])
    s1, s2, oos = OOS(pred_nn, test_data[y])

    with open("result_"+str(year)+".csv", 'a') as file:
        file.write("{},{},{},{},{},{}\n".format('NN4', year, rmse, s1, s2, oos))

#============================================================================
### NN5

    hidden_dims = [32, 16, 8, 4, 2]

    model = fit_nn(train_data, valid_data, var, y, hidden_dims = hidden_dims, num_epochs = num_epochs, 
    batch_size = batch_size, lr = lr, set_seed = set_seed)
    pred_nn = model(x_test).detach().numpy()

    rmse = RMSE(pred_nn, test_data[y])
    s1, s2, oos = OOS(pred_nn, test_data[y])

    with open("result_"+str(year)+".csv", 'a') as file:
        file.write("{},{},{},{},{},{}\n".format('NN5', year, rmse, s1, s2, oos))
        
#============================================================================
### Our Tree

    split_dims, leaf_dims = [32, 16, 8], [32, 16]

    our_tree = usnrt.grow_tree(train_data, valid_data, var, y, split_dims = split_dims, leaf_dims = leaf_dims,  Nmin = Nmin, num_epochs = num_epochs, batch_size = batch_size, lr = lr, index = [1], region_info = 'All Data')
    pred = our_tree.predict_all(test_data.drop(y, axis = 1), var_cont, var_cate)
    
    rmse = RMSE(pred, test_data[y])
    s1, s2, oos = OOS(pred, test_data[y])

    with open("result_"+str(year)+".csv", 'a') as file:
        file.write("{},{},{},{},{},{}\n".format('Our Tree', year, rmse, s1, s2, oos))