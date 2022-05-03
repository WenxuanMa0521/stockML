# -*- coding: utf-8 -*-
"""
Author: Wenxuan Ma @ RUC
mawenxuan@ruc.edu.cn
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

data_2016 = pd.read_csv("result_2016.csv")
data_2017 = pd.read_csv("result_2017.csv")
data_2018 = pd.read_csv("result_2018.csv")
data_2019 = pd.read_csv("result_2019.csv")
data_2020 = pd.read_csv("result_2020.csv")

plt.plot(data_2016.method, data_2016.rmse, color='orangered', marker='o',label = '2016')
plt.plot(data_2016.method, data_2017.rmse, color='lawngreen', marker='*',label = '2017')
plt.plot(data_2016.method, data_2018.rmse, color='aqua', marker='D',label = '2018')
plt.plot(data_2016.method, data_2019.rmse, color='royalblue', marker='>',label = '2019')
plt.plot(data_2016.method, data_2020.rmse, color='crimson', marker='1',label = '2020')
plt.legend()
plt.xlabel('Method')
plt.ylabel('RMSE')
plt.xticks(data_2016.method, data_2016.method, rotation = 45)
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300
plt.savefig('./RMSE')