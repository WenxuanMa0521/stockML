import pandas as pd
import numpy as np

from prediction import *

import warnings
warnings.filterwarnings("ignore")

#==============================================================================
# Data Preparation

data = pd.read_csv('stock.csv')

var = ['mvel1', 'beta', 'betasq', 'chmom',
       'dolvol', 'idiovol', 'indmom', 'mom1m', 'mom6m', 'mom12m', 'mom36m',
       'pricedelay', 'turn', 'absacc', 'acc', 'age', 'agr', 'bm', 'bm_ia',
       'cashdebt', 'cashpr', 'cfp', 'cfp_ia', 'chatoia', 'chcsho', 'chempia',
       'chinv', 'chpmia', 'convind', 'currat', 'depr', 'divi', 'divo', 'dy',
       'egr', 'ep', 'gma', 'grcapx', 'grltnoa', 'herf', 'hire', 'invest',
       'lev', 'lgr', 'mve_ia', 'operprof', 'orgcap', 'pchcapx_ia', 'pchcurrat',
       'pchdepr', 'pchgm_pchsale', 'pchquick', 'pchsale_pchinvt',
       'pchsale_pchrect', 'pchsale_pchxsga', 'pchsaleinv', 'pctacc', 'ps',
       'quick', 'rd', 'rd_mve', 'rd_sale', 'realestate', 'roic', 'salecash',
       'saleinv', 'salerec', 'secured', 'securedind', 'sgr', 'sin', 'sp',
       'tang', 'tb', 'aeavol', 'cash', 'chtx', 'cinvest', 'ear', 'nincr',
       'roaq', 'roavol', 'roeq', 'rsup', 'stdacc', 'stdcf', 'ms', 'baspread',
       'ill', 'maxret', 'retvol', 'std_dolvol', 'std_turn', 'zerotrade',
       'sic2']
y = ['return']

data[var] = (data[var] - data[var].mean())/data[var].std()

#==============================================================================
# Data Splitting
# train: 9 years, valid: 6 year, test: 1 year

for i in range(5):

    train_data = data.iloc[data[(data.year >= 2000+i) & (data.year <= 2009+i)].index]
    valid_data = data.iloc[data[(data.year >= 2010+i) & (data.year <= 2015+i)].index]
    test_data = data.iloc[data[(data.year == 2016+i)].index]
    
    pred(train_data, valid_data, test_data, var, y, 2016+i)