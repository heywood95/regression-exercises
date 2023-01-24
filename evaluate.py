#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import pearsonr, spearmanr

from sklearn.model_selection import train_test_split

import wrangle

from math import sqrt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score


# In[2]:


def plot_residuals(y, yhat):
    ''' This function creates a residual plot'''
  
    plt.scatter(y, (yhat - y))
    plt.xlabel('yhat')
    plt.ylael('residuals')
    plt.show()


# In[3]:


def regression_errors(y, yhat):
    '''This function returns the values for SSE, ESS, TSS, MSE, and RMSE'''
    
    preds['yhat_res_squared'] = preds['yhat_res'] ** 2
    sse_yhat = preds['yhat_res_squared'].sum()
    
    preds['yhat_mean_res'] = preds['yhat'] - preds['baseline_preds']
    preds['yhat_mean_res_squared'] = preds['yhat_mean_res'] ** 2
    ess_baseline = 0
    ess_yhat = preds['yhat_mean_res_squared'].sum()
    
    tss_yhat = sse_yhat + sse_yhat
    
    mse_yhat = sse_yhat / len(preds)
    
    rmse_yhat = sqrt(mse_yhat)
    
    return sse_yhat, ess_yhat, tss_yhat, mse_yhat, rmse_yhat


# In[4]:


def baseline_mean_errors(y):
    '''This function returns the values for SSE, MSE, and RMSE for the baseline model'''
    preds['baseline_res_squared'] = preds['baseline_res']  ** 2
    sse_baseline = preds['baseline_res_squared'].sum()
    mse_baseline = sse_baseline / len(preds)
    rmse_baseline = sqrt(mse_baseline)
    
    return sse_baseline, mse_baseline, rmse_baseline


# In[5]:


def better_than_baseline(y, yhat):
    if sse_yhat < sse_baseline:
        return('Model outperforms baseline model.')
    else:
        return('Baseline model outperforms the model')


# In[ ]:




