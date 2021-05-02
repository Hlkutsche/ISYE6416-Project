#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 20:34:26 2021

@author: hallekutsche
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import sklearn.linear_model as sk
import sklearn.model_selection as skm
import sklearn.metrics as m
import seaborn as sns

def ridge_reg(data, y):

    coef = []
    ridge = sk.RidgeCV( normalize = True)
    ridge.fit(data, y)
    predict = ridge.predict(data)
    coef.append(ridge.score(data,y))

    return(coef)


def lasso(data, y):

    data_train, data_test, y_train, y_test = skm.train_test_split(data, y, test_size=0.75, 
                                                              random_state =1 )
    lasso = sk.LassoCV(normalize = True, max_iter= 1000)

    lasso.fit(data_train, y_train)
    lasso_r = lasso.predict(data_test)
    lasso_coef= lasso.coef_
    train = lasso.score(data_train, y_train)
    test = lasso.score(data_test, y_test)
    alphas, coef, zz = lasso.path(data, y)
    # Find best score of all iterations and save coefficients for that score
    max_r_index = np.argmax(y_test)
    best_r = lasso_r[max_r_index]
    
    return(train, test, lasso_coef, coef, best_r)

# Create dataset
data =  pd.read_csv('data_cleaned.csv', sep=','  , engine='python')

# Dependent vars
cancer = data.cancer
chd = data.chd
mentalh = data.mentalh

# Independent vars
data = data.drop('cancer', axis = 1)
data = data.drop('chd', axis =1)
data = data.drop('mentalh', axis = 1)


coefC = ridge_reg(data,cancer)
coefH = ridge_reg(data,chd)
coefM = ridge_reg(data,mentalh)




C_train, C_test, C_lasso, C_path, C_r = lasso(data,cancer)
H_train, H_test, H_lasso, H_path , H_r= lasso(data,chd)
M_train, M_test, M_lasso, M_path, M_r = lasso(data,mentalh)

xx = np.sum(np.abs(C_path.T), axis = 1)
xx /= xx[-1]


plt.plot(xx, C_path.T)
plt.title('Cancer Lasso Path')
plt.ylabel('R squared')
plt.xlabel('Iterations')
plt.show()
plt.clf()

xx = np.sum(np.abs(H_path.T), axis = 1)
xx /= xx[-1]

plt.plot(xx, H_path.T)
plt.title('CHD Lasso Path')
plt.ylabel('R squared')
plt.xlabel('Iterations')
plt.show()
plt.clf()


xx = np.sum(np.abs(M_path.T), axis = 1)
xx /= xx[-1]

plt.plot(xx, M_path.T)
plt.title('Mental health Lasso Path')
plt.ylabel('R squared')
plt.xlabel('Iterations')
plt.show()

print('Cancer Ridge Coefficient ', coefC, 'Cancer Lasso R Squared ', C_r )
print('CHD Ridge Coefficient ', coefH,'CHD Lasso R Squared ', H_r)
print('Mental Health Ridge Coefficient ', coefM, 'Mental Health Lasso R Squared ', M_r)
