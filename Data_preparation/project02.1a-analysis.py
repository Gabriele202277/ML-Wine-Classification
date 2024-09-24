# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 16:39:39 2022

@authors: Gabriele, Panagiotis, Naika
"""


import numpy as np
import pandas as pd
import xlrd
import matplotlib.pyplot as plt
from scipy.linalg import svd
import sklearn.linear_model as lm

plt.rcParams['figure.dpi'] = 600
doc = xlrd.open_workbook('wine.xls').sheet_by_index(0)


# Extract attribute names
attributeNames = doc.row_values(rowx=0, start_colx=1, end_colx=14)


classLabels = doc.col_values(0,1,179) # check out help(doc.col_values)
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames,range(len(classNames))))

y = np.array([classDict[value] for value in classLabels])

N = len(y)
M = len(attributeNames)
C = len(classNames)

X = np.zeros((N, M))

for i in range(0,M):
    X[:,i] = np.array(doc.col_values(i+1,1,N+1)).T # untransformed dataset
    
X_data = pd.DataFrame(data = np.c_[X,np.expand_dims(y, axis = 1)], columns = attributeNames + ['Class'])

with pd.ExcelWriter('OUR_wine_data.xlsx') as writer:  
    X_data.to_excel(writer, sheet_name = 'original_dataset')
    X_data.describe().to_excel(writer, sheet_name = 'original_dataset_descr')
    
phenols_idx = attributeNames.index('Phenols')
y = X[:,phenols_idx]

X_cols = np.asarray(list(range(0,phenols_idx)) + list(range(phenols_idx+1,len(attributeNames))))

X = X[:,X_cols]
M = len(X[0])

attributeNames_reg = [None] * M

for i in range(0,len(X_cols)):
    attributeNames_reg[i] = attributeNames[X_cols[i]]
    



# Fit ordinary least squares regression model
model = lm.LinearRegression()
model.fit(X,y)

# Predict Phenols content
y_est = model.predict(X)
residual = y_est-y

for i in range(0,M):
    plt.figure()
    plt.plot(X[:,i], residual, '.')
    plt.xlabel(attributeNames_reg[i])
    plt.ylabel('Residual')



