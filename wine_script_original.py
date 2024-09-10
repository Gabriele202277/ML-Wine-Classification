# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 16:22:31 2021

@author: Katharina
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn as sk
from sklearn.preprocessing import StandardScaler

# adjust filepath
file_path = 'wine.data'
winedata = pd.read_csv(file_path)

#%%
# Preprocessing 
raw_data = winedata.to_numpy() 
cols = range(1, 14) 
X_original = raw_data[:, cols]
attributeNames = np.asarray(winedata.columns[cols])

N = 178

X1 = X_original - np.ones((N, 1))*X_original.mean(0)  # transformed dataset 1. substract the mean value
X2 = X1*(1/np.std(X1,0))    

# Standardize 
#https://towardsdatascience.com/how-and-why-to-standardize-your-data-996926c2c832
scaler = StandardScaler()
scaled_data = scaler.fit_transform(X_original) 
X = scaled_data

a = X2==X

#Proofing the Standardization
print()
print("Verify that the mean of each feature (column) is 0", scaled_data.mean(axis = 0))
print()
print("Verify that the std of each feature (column) is 1", scaled_data.std(axis = 0))
print()

#%%
## Classification problem (ex1.5.1)

# Sort out the attributes 
classLabels = raw_data[:,0]
classNames_original = np.unique(classLabels)
classNames = np.array(["cultivar 1","cultivar 2", "cultivar 3"])

#How can I give the classNames names instead of numbers as in the original dataset?
#classNames[1,2,3]= classNames('cultivar 1', 'cultivar 2', 'cultivar 3')
classDict = dict(zip(classNames_original,range(len(classNames_original))))
y = np.array([classDict[cl] for cl in classLabels])

N, M = X.shape
C = len(classNames)


X_c = X.copy();
y_c = y.copy();
attributeNames_c = attributeNames.copy();
#%%
##Regression problem (ex.1.5.4)

# we want to predict Total phenols content (0) so we extract this information from the 
# X Matrix and make it the new y_r vector
data = np.concatenate((X_c, np.expand_dims(y_c,axis=1)), axis=1)
y_r = data[:, 5]
X_r = np.delete(data,5,1)

#One-out-of-K encoding
# since our wine cultivar classes are ranged by now, we split them up to binary 
# one-out-of-K encoding, giving each class (cultivar) it's own column 
cultivar= np.array(X_r[:, -1], dtype=int).T
K = cultivar.max()+1
cultivar_encoding = np.zeros((cultivar.size, K))
cultivar_encoding[np.arange(cultivar.size), cultivar] = 1

#replace the last row in X_r with the encoded version 
X_r = np.concatenate( (X_r[:, :-1], cultivar_encoding), axis=1) 

# We need to update the attribute names and store the Total phenols name 
# as the name of the target variable for a regression:
targetName_r = attributeNames_c[5]
attributeNames_r = np.delete(attributeNames_c,5,0)
#Update M 
N,M = X_r.shape



