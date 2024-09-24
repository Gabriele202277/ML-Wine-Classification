# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 16:39:39 2022

@authors: Gabriele, Panagiotis, Naika
"""

import numpy as np
import xlrd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import rlr_validate
from toolbox_02450 import train_neural_net, draw_neural_net, train_neural_net_mod
import torch
from scipy import stats

plt.rcParams['figure.dpi'] = 600
doc = xlrd.open_workbook('..\Data\wine.xls').sheet_by_index(0)

# Import original dataset
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

# Transform dataset for phenols prediction through linear regression (Class attribute is not included in the analysis)
    
phenols_idx = attributeNames.index('Phenols')
y = np.reshape(X[:,phenols_idx], (-1,1))
X_cols = np.asarray(list(range(0,phenols_idx)) + list(range(phenols_idx+1,len(attributeNames))))
X = X[:,X_cols]
M = len(X[0])
attributeNames_reg = [None] * M
for i in range(0,len(X_cols)):
    attributeNames_reg[i] = attributeNames[X_cols[i]]
 
# Regularization dataset and parameters: add offset attribute

X_tilde = np.concatenate((np.ones((X.shape[0],1)),X),1)
M_tilde = M+1
lambdas = np.power(10.,range(-2,2))

# ANN dataset and parameters, set model:

X = stats.zscore(X)
h= np.array([3,5,7]) 
n_replicates = 2        # number of networks trained in each k-fold
max_iter = 10000
loss_fn_ann = torch.nn.MSELoss() 

# Create crossvalidation partition for evaluation

K = 10                                                                          # EXTERNAL FOLDS NUMBER
CV = model_selection.KFold(n_splits=K,shuffle=True)


# Initialize error variables

Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))

Error_train_rlr = np.empty((K,1))
Error_test_rlr = np.empty((K,1))

Error_train = np.empty((K,1))
Error_test = np.empty((K,1))

Error_train_ANN = np.empty((K,1))
Error_test_ANN = np.empty((K,1))

# Initialize weights' vectors

mu = np.empty((K, M_tilde-1))
sigma = np.empty((K, M_tilde-1))

w_rlr = np.empty((M_tilde,K))
w_noreg = np.empty((M_tilde,K))


opt_lambda = np.empty(K)

opt_ann_int_err = np.empty(K)
opt_h = np.empty(K)


k=0
for train_index, test_index in CV.split(X):
              
    # Train set and test set: X_test_tilde is for regularizatio, X_test_ann for ANN and baseline
    
    X_tilde_train = X_tilde[train_index,:] # input inner loop for regularization
    X_tilde_test = X_tilde[test_index,:]
    
    X_train = X[train_index,:] # input inner loop for ANN
    X_test = X[test_index,:]
     
    y_train =  y[train_index] # input inner loop both ANN and regularization
    y_test = y[test_index]
    
   
    #####1 Baseline model error evaluation:
    
    Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]
    
    print('Computing CV fold: {0}'.format(k+1))
    print('Baseline model:')
    print('- Training error: {0}'.format(Error_train_nofeatures[k]))
    print('- Test error: {0}'.format(Error_test_nofeatures[k]))
  
    
    ##### 2 Internal validation for linear regression:
        
    k_inn_regr = 10                                                            # internal folds for regularization
        
    # Regularization
    
    opt_val_err, opt_lambda[k], mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_tilde_train, y_train, lambdas, k_inn_regr)
   
    # Standardization, estimation of y through the defined model and correspondent error:
        
    mu[k, :] = np.mean(X_tilde_train[:, 1:], 0)
    sigma[k, :] = np.std(X_tilde_train[:, 1:], 0)
    X_tilde_train[:, 1:] = (X_tilde_train[:, 1:] - mu[k, :] ) / sigma[k, :] 
    X_tilde_test[:, 1:] = (X_tilde_test[:, 1:] - mu[k, :] ) / sigma[k, :] 
    Xty = X_tilde_train.T @ y_train
    XtX = X_tilde_train.T @ X_tilde_train
    
    # Estimate phenols values - with regularization:
        
    lambdaI = opt_lambda[k] * np.eye(M_tilde)
    lambdaI[0,0] = 0 # do no regularize the bias term!
    w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    
    y_est_train_rgr = np.reshape(X_tilde_train @ w_rlr[:,k], (-1,1))
    y_est_test_rgr = np.reshape(X_tilde_test @ w_rlr[:,k], (-1,1))
    
    Error_train_rlr[k] = np.square(y_train - y_est_train_rgr).sum(axis=0)/y_train.shape[0]   
    Error_test_rlr[k] = np.square(y_test - y_est_test_rgr).sum(axis=0)/y_test.shape[0]
    
    
    # Error_train_rlr[k] = np.square(y_train - np.reshape((X_tilde_train @ w_rlr[:,k]), (-1,1)).sum(axis=0))/y_train.shape[0]
    # Error_test_rlr[k] = np.square(y_test - np.reshape((X_tilde_test @ w_rlr[:,k]), (-1,1)).sum(axis=0))/y_test.shape[0]
    
    print('Regularization model:')
    print('- Optimal lambda value: {0}'.format(opt_lambda[k]))
    print('- Training error: {0}'.format(Error_train_rlr[k]))
    print('- Test error: {0}'.format(Error_test_rlr[k]), '\n\n')
    
    # Estimate phenols values - without regularization:
    
    # w_noreg[:,k] = np.linalg.solve(XtX,Xty).squeeze()
    
    # Error_train[k] = np.square((y_train - np.reshape((X_tilde_train @ w_noreg[:,k]), (-1,1))).sum(axis=0))/y_train.shape[0]
    # Error_test[k] = np.square((y_test - np.reshape((X_tilde_test @ w_noreg[:,k]), (-1,1))).sum(axis=0))/y_test.shape[0]
    
    ##### 3 Internal validation for ANN ##########################
    
    k_ann = 3                                                                  # internal folds for regularization
    n_replicates_inn = 2
    max_iter_inn = 5000
    
    CV2 = model_selection.KFold(k_ann, shuffle=True) # set the internal layer division for ANN

    inn_error_test_ann = np.empty((np.shape(h)[0],k_ann))      # vector initialization: error per each internal fold
      

    for (kk, (train_index_ann, test_index_ann)) in enumerate(CV2.split(X_train,y_train)):
        
        X_train_ann = torch.Tensor(X_train[train_index_ann,:])
        y_train_ann = torch.Tensor(y_train[train_index_ann])
        X_test_ann  = torch.Tensor(X_train[test_index_ann,:])
        y_test_ann  = torch.Tensor(y_train[test_index_ann])
        
        i = 0     
        for h_inn in h:
            
            model_inn = lambda: torch.nn.Sequential(
                        torch.nn.Linear(M, h_inn), 
                        torch.nn.Tanh(),   
                        torch.nn.Linear(h_inn, 1)
                        )
                
            net_inn, final_loss_inn, learning_curve_inn = train_neural_net_mod(model_inn,
                                                                loss_fn_ann,
                                                                X=X_train_ann,
                                                                y=y_train_ann,
                                                                n_replicates = n_replicates_inn,
                                                                max_iter = max_iter_inn)
            # Build error matrix: estimate only TEST error
            
            y_estim_inn = net_inn(X_test_ann)
            eu_dist_inn = (y_estim_inn.float()-y_test_ann.float())**2
            inn_error_test_ann[i,kk] = (sum(eu_dist_inn).type(torch.float)/len(y_test_ann)).data.numpy()
            i+=1
            
    # Select best lambda and cor. error for inner layer
    
    opt_ann_int_err[k] = np.min(np.mean(inn_error_test_ann,axis=1))
    opt_h[k] = h[np.argmin(np.mean(inn_error_test_ann,axis=1))]
    
    # Train and evaluate best models obtained through inner layers
       
    X_train = torch.Tensor(X_train)  # Transform dataset in tensor
    X_test = torch.Tensor(X_test)
    y_train = torch.Tensor(y_train)
    y_test = torch.Tensor(y_test)
    
    h_eval = h[np.argmin(np.mean(inn_error_test_ann,axis=1))]
      
    model_ext = lambda: torch.nn.Sequential(  # create model
                torch.nn.Linear(M, h_eval), 
                torch.nn.Tanh(),   
                torch.nn.Linear(h_eval, 1)
                )
           
    net, final_loss, learning_curve = train_neural_net_mod(model_ext,
                                                       loss_fn_ann,
                                                       X=X_train,
                                                       y=y_train,
                                                       n_replicates = n_replicates,
                                                       max_iter = max_iter)
    y_train_est = net(X_train)
    y_test_est = net(X_test)
   
    eu_dist_train = (y_train_est.float()-y_train.float())**2
    eu_dist_test = (y_test_est.float()-y_test.float())**2
     
    Error_train_ANN[k] = (sum(eu_dist_train).type(torch.float)/len(y_train)).data.numpy()
    Error_test_ANN[k] = (sum(eu_dist_test).type(torch.float)/len(y_test)).data.numpy()
    
    print('Artificial neural network:')
    print('- Optimal hidden units number: {0}'.format(opt_h[k]))
    print('- Training error: {0}'.format(Error_train_ANN[k]))
    print('- Test error: {0}'.format(Error_test_ANN[k]), '\n\n')
    
    
    
            
    
    k+=1



    