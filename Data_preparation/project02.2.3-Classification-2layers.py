# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 16:39:39 2022

@authors: Gabriele, Panagiotis, Naika
"""

import numpy as np
import xlrd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn import model_selection 
from sklearn.linear_model import LogisticRegression
from toolbox_02450 import rocplot, confmatplot
from sklearn.neighbors import KNeighborsClassifier
from numpy import cov

plt.rcParams['figure.dpi'] = 600
doc = xlrd.open_workbook(r'C:\Users\bonas\OneDrive\Υπολογιστής\DTU\4th SEMESTER\Introduction to machine learning and data mining\02450Toolbox_Python\02450Toolbox_Python\Tools\wine.xls').sheet_by_index(0)

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
 
    
# Define a new y vector: y=0 for class 1 and 3, y=1 for class 2
 
y_lr = y.copy()    
    
for i in range(0,len(y)):
    if y[i] != 1:
        y_lr[i] = 0

# Logistic regression:
    
k_lr = 5
CVlr = model_selection.KFold(n_splits=k_lr,shuffle=True)   
lambda_interval = np.power(10., range(-2,2))  #lambda interval

# KNN method:
 

k_knn = 5
CVknn = model_selection.KFold(n_splits=k_knn,shuffle=True)   
knn = np.logspace(-2, 2, 20)

K_values = np.arange(5,31,1)   



# External cross-validation layer

K = 10                                                                          # EXTERNAL FOLDS NUMBER
CV = model_selection.KFold(n_splits=K,shuffle=True)


# Initialize error variables

Error_train_nofeatures = np.zeros((K,1))
Error_test_nofeatures = np.zeros((K,1))

inn_error_lr = np.zeros((len(lambda_interval),k_lr))      
Error_train_lr = np.zeros((K,1))
Error_test_lr = np.zeros((K,1))

inn_error_knn = np.zeros((len(K_values),k_knn))   
Error_train_knn = np.zeros((K,1))
Error_test_knn = np.zeros((K,1))


# mu_out = np.empty((K, M-1))
# sigma_out = np.empty((K, M-1))

# mu_inn = np.empty((K, M-1))
# sigma_inn = np.empty((K, M-1))

opt_lambda = np.zeros(K)
opt_lr_int_err = np.zeros(K)

opt_K = np.empty(K, dtype='int')
opt_knn_int_err = np.zeros(K)


k=0
for train_index, test_index in CV.split(X):
                
    X_train = X[train_index,:]    # OUTER fold
    X_test = X[test_index,:]
     
    mu_train_out = np.mean(X_train, 0)   # standardization
    sigma_train_out = np.std(X_train, 0)
    
    mu_test_out = np.mean(X_test, 0)
    sigma_test_out = np.std(X_test, 0)

    X_train_st = (X_train - mu_train_out) / sigma_train_out
    X_test_st = (X_test - mu_test_out) / sigma_test_out
    
    # standarlize data 
    #Xtrain=(X_train - mu_train_out) / sigma_train_out
    #X_test=(X_test - mu_test_out) / sigma_test_out
    
    y_train =  y_lr[train_index]
    y_test = y_lr[test_index]
    
   
    ##### 1 Baseline model error evaluation: ##################################
    
    # Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0]
    # Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]
    
    
    mode_train = stats.mode(y_train)
    
    y_train_baseline = np.zeros([len(y_train)])
    y_test_baseline = np.zeros([len(y_test)])
    
    for i in range(0, len(y_test)):  
        y_train_baseline[i] = mode_train[0]
        y_test_baseline[i] = mode_train[0]
    
    Error_train_nofeatures[k]= (np.sum(y_train_baseline != y_train))/len(y_train) 
    Error_test_nofeatures[k] =(np.sum(y_test_baseline != y_test))/len(y_test) 

    print('Computing CV fold: {0}'.format(k+1))
    print('Baseline model:')
    print('- Training error: {0}'.format(Error_train_nofeatures[k]))
    print('- Test error: {0}'.format(Error_test_nofeatures[k]))
    

  
    
    ##### 2 Internal validation for logistic regression: ######################
        
    kk = 0
    for (train_index_lr, test_index_lr) in (CVlr.split(X_train,y_train)):
        
        X_train_lr = X_train_st[train_index_lr]
        y_train_lr = y_train[train_index_lr]
        X_test_lr  = X_train_st[test_index_lr]
        y_test_lr  = y_train[test_index_lr]
        
    # Train and test logistic regression internal folds
        
        i=0
        for l in lambda_interval:
            mdl = LogisticRegression(penalty='l2', C=1/l)
            mdl.fit(X_train_lr, y_train_lr)

            y_test_est_lr_in = mdl.predict(X_test_lr).T
            
            #train_error_lo[g] = np.sum(y_train_est_lo != y_train_lo) / len(y_train_lo)
            #test_error_lo[g] = np.sum(y_test_est_lo != y_test_lo) / len(y_test_lo)
            
            test_misclassified_lr_in = np.abs(y_test_est_lr_in - y_test_lr)
            test_err_lr_in = sum(test_misclassified_lr_in)/len(y_test_lr)
            inn_error_lr[i,kk] = test_err_lr_in # store error in a matrix: rows=models, columns=inner fold of logistic regression
            
            #w_est = mdl.coef_[0] 
            #coefficient_norm[l] = np.sqrt(np.sum(w_est**2))
            # end of the inner fold for one k 
        # end of the inner Fold
            i+=1
        kk+=1
    
    # Determine optimal lambda in inner folds
    
    opt_lr_int_err[k] = np.min(np.mean(inn_error_lr,axis=1))
    opt_lambda[k] = lambda_interval[np.argmin(np.mean(inn_error_lr,axis=1))]
    
    # Train logistic regression model on training data from outer splits, using opt_lambda
    
    model_lr = LogisticRegression(penalty='l2', C=1/opt_lambda[k])
    model_lr.fit(X_train_st, y_train)
    
    # Test model on the test data from the outer splits
    
    y_train_est_lr = model_lr.predict(X_train_st).T
    y_test_est_lr = model_lr.predict(X_test_st).T
       
    # Determine train and test errors - outer fold
    
    train_misclassified_lr = np.abs(y_train_est_lr - y_train) 
    test_misclassified_lr = np.abs(y_test_est_lr - y_test)
    
    Error_train_lr[k]  = sum(train_misclassified_lr)/len(y_train)
    Error_test_lr[k] = sum(test_misclassified_lr)/len(y_test)
    
    print('Logistic regression model:')
    print('- Optimal lambda value: {0}'.format(opt_lambda[k]))
    print('- Training error: {0}'.format(Error_train_lr[k]))
    print('- Test error: {0}'.format(Error_test_lr[k]), '\n\n')
    

    ##### 3 Internal validation for KNN: ######################################
    
    ii = 0
    for (train_index_knn, test_index_knn) in (CVknn.split(X_train,y_train)):
        
        X_train_knn = X_train[train_index_knn]   # not standardized dataset
        y_train_knn = y_train[train_index_knn]
        X_test_knn  = X_train[test_index_knn]
        y_test_knn  = y_train[test_index_knn]
        
    # Train and test logistic regression internal folds
        
        j=0
        for nk in K_values:
            metric_params={'V': cov(X_train, rowvar=False)}
            knclassifier = KNeighborsClassifier(n_neighbors=nk, metric = 'mahalanobis',metric_params=metric_params);
            knclassifier.fit(X_train_knn, y_train_knn);

            y_test_est_knn_in = knclassifier.predict(X_test_knn);
            
            test_misclassified_knn_in = np.abs(y_test_est_knn_in - y_test_knn)
            test_err_knn_in = sum(test_misclassified_knn_in)/len(y_test_knn)
            inn_error_knn[j,ii] = test_err_lr_in # store error in a matrix: rows=models, columns=inner fold of logistic regression
            j+=1
        ii+=1
    
    # Determine optimal neighbours' number in inner folds
    
    opt_knn_int_err[k] = np.min(np.mean(inn_error_knn, axis=1))
    opt_K[k] = K_values[np.argmin(np.mean(inn_error_knn,axis=1))]
    
    # Train KNN model on training data from outer splits, using opt_lambda
      
    model_knn = KNeighborsClassifier(n_neighbors=opt_K[k], metric = 'mahalanobis',metric_params=metric_params);
    model_knn.fit(X_train, y_train);
    
    # Test model on the test data from the outer splits
    
    y_train_est_knn = model_knn.predict(X_train);
    y_test_est_knn = model_knn.predict(X_test);
       
    # Determine train and test errors - outer fold
    
    train_misclassified_knn = np.abs(y_train_est_knn - y_train) 
    test_misclassified_knn = np.abs(y_test_est_knn - y_test)
    
    Error_train_knn[k]  = sum(train_misclassified_knn)/len(y_train)
    Error_test_knn[k] = sum(test_misclassified_knn)/len(y_test)
    
    print('KNN model:')
    print('- Optimal K value: {0}'.format(opt_K[k]))
    print('- Training error: {0}'.format(Error_train_knn[k]))
    print('- Test error: {0}'.format(Error_test_knn[k]), '\n\n')
    
    
    
    
            
    
    k+=1
    
print('\n\nEND')



    