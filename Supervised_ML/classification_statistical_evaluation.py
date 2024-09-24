

import numpy as np
from scipy import stats
import xlrd
from toolbox_02450 import mcnemar
from sklearn import model_selection 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from numpy import cov

doc = xlrd.open_workbook('wine.xls').sheet_by_index(0)

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
 
y_lr = y.copy()    
    
for i in range(0,len(y)):
    if y[i] != 1:
        y_lr[i] = 1
    else:
        y_lr[i] = 0
    
# Set parameters for statistical evaluation - setup II

loss = 2
K = 10
m = 3    # folds til 3*K=30
J = 0

y_hat_lr = []
y_hat_knn = []
y_hat_base = []
y_true = []

kf = model_selection.KFold(n_splits=K)

# Setting parameter for logistic regression:

lambda_lr = 1

# KNN settings

k_num = 5

for dm in range(m):
        
    k = 0
    for (train_index, test_index) in (kf.split(X)):
        
        # Split dataset:
        
        X_train = X[train_index,:]    # not standardized - KNN datased
        X_test = X[test_index,:]
        y_train =  y_lr[train_index]
        y_test = y_lr[test_index]
 
        mu_train = np.mean(X_train, 0)   # standardization
        sigma_train = np.std(X_train, 0)
        mu_test = np.mean(X_test, 0)
        sigma_test = np.std(X_test, 0)

        X_train_lr = (X_train - mu_train) / sigma_train
        X_test_lr = (X_test - mu_test) / sigma_test     
        
        # Logistic regression
                       
        mdl = LogisticRegression(penalty='l2', C=1/lambda_lr)
        mdl.fit(X_train_lr, y_train)
                
        y_hat_lr = np.append(y_hat_lr, mdl.predict(X_test_lr).T, axis=0)
        
        # KNN
        
        
        metric_params={'V': cov(X_train, rowvar=False)}
        knclassifier = KNeighborsClassifier(n_neighbors=k_num, metric = 'mahalanobis',metric_params=metric_params);
        knclassifier.fit(X_train, y_train);
        
        y_hat_knn = np.append(y_hat_knn, knclassifier.predict(X_test), axis=0);
        
               
        print('dm = ', dm, 'k = ', k)
        
        # Baseline model
        
        mode_train = stats.mode(y_train, axis=0)
        y_base = np.zeros(len(y_test))
        y_base[:] = mode_train[0] 
        y_hat_base = np.append(y_hat_base, y_base, axis = 0)        
        
        
        y_true = np.append(y_true, y_test, axis = 0)
        
        k+=1
        
# Initialize parameters and run test appropriate for setup II
alpha = 0.05
rho = 1/K


# Compare base and logistic regression



[thetahat_base_lr, CI_base_lr, pII_base_lr] = mcnemar(y_true, y_hat_base, y_hat_lr, alpha=alpha)
print('\n\nCompare baseline model vs logistic regression:\np-value: {:.2f}%'.format(pII_base_lr*100), 
      '\nCI [{:.2f}'.format(CI_base_lr[0]),' ; {:.2f}]'.format(CI_base_lr[1]), '\nr hat = {:.2f}'.format( thetahat_base_lr) )

if pII_base_lr  <= alpha and thetahat_base_lr > 0:
    print('\nBase model better than logistic regression')
elif pII_base_lr  < alpha and  thetahat_base_lr < 0:
    print('\nLogistic regression better than base model')   
elif pII_base_lr > alpha:
    print('\nNull hp. is TRUE: two models have same performance')
    
# Compare base and KNN

[thetahat_base_knn, CI_base_knn, pII_base_knn] = mcnemar(y_true, y_hat_base, y_hat_knn, alpha=alpha)
print('\n\nCompare baseline model vs KNN:\np-value: {:.2f}%'.format(pII_base_knn*100), 
      '\nCI [{:.2f}'.format(CI_base_knn[0]),' ; {:.2f}]'.format(CI_base_knn[1]), '\nr hat = {:.2f}'.format( thetahat_base_knn) )

if pII_base_knn <= alpha and thetahat_base_knn > 0:
    print('\nBase model better than KNN')
elif pII_base_knn < alpha and  thetahat_base_knn < 0:
    print('\nKNN better than base model')   
elif pII_base_knn > alpha:
    print('\nNull hp. is TRUE: two models have same performance')
    
# # Compare lr and KNN

[thetahat_lr_knn, CI_lr_knn, pII_lr_knn] = mcnemar(y_true, y_hat_lr, y_hat_knn, alpha=alpha)
print('\n\nCompare logistic regression vs KNN:\np-value: {:.2f}%'.format(pII_lr_knn*100), 
      '\nCI [{:.2f}'.format(CI_lr_knn[0]),' ; {:.2f}]'.format(CI_lr_knn[1]), '\nr hat = {:.2f}'.format( thetahat_lr_knn) )

if pII_lr_knn <= alpha and thetahat_lr_knn > 0:
    print('\nLogistic regression better than KNN')
elif pII_lr_knn < alpha and  thetahat_lr_knn < 0:
    print('\nKNN better than logistic regression')   
elif pII_lr_knn > alpha:
    print('\nNull hp. is TRUE: two models have same performance')
    
    
    





