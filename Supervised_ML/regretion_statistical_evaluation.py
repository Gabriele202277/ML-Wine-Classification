

import numpy as np
import xlrd
from toolbox_02450 import train_neural_net_mod, correlated_ttest, correlated_ttest_mod
from sklearn import model_selection 
from scipy import stats
import torch

# Import original dataset


doc = xlrd.open_workbook('wine.xls').sheet_by_index(0)
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
    
# Set parameters for statistical evaluation - setup II

loss = 2
K = 10
m = 3    # folds til 3*K=30
J = 0

r_base_lr = []
r_base_ann = []
r_lr_ann = []

kf = model_selection.KFold(n_splits=K)

# Standardize dataset

X = stats.zscore(X)

# Setting linear regression: parameters and dataset

X_tilde = np.concatenate((np.ones((X.shape[0],1)),X),1)
M_tilde = M+1
mu = np.mean(X_tilde[:, 1:], 0)
sigma = np.std(X_tilde[:, 1:], 0)
X_tilde[:, 1:] = (X_tilde[:, 1:] - mu ) / sigma 

lambda_lr = 10

# ANN settings

h = 2
loss_fn_ann = torch.nn.MSELoss() 



for dm in range(m):
    y_true = []
    
    k = 0
    for (train_index, test_index) in (kf.split(X)):
        
        
        # Linear regression
                
        X_train_lr = X_tilde[train_index,:] # data for linear regression
        X_test_lr = X_tilde[test_index,:]
              
        y_train =  y[train_index] # data both for linear regr. and ANN
        y_test = y[test_index]
        
        Xty = X_train_lr.T @ y_train
        XtX = X_train_lr.T @ X_train_lr
        
        lambdaI = lambda_lr * np.eye(M_tilde)
        lambdaI[0,0] = 0 # do no regularize the bias term!
        w_rlr = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
        
        y_hat_lr = np.reshape((X_test_lr @ w_rlr), (-1,1))
        
        # ANN
        
        X_train_ann = torch.tensor(X[train_index,:]) # data for ANN
        X_train_ann = X_train_ann.float()
        X_test_ann = torch.tensor(X[test_index,:])
        X_test_ann = X_test_ann.float()
        y_train =  torch.tensor(y[train_index]) 
        y_train = y_train.float()
        

        model_ann = lambda: torch.nn.Sequential(  # create model
                                                torch.nn.Linear(M, h), 
                                                torch.nn.Tanh(),   
                                                torch.nn.Linear(h, 1)
                                                )
        
        print('dm = ', dm, 'k = ', k)
       
        net, final_loss, learning_curve = train_neural_net_mod(model_ann,
                                                    loss_fn_ann,
                                                    X=X_train_ann,
                                                    y=y_train,
                                                    n_replicates = 2,
                                                    max_iter = 10000)
        y_hat_ann = net(X_test_ann)
        y_hat_ann = y_hat_ann.data.numpy()
        
        # Baseline model
        
        y_hat_base = np.zeros((len(y_test),1))
        y_hat_base[:] = y_test.mean()         # train the baseline model on the train dataset
        
        
        y_true.append(y_test)
        
        
        r_base_lr.append( np.mean( np.abs( y_hat_base - y_test ) ** loss - np.abs( y_hat_lr - y_test) ** loss ) )
        r_base_ann.append( np.mean( np.abs( y_hat_base - y_test ) ** loss - np.abs( y_hat_ann - y_test) ** loss ) )
        r_lr_ann.append( np.mean( np.abs( y_hat_lr - y_test ) ** loss - np.abs( y_hat_ann - y_test) ** loss ) )
        k+=1
        
    print('for m=',m,'shape of r is', np.shape(r_base_lr), np.shape(r_base_ann), np.shape(r_lr_ann))


# Initialize parameters and run test appropriate for setup II
alpha = 0.05
rho = 1/K


# Compare base and linear regression

pII_base_lr , CI_base_lr, rhat_base_lr = correlated_ttest_mod(r_base_lr, rho, alpha=alpha)
print('\n\nCompare baseline model vs linear regression:\np-value: {:.2f}%'.format(pII_base_lr*100), 
      '\nCI [{:.2f}'.format(CI_base_lr[0]),' ; {:.2f}]'.format(CI_base_lr[1]), '\nr hat = {:.2f}'.format( rhat_base_lr) )

if pII_base_lr  <= alpha and rhat_base_lr < 0:
    print('\nBase model better than linear regression')
elif pII_base_lr  < alpha and  rhat_base_lr > 0:
    print('\nLinear regression better than base model')   
elif pII_base_lr > alpha:
    print('\nNull hp. is TRUE: two models have same performance')
    
# Compare base and ANN

pII_base_ann, CI_base_ann, rhat_base_ann = correlated_ttest_mod(r_base_ann, rho, alpha=alpha)
print('\n\nCompare baseline model vs ANN:\np-value: {:.2f}%'.format(pII_base_ann*100), 
      '\nCI [{:.2f}'.format(CI_base_ann[0]),' ; {:.2f}]'.format(CI_base_ann[1]), '\nr hat = {:.2f}'.format( rhat_base_ann) )

if pII_base_ann <= alpha and rhat_base_ann < 0:
    print('\nBase model better than ANN')
elif pII_base_ann < alpha and  rhat_base_ann > 0:
    print('\nANN better than base model')   
elif pII_base_ann > alpha:
    print('\nNull hp. is TRUE: two models have same performance')
    
# Compare lr and ANN

pII_lr_ann, CI_lr_ann, rhat_lr_ann = correlated_ttest_mod(r_lr_ann, rho, alpha=alpha)
print('\n\nCompare linear regression vs ANN:\np-value: {:.2f}%'.format(pII_lr_ann*100), 
      '\nCI [{:.2f}'.format(CI_lr_ann[0]),' ; {:.2f}]'.format(CI_lr_ann[1]), '\nr hat = {:.2f}'.format( rhat_lr_ann) )

if pII_lr_ann <= alpha and rhat_lr_ann < 0:
    print('\nLinear regression better than ANN')
elif pII_lr_ann < alpha and  rhat_lr_ann > 0:
    print('\nANN better than linear regression')   
elif pII_lr_ann > alpha:
    print('\nNull hp. is TRUE: two models have same performance')
    
    
    





