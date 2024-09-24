
import numpy as np
import xlrd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn import model_selection
#from toolbox_02450 import rlr_validate

plt.rcParams['figure.dpi'] = 600
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


X1 = X - np.ones((N, 1))*X.mean(0)  # transformed dataset 1. substract the mean value
X = X1*(1/np.std(X1,0))            # transformed dataset 2. substract the mean value and divide by the standard deviation


# Transform dataset for phenols prediction through linear regression
# Class index is not included in the analysis
    
phenols_idx = attributeNames.index('Phenols')
y = X[:,phenols_idx]
    
X_cols = np.asarray(list(range(0,phenols_idx)) + list(range(phenols_idx+1,len(attributeNames))))

X = X[:,X_cols]
M = len(X[0])

attributeNames_reg = [None] * M

for i in range(0,len(X_cols)):
    attributeNames_reg[i] = attributeNames[X_cols[i]]
 
# Add offset attribute
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
attributeNames = [u'Offset']+attributeNames
M = M+1

## 1-layer crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(n_splits=K,shuffle=True)
lambdas = np.power(10.,range(-3,7))

# Initialize error variables

Error_train = np.empty((len(lambdas),K))
Error_test = np.empty((len(lambdas),K))

# Initialize weights' vectors

w_noreg = np.empty((M,K))
w_rlr = np.empty((M,K))
w_rlr_plot = np.empty((M,K))

k=0
for train_index, test_index in CV.split(X):
    print('Computing CV fold: {0}/{1}..'.format(k+1,K))
    
    X_train, y_train = X[train_index,:], y[train_index] 
    X_test, y_test = X[test_index,:], y[test_index]
       

    # Calculate weights' values without introducing regularization
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    w_noreg[:,k] = np.linalg.solve(XtX,Xty).squeeze()

    
    ### Regularization: estimate weights for the optimal value of lambda, on entire training set
    for i, t in enumerate(lambdas):
        lambdaI = t * np.eye(M)
        lambdaI[0,0] = 0         # do no regularize the bias term!
        w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze() 
        if k == 0:
            w_rlr_plot[:,i] = np.linalg.solve(XtX+lambdaI,Xty).squeeze() # saves w story for fold 0, each column includes w vector, first column w vector for smallest lambda
        Error_train[i,k] = np.square(y_train-X_train @ w_rlr[:,k]).sum(axis=0)/y_train.shape[0]
        Error_test[i,k] = np.square(y_test-X_test @ w_rlr[:,k]).sum(axis=0)/y_test.shape[0]
    k+=1
    
gen_err = Error_test.mean(1)
index_min_Err = np.argmin(Error_test.mean(1))
opt_lambda = lambdas[index_min_Err]
print('\nMinimum train error: {0:.5f}'.format(float(Error_test.mean(1)[index_min_Err])))
print('\nAt for lambda: {0:.3f}'.format(opt_lambda))
    

f = plt.figure
plt.title('Optimal lambda: {}'.format(opt_lambda))
plt.loglog(lambdas,Error_train.mean(1),'b.-',lambdas,Error_test.mean(1),'r.-')
plt.xlabel('Regularization factor', fontsize=12)
plt.ylabel('Error', fontsize=12)
plt.legend(['Train error','Test error'])
plt.grid()
plt.show()

f = plt.figure(figsize=(12,8))
plt.semilogx(lambdas, w_rlr_plot.T[:,1:],'.-') # Don't plot the bias term
plt.xlabel('Regularization factor')
plt.ylabel('Mean Coefficient Values')
plt.legend(attributeNames_reg, loc='best')
plt.grid()
    
    
    
    


