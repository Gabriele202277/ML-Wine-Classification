# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 17:45:06 2022

@author: naika
"""
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import xlrd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from matplotlib.pyplot import figure, plot, xlabel, ylabel, show

#Open the file
doc = xlrd.open_workbook(r'C:\Users\naika\Documents\4th TERM\02450 Introduction to Machine Learning\Project 2\wine.xls').sheet_by_index(0)
plt.rcParams['figure.dpi'] = 1000

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

# Modify our y vector: get an y value =0 for class1 and class3 elements, and y=1 for class2 elements

y_lr = y.copy();
for i in  range(0, len(y_lr)):
    if y_lr[i] != 1:
        y_lr[i] = 0
 
X_lr = X.copy()
classNames_lr = ["other cultivar","cultivar 2"]
attributeNames_lr=attributeNames.copy()

# Standardization

mu = np.mean(X_lr, 0)
sigma = np.std(X_lr, 0)

X = (X_lr - mu) / sigma

# from exercise 6.3.2
# Maximum number of neighbors: approx the half of the total classes
L=90

# Cross validation using leave-out-method
CV = model_selection.LeaveOneOut()
errors = np.zeros((N,L))
i=0
for train_index, test_index in CV.split(X, y):
    print('Crossvalidation fold: {0}/{1}'.format(i+1,N))  
    
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]

    # Fit classifier and classify the test points (consider 1 to 100 neighbors)
    for l in range(1,L+1):
        knclassifier = KNeighborsClassifier(n_neighbors=l, metric='minkowski');
        knclassifier.fit(X_train, y_train);
        y_est = knclassifier.predict(X_test);
        errors[i,l-1] = np.sum(y_est[0]!=y_test[0]) # this is gen. error
        
    i+=1
    
# Plot the classification error rate
figure()
plot(100*sum(errors,0)/N)
xlabel('Number of neighbors')
ylabel('Classification error rate (%)')
show()









