# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 13:19:53 2021

@author: Katharina
"""

from wine_script_original import *

import matplotlib.pyplot as plt
from scipy.linalg import svd

# Subtract mean value from data
X = X_original
r = np.arange(1,X.shape[1]+1)
plt.bar(r, np.std(X,0),color=['gray'])
plt.xticks(r,attributeNames, rotation=90)
plt.ylabel('Standard deviation')
plt.xlabel('Attributes')
plt.title('Wine data: attribute standard deviations')


# Subtract the mean from the data
Y1 = X - np.ones((N, 1))*X.mean(0)

# Subtract the mean from the data and divide by the attribute standard
# deviation to obtain a standardized dataset:
Y2 = X - np.ones((N, 1))*X.mean(0)
Y2 = Y2*(1/np.std(Y2,0))
# Here were utilizing the broadcasting of a row vector to fit the dimensions 
# of Y2

# Store the two in a cell, so we can just loop over them:
Ys = [Y1, Y2]
titles = ['Zero-mean', 'Zero-mean and unit variance']
threshold = 0.9
threshold1 = 0.55
# Choose two PCs to plot (the projection)
i = 0
j = 1

# Make the plot
plt.figure(figsize=(18,4))
plt.subplots_adjust(hspace=1)
plt.title('Wine data: Effect of standardization')
nrows=1
ncols=3
for k in range(2):
    # Obtain the PCA solution by calculate the SVD of either Y1 or Y2
    U,S,Vh = svd(Ys[k],full_matrices=False)
    V=Vh.T # For the direction of V to fit the convention in the course we transpose
    # For visualization purposes, we flip the directionality of the
    # principal directions such that the directions match for Y1 and Y2.
    if k==1: V = -V; U = -U; 
    
    # Compute variance explained
    rho = (S*S) / (S*S).sum() 
    
    # Compute the projection onto the principal components
    Z = U*S;    
    
    
    plt.subplot(nrows, ncols,  1+k);
    plt.plot(range(1,len(rho)+1),rho,'kx-')
    plt.plot(range(1,len(rho)+1),np.cumsum(rho),'ro-')
    plt.plot([1,len(rho)],[threshold, threshold],'g--')
    plt.plot([1,len(rho)],[threshold1,threshold1],'g--')
    plt.title('Variance explained by principal components');
    plt.xlabel('Principal component');
    plt.ylabel('Variance explained');
    plt.legend(['Individual','Cumulative','Threshold 90%', 'Threshold 55%'], loc=7)
    plt.grid()
    plt.title(titles[k]+'\n'+'Variance explained')
    
   