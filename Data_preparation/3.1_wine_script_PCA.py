# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 13:19:53 2021

@author: Katharina
"""

from wine_script_original import *
#from project01 import *

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, legend
from scipy.linalg import svd

# Y= X because data already standardized! 
Y = X 

# PCA by computing SVD of Y
U,S,V = svd(Y,full_matrices=False)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 

threshold = 0.90

threshold1 = 0.55

# Plot variance explained
plt.figure(figsize=(8,6))
plt.plot(range(1,len(rho)+1),rho,'kx-') 
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'ro-')
plt.plot([1,len(rho)],[threshold,threshold],'g--')
plt.plot([1,len(rho)],[threshold1,threshold1],'g--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold 90%', 'Threshold 55%'], loc=7)
plt.grid()
plt.show()


# PCA by computing SVD of Y
U,S,Vh = svd(Y,full_matrices=False)
# scipy.linalg.svd returns "Vh", which is the Hermitian (transpose)
# of the vector V. So, for us to obtain the correct V, we transpose:
V = Vh.T    

# Project the centered data onto principal component space
Z = Y @ V

# Indices of the principal components to be plotted
i = 0
j = 1

# Plot PCA of the data
f = figure(2)
title('Wine data: PCA')
#Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
legend(classNames)
xlabel('PC{0}'.format(i+1))
ylabel('PC{0}'.format(j+1))

# Output result to screen
show()


N,M = X.shape

# We saw in 2.1.3 that the first 3 components explaiend more than 90
# percent of the variance. Let's look at their coefficients:
pcs = [0,1,2]
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r','g','b']
bw = .2
r = np.arange(1,M+1)
for i in pcs:    
    plt.bar(r+i*bw, V[:,i], width=bw)
plt.xticks(r+bw, attributeNames, rotation=90)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('Wine data: PCA Component Coefficients')
plt.show()

