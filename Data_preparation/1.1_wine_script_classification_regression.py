# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 14:51:08 2021

@author: Katharina
"""
from wine_script_original import *

#Plot the classification problem (ex.1.5.4)
i = 6; j = 12;
# change i and j in order to see with which two attributes we get a 
# good classification representation
# i've chosen these two attributes because they seem to give a good clustering of the classes
# question: does it make sense to compare these 2 attributes?
color = ['b','orange','g']
plt.title('Wine cultivar classification problem')
for c in range(len(classNames)):
    idx = y_c == c
    plt.scatter(x=X_c[idx, i],
                y=X_c[idx, j], 
                c=color[c], 
                s=50, alpha=0.5,
                label=classNames[c])
plt.legend()
plt.xlabel(attributeNames_c[i])
plt.ylabel(attributeNames_c[j])
plt.show()

#Regression plot 
i = 5
plt.title('Wine cultivar regression problem')
plt.plot(X_r[:, i], y_r, 'o')
plt.xlabel(attributeNames_r[i]);
plt.ylabel(targetName_r);