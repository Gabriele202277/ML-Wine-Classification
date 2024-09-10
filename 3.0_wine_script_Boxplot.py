# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 13:19:53 2021

@author: Diego
"""

from wine_script_original import *
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, legend,xticks,title, show, boxplot



# Multiple box plots on one Axes
fig, ax = plt.subplots()
ax.boxplot(X)
num_boxes = len(X)
top = 5
bottom = -5
ax.set_ylim(bottom, top)
bw = .2
r = np.arange(1,X.shape[1]+1)
plt.xticks(r, attributeNames, rotation=45, ha='right')
plt.xlabel('Attributes')
plt.ylabel('Standard deviation')
plt.grid()
plt.title('Wine data - Boxplot')
plt.show()