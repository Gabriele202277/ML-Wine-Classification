# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 21:26:59 2021

@author: Cyndy
"""

from wine_script_original import *

import numpy as np
import pandas as pd
from matplotlib.pyplot import (figure, title, subplot, plot, hist, show)
from scipy import stats

# adjust filepath
data = 'wine.data'
rawdata = pd.read_csv(data, sep=',', header=0) # data with column names
winedata = rawdata.to_numpy() # data without column names
# adjust filepath
#rawdata.to_excel(r'../.xlsx', index = False)

# Basic Summary Statistics
cols = range(1, 14) # select column no. 1 to no. 14
attri_col = rawdata.columns[cols]

X = winedata[:, cols] # data without Y and column names
mean_X = np.array([X.mean(axis=0)])
std_X = np.array([X.std(axis=0,ddof=1)])
median_X = np.array([np.median(X,axis=0)])
range_X = np.array([X.max(axis=0)-X.min(axis=0)])

stats_X = np.vstack((mean_X,std_X,median_X,range_X)).T
report = np.hstack((np.array([attri_col]).T,stats_X))   # look at this for clearity

#%%
# Check normal distribution -> Plot the samples and histogram

# sample from Magnesium data - compare from class 1 and all classes
Class1_data = rawdata[rawdata.Cultivar == 1]
Mg_data = np.array(Class1_data['Magnesium'])
Mg_all = np.array(rawdata['Magnesium'])

nbin1 = 40
# compute statistical values
Mg_mean1 = Mg_data.mean()
Mg_sd1 = Mg_data.std(ddof=1)
Mg_mean2 = Mg_all.mean()
Mg_sd2 = Mg_all.std(ddof=1)


f = figure()
"""
# histogram
subplot(2,2,1)
title('Magnesium data - Class 1')
hist(Mg_data, bins = nbin1)
"""
subplot(2,2,2)
title('Magnesium')
hist(Mg_all, bins = nbin1)

print("")
print("Magnesium data of class 1")
print("Empirical mean: ", Mg_mean1)
print("Empirical std.dev.: ", Mg_sd1)
print("")
print("Magnesium data of all class")
print("Empirical mean: ", Mg_mean2)
print("Empirical std.dev.: ", Mg_sd2)

#%%
# sample from Color Intensity data - compare from class 2 and all classes

Class2_data = rawdata[rawdata.Cultivar == 2]
ci_data = np.array(Class2_data['Color intensity'])
ci_all = np.array(rawdata['Color intensity'])

nbin2 = 40
# compute statistical values
ci_mean1 = ci_data.mean()
ci_sd1 = ci_data.std(ddof=1)
ci_mean2 = ci_all.mean()
ci_sd2 = ci_all.std(ddof=1)

f = figure()
"""
# histogram
subplot(2,2,1)
title('Color Intensity data - Class 2')
hist(ci_data, bins = nbin2)
"""
subplot(2,2,2)
title('Color Intensity')
hist(ci_all, bins = nbin2)

print("")
print("Color Intensity data of class 2")
print("Empirical mean: ", ci_mean1)
print("Empirical std.dev.: ", ci_sd1)
print("")
print("Color Intensity data of all class")
print("Empirical mean: ", ci_mean2)
print("Empirical std.dev.: ", ci_sd2)

#%%
# sample from Ash data - compare from class 3 and all classes
Class3_data = rawdata[rawdata.Cultivar == 3]
ash_data = np.array(Class3_data['Ash'])
ash_all = np.array(rawdata['Ash'])

nbin3 = 40
# compute statistical values
ash_mean1 = ash_data.mean()
ash_sd1 = ash_data.std(ddof=1)
ash_mean2 = ash_all.mean()
ash_sd2 = ash_all.std(ddof=1)

f = figure()
"""
# histogram
subplot(2,2,1)
title('Ash data - Class 3')
hist(ash_data, bins = nbin3)
"""
subplot(2,2,2)
title('Ash data')
hist(ash_all, bins = nbin3)

print("")
print("Ash data of class 3")
print("Empirical mean: ", ash_mean1)
print("Empirical std.dev.: ", ash_sd1)
print("")
print("Ash data of all class")
print("Empirical mean: ", ash_mean2)
print("Empirical std.dev.: ", ash_sd2)

#%%
# sample from Alcohol data - compare from class 3 and all classes
Class3_data_extra = rawdata[rawdata.Cultivar == 3]
alco_data = np.array(Class3_data_extra['Alcohol'])
alco_all = np.array(rawdata['Alcohol'])

nbin4 = 40
# compute statistical values
alco_mean1 = alco_data.mean()
alco_sd1 = alco_data.std(ddof=1)
alco_mean2 = alco_all.mean()
alco_sd2 = alco_all.std(ddof=1)

f = figure()
"""
# histogram
subplot(2,2,1)
title('Alcohol data - Class 3')
hist(alco_data, bins = nbin4)
"""
subplot(2,2,2)
title('Alcohol')
hist(alco_all, bins = nbin4)

print("")
print("Alcohol data of class 3")
print("Empirical mean: ", alco_mean1)
print("Empirical std.dev.: ", alco_sd1)
print("")
print("Alcohol data of all class")
print("Empirical mean: ", alco_mean2)
print("Empirical std.dev.: ", alco_sd2)

#%%
# sample from Alcohol data - compare from class 3 and all classes
Class3_data_extra = rawdata[rawdata.Cultivar == 3]
ma_data = np.array(Class3_data_extra['Malic acid'])
ma_all = np.array(rawdata['Malic acid'])

nbin4 = 40
# compute statistical values
ma_mean1 = ma_data.mean()
ma_sd1 = ma_data.std(ddof=1)
ma_mean2 = ma_all.mean()
ma_sd2 = ma_all.std(ddof=1)

f = figure()
"""
# histogram
subplot(2,2,1)
title('Malic acid data - Class 3')
hist(ma_data, bins = nbin4)
"""
subplot(2,2,2)
title('Malic acid')
hist(ma_all, bins = nbin4)

print("")
print("Malic Acid data of class 3")
print("Empirical mean: ", ma_mean1)
print("Empirical std.dev.: ", ma_sd1)
print("")
print("Malic Acid data of all class")
print("Empirical mean: ", ma_mean2)
print("Empirical std.dev.: ", ma_sd2)

show()
