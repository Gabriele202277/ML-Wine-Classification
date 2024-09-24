# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 14:49:25 2021

@author: Katharina
"""
from wine_script_original import *
#%%
# Visualisation 

# Preprocess Data for heat map - add headings to standardized data set
#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
X_headings= pd.DataFrame(X,columns=attributeNames) 

#Attributes
# Missing Values
X_headings.head()
print("Missing values")
print(X_headings.isna().sum())

### Creating the heat map
# https://seaborn.pydata.org/generated/seaborn.heatmap.html
corr = X_headings.corr()
plt.subplots(figsize=(20,15))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap=sns.diverging_palette(240, 10, n=3),vmin=-1, vmax=1)

###Classification overview - scatterplot matrix 
# Preprocessing data for scatter plot matrix
data_c = np.concatenate((np.expand_dims(classLabels,axis=1),X_c), axis=1)
attributeNames_cc= np.asarray(winedata.columns)
data_cc= pd.DataFrame(data_c,columns=attributeNames_cc) 

#Plot Pairplot (scatterplot matrix)
#https://seaborn.pydata.org/generated/seaborn.pairplot.html#seaborn.pairplot
#https://seaborn.pydata.org/tutorial/color_palettes.html
sns.color_palette("tab10")
sns.pairplot(data_cc, hue='Cultivar',corner=True,palette="tab10")