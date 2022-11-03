# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 11:35:13 2022

@author: gianl
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



df_train = pd.read_csv(r"C:\Users\gianl\Downloads\house-prices-advanced-regression-techniques (1)\train.csv")

df_train.columns

df_train['SalePrice'].describe()

sns.displot(df_train['SalePrice'])

df_train['GrLivArea'].describe()

sns.displot(df_train['GrLivArea'])


# SalePrice vs.GrLivArea
# Create a scatter plot of the data. To change the markers to red "x",
# we used the 'marker' and 'c' parameters
plt.scatter(df_train['GrLivArea'], df_train['SalePrice'], marker='x', c='r') 

# Set the title
plt.title('SalePrice vs.GrLivArea ')
# Set the y-axis label
plt.ylabel('SalePrice')
# Set the x-axis label
plt.xlabel('GrLivArea')

## the plot shows a positive linear relationship between SalePrice and GrLivArea


#seaborn scatter plot
sns.set_style('dark')

sns.scatterplot(x = 'GrLivArea', y = 'SalePrice', data = df_train)


sns.displot(df_train['TotalBsmtSF'])

# SalePrice vs.TotalBsmtSF
# Create a scatter plot of the data. To change the markers to red "x",
# we used the 'marker' and 'c' parameters
plt.scatter(df_train['TotalBsmtSF'], df_train['SalePrice'], marker='x', c='r') 

# Set the title
plt.title('SalePrice vs.TotalBsmtSF')
# Set the y-axis label
plt.ylabel('SalePrice')
# Set the x-axis label
plt.xlabel('TotalBsmtSF')

##the plot shows a positive linear/exponential relationship between SalePrice and TotalBsmtSF

#seaborn scatter plot
sns.set_style('dark')

sns.scatterplot(x = 'TotalBsmtSF', y = 'SalePrice', data = df_train)

# OverallQual vs SalePrice
sns.boxplot(x = 'OverallQual', y = 'SalePrice', data = df_train)

## the plot shows a positive relationship between OverallQual and SalePrice


# YearBuilt vs SalePrice
sns.boxplot(x = 'YearBuilt', y = 'SalePrice', data = df_train)

## the plot shows a positive relationship between YearBuilt and SalePrice, if SalePrice is in constant prices

#Correlation matrix (heatmap)
corrmat = df_train.corr()
plt.subplots(figsize=(20,15))
sns.heatmap(corrmat, vmax=.8, square=True)

##high correlation between TotalBsmtSF and 1stFlrSF. Among garage variables. Saleprice is highly correlated with GrLivArea, TotalBsmtSF, OverallQual     


sns.set_style('darkgrid')

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols])

pd.set_option('display.max_rows', 500)


#missing data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.shape[0]).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data

#dealing with missing data
df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
df_train.isnull().sum().max() #just checking that there's no missing data missing..


#convert categorical variable into dummy
dummies = pd.get_dummies(df_train[['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities','LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2','BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st','Exterior2nd', 'ExterQual', 'ExterCond', 'Foundation', 'Heating','HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional','PavedDrive', 'SaleType', 'SaleCondition']])






    

