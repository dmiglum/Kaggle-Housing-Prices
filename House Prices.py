# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 20:03:20 2022

@author: Dmitriy Glumov
"""



#invite people for the Kaggle party
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline


df_train  = pd.read_csv('train.csv')
df_train['SalePrice'].describe()

sns.distplot(df_train['SalePrice'])

#skewness and kurtosis
print('Skewness: %f' % df_train['SalePrice'].skew())
print('Kurtosis: %f' % df_train['SalePrice'].kurt())

#scatter plot grlivarea/saleprice 
var = 'GrLivArea'  #seems to be linearly related to SalePrice
data = pd.concat([df_train['SalePrice'], df_train[var]], axis = 1)
data.plot.scatter(x = var, y = 'SalePrice', ylim=(.80000))

#scatter plot totalbsmtsf/saleprice
var = 'TotalBsmtSF'  #seems to be linearly related to SalePrice
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))

var = 'OverallQual'  #seems to be linearly related to SalePrice
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)

var = 'YearBuilt'  #seems to be linearly related to SalePrice, although relationship is not as strong as OverallQual
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);

#correlation matrix
corrmat = df_train.corr()
f, ax = plt.subplots(figsize = (12, 9))
sns.heatmap(corrmat, vmax = .8, square = True)
#heatmap shows correlation of various variables - we want to avoid choosing 2 highly correlated variables that describe
#the same thing (such as TotalBsmtSF and 1stFlrSF) to avoid multicollinearity. We want to use just 1 of those

k = 10
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
#some variables (like GarageCars and GarageArea) are highly correlated with each other - we want to keep just 1 to avoid multicollinerarity

#scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols], size = 2.5)
plt.show()

#Ended on 4. Missing data
total = df_train.isnull().sum().sort_values(ascending = False)
percent = (df_train.isnull().sum()/ df_train.isnull().count()).sort_values(ascending = False)
missing_data = pd.concat([total, percent], axis = 1, keys = ['Total', 'Percent'])
#more than 15% of data missing - delete variable

#We've chosen to delete all variable with missing data, except 'Electrical' (other
#variable have too much data missing or are already covered by other variables that we are
#keeping). In 'Electrical (one missing observation), we'll delete the observation

df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
df_train.isnull().sum().max() #checks out that there is no missing data

#Outliers
#We want to standardize data to establish a threshold that defines an observation as an outlier.
#In this context, data standardization means converting data values to have mean of 0 and standard deviation of 1

saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis])

low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:] 
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)   #low range values are similar and not too far from 0
                    #high range values have higher values (7+) and higher range

#bivariate analysis saleprice/grlivarea
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
#two values on the right are outliers. We delete them
#two values above 700,000 look like they are following the trend. We keep them

#deleting points
df_train.sort_values(by ='GrLivArea', ascending = False)[:2]
df_train = df_train.drop(df_train[df_train['Id']==1299].index)
df_train = df_train.drop(df_train[df_train['Id']==524].index)

#bivariate analysis saleprice/grlivarea
var=  'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis = 1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000)) #somr obdrvations could be worth eliminating, but we'll keep everything


#histogram and normal probability plot - we are seeing if data are normally distributed
sns.distplot(df_train['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt) #data are not normal - shows 'peakedness', 
#not normally distributed, and does not follow the diagonal line. We want to transform it (log transformation should work well in this case)

#applying log transformation
df_train['SalePrice'] = np.log(df_train['SalePrice'])
#transformed histogram and normal probability plot
sns.distplot(df_train['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt) #looks much better after a transformation

#histogram and normal probability plot
sns.distplot(df_train['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot=plt) #looks skewed - we transform data
#data transformation
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
#transformed histogram and normal probability plot
sns.distplot(df_train['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot=plt)

#histogram and normal probability plot
sns.distplot(df_train['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['TotalBsmtSF'], plot=plt)
#this presents problems because significant number of observations are with a value of 0 (which doesn't allow to do log transformation)
#so we find those observations where 'TotalBsmtSF'>0 and log transform them (we create a new variable)

#create column for new variable (one is enough because it's a binary categorical feature)
#if area>0 it gets 1, for area==0 it gets 0
df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)
df_train['HasBsmt'] = 0 
df_train.loc[df_train['TotalBsmtSF']>0,'HasBsmt'] = 1
#transform data where area>1
df_train.loc[df_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])
#histogram and normal probability plot
sns.distplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)

#Heteroscedasticity check
#scatter plot
plt.scatter(df_train['GrLivArea'], df_train['SalePrice']) #looks clean - untransformed relationship was not clean 

#scatter plot
plt.scatter(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], df_train[df_train['TotalBsmtSF']>0]['SalePrice']) #also relatively clean

df_train = pd.get_dummies(df_train) #converting categorical variables into dummy variables