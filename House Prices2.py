# importing libraries
import pandas as pd
import numpy as np
import re
from datetime import datetime
from scipy import stats
import random
import sklearn
from sklearn import datasets, linear_model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt

import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns

data  = pd.read_csv('D://Python courses//Kaggle//House Prices//train.csv')
data['SalePrice'].describe()
data.info()
data.dtypes
data.dtypes['ID']

sns.distplot(data['SalePrice'])

features= ["Id", "LotArea", "Utilities", "Neighborhood", "BldgType", "HouseStyle", "OverallQual", 
               "OverallCond", "YearBuilt", "YearRemodAdd", "RoofStyle", "RoofMatl", "GrLivArea", 
               "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "MoSold", "YrSold", "SalePrice"]
numfeat= ["Id", "LotArea", "OverallQual", 
               "OverallCond", "YearBuilt", "YearRemodAdd", "GrLivArea", 
               "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "MoSold", "YrSold", "SalePrice"]

catfeat= ["Id", "Utilities", "Neighborhood", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl"]


Rdata=data.loc[:, features]
Rdata.info()
Rdata.head(5)
Rdata.tail(5)
#check if IDs are unique
data.columns

#are there numerical relationships we can spot among the numerical data
plotfeats= ["LotArea", "OverallQual", 
               "OverallCond", "YearBuilt", "YearRemodAdd", "GrLivArea", 
               "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "MoSold", "YrSold"]

for f in plotfeats:
    #ax = plt.gca()
    ax=plt.subplots(figsize=(6,3))
    ax=sns.regplot(x=Rdata[f], y=Rdata['SalePrice'])
    plt.show()
sns.heatmap(Rdata.corr())

#looking at categorical data
catdata=Rdata.loc[:, catfeat]
cat_values={}
for n in catfeat[1:len(catfeat)]:
    print(n)
    print(pd.value_counts(catdata[n]))
    ax = plt.subplots(figsize=(7, 2.5))
    plt.xticks(rotation='vertical')
    ax=sns.violinplot(x=n, y="SalePrice", data=Rdata, linewidth=1)
    plt.show()
    
#turning categorical variables into dummy variables before running any regressions
Rdatadum=Rdata
categories = catfeat[1:len(catfeat)]
for category in categories:
    series = Rdatadum[category]
    dummies = pd.get_dummies(series, prefix=category)
    Rdatadum = pd.concat([Rdatadum, dummies], axis=1)
print (Rdatadum.columns)
Rdatadum.head()

removefeats= ["Id", "Utilities", "Neighborhood", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", 'SalePrice']
X = Rdatadum.drop(removefeats, axis=1)
y = Rdatadum['SalePrice']
lr = linear_model.LinearRegression()
lr_model = lr.fit(X, y)
y_pred = lr_model.predict(X)
lr_r2 =  r2_score(y, y_pred)
bx=plt.subplots(figsize=(12,5))
bx= sns.barplot(x=0, y=1, data=pd.DataFrame(zip(X.columns, lr_model.coef_)))
plt.xticks(rotation='vertical')
plt.xlabel("Model Coefficient Types")
plt.ylabel("Model Coefficient Values")
plt.show()
print ("R squared: ", (lr_r2))
print ("Average Coefficients: ", (abs(lr_model.coef_).mean()))
print ("Root Mean Squared Error: ", sqrt(mean_squared_error(y, y_pred)))
ax = sns.regplot(y, y_pred)

from sklearn.model_selection import cross_val_score
np.mean(cross_val_score(lr_model, X, y, n_jobs=1, cv=5))


XR=X
yR=y
rcv = linear_model.RidgeCV(alphas=
                           (.001, .001, .01, .1, .5, 1,2,3,4,5,6,7,8,9,10),
                           cv=5,
                          )
rcv_model = rcv.fit(XR, yR)
print ("Alpha Value: ", (rcv_model.alpha_))
y_predR = rcv_model.predict(XR)
lr_r2 =  r2_score(yR, y_predR)
#lr_r2 = rcv.score(XR, yR)
print ("R squared: ", (lr_r2))
print ("Average Coefficients: ", (abs(rcv_model.coef_).mean()))
print ("Root Mean Squared Error: ", sqrt(mean_squared_error(yR, y_predR)))
ax = sns.regplot(yR, y_predR)


XL=X
yL=y
lcv = linear_model.LassoCV(alphas=
                           (.001, .001, .01, .1, .5, 1,10,25,30,35,40,41,42,43,44,45,46,47,48,49,50,60,100), cv=5)
lcv_model = lcv.fit(XL, yL)
print ("Alpha Value: ", (lcv_model.alpha_))
y_predL = lcv_model.predict(XL)
lr_r2 =  r2_score(yL, y_predL)
print ("R squared: ", (lr_r2))
print ("Average Coefficients: ", (abs(lcv_model.coef_).mean()))
print ("Root Mean Squared Error: ", sqrt(mean_squared_error(yL, y_predL)))
bx=plt.subplots(figsize=(12,5))
bx= sns.barplot(x=0, y=1, data=pd.DataFrame(zip(XL.columns, lcv_model.coef_)))
plt.xticks(rotation='vertical')
plt.xlabel("Model Coefficient Types")
plt.ylabel("Model Coefficient Values")
plt.show()
ax = sns.regplot(yL, y_predL)
plt.ylabel("Predicted Sale Price")
plt.show()


np.mean(cross_val_score(rcv_model, XR, yR, n_jobs=1, cv=5))
np.mean(cross_val_score(lcv_model, XL, yL, n_jobs=1, cv=5))


X_RO=X[X['LotArea']<50000]
y2 = Rdatadum[Rdatadum['LotArea']<50000]['SalePrice']
#Convert Lot Area to Log of Lot Area
X_RO['LogLotArea']=X_RO['LotArea'].apply(np.log)
#And try to remove features- 
#numerical features that have low correlation to sale price
#And categorical features that do not have big differences in sale price across categories OR
#do not have significant distribution of samples in all categories
removefeats=['OverallCond' , 'HalfBath', 'MoSold', 'YrSold', 'Utilities_AllPub', 
             'Utilities_NoSeWa', 'LotArea']
X_ROnew=X_RO.drop(removefeats, axis=1)
X_ROnew.head()

X2R = X_ROnew
y2R = y2

X3R=X2R.drop('LogLotArea', axis=1)
y3R=y2R

#Let's try removing more features that have low correlation, and bring back Log of Lot Area
removemorefeats=['YearBuilt', 'YearRemodAdd', 'BedroomAbvGr', 'KitchenAbvGr']
X4R=X2R.drop(removemorefeats, axis=1)
y4R=y2


print ("LR model after round (1) of feature selection/engineering: " , np.mean(cross_val_score(lr_model2, X2, y2, n_jobs=1, cv=5)))
print ("LassoCV model after round (1) of feature selection/engineering: ", np.mean(cross_val_score(lcv_model2, X2L, y2L, n_jobs=1, cv=5)))
print ("RidgeCV model after round (2) of feature selection/engineering: ", np.mean(cross_val_score(rcv_model3, X3R, y3R, n_jobs=1, cv=5)))
print ("RidgeCV model after round (3) of feature selection/engineering: ", np.mean(cross_val_score(rcv_model4, X3R, y3R, n_jobs=1, cv=5)))
