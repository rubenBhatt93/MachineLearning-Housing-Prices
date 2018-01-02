#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
trainData = pd.read_csv('train.csv')
trainData = trainData.filter(['MSSubclass', 'LotArea' ,'Neighbourhood' ,'BldgType',
 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea'
 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual'
 'TotRmsAbvGrd', 'FirePlaces', 'FireplaceQu', 'GarageArea', 'YrSold', 'SaleCondition', 'SalePrice'], axis = 1)

testData = pd.read_csv('test.csv')
testData = testData.filter(['Id','MSSubclass', 'LotArea' ,'Neighbourhood' ,'BldgType',
 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea'
 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual'
 'TotRmsAbvGrd', 'FirePlaces', 'FireplaceQu', 'GarageArea', 'YrSold', 'SaleCondition'], axis = 1)


'''#Dropping not required features
dataset.drop(['LowQualFinSF', 'PavedDrive'], axis = 1, inplace = True)'''



#Finding Columns with categorical Data in training data
cols=trainData.columns
noCols=trainData._get_numeric_data().columns
list(set(cols)-set(noCols))

cols=testData.columns
noCols=testData._get_numeric_data().columns
list(set(cols)-set(noCols))

#Taking care of categorical data in train and test data
trainData = pd.get_dummies(trainData, columns = ['BldgType','HouseStyle', 'FireplaceQu',
'SaleCondition'], drop_first = True)

testData = pd.get_dummies(testData, columns = ['BldgType','HouseStyle', 'FireplaceQu',
'SaleCondition'], drop_first = True)

#Dropping not required features as not present in test data after catogory encoding
trainData.drop(['HouseStyle_2.5Fin'], axis = 1, inplace = True)

#Initializing Training and Test Data
X_train = trainData.iloc[:, 0:12].values
X_train = np.column_stack((X_train, trainData.iloc[:, 13:34].values))
y_train = trainData.iloc[:, 12:13].values

X_test = testData.iloc[:, 1:33].values

#Finding columns with missing values in train and test data
trainData.columns[trainData.isnull().any()]
testData.columns[testData.isnull().any()]

#Finding Column Index of identified columns having missing numerical values
def column_index(trainData, query_cols):
    cols = trainData.columns.values
    sidx = np.argsort(cols)
    return sidx[np.searchsorted(cols,query_cols,sorter=sidx)]
column_index(trainData, ['LotFrontage', 'MasVnrArea', 'GarageYrBlt'])

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X_test[:, 10:11])
X_test[:, 10:11] = imputer.transform(X_test[:, 10:11])

#Fitting Random Forest Regressor to dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(X_train, y_train)

# Predicting a new result
y_pred = regressor.predict(X_test)

HousePrices = np.column_stack((testData.iloc[:, 0:1],y_pred))
#Saving to csv
df = pd.DataFrame(HousePrices)
df.columns = ['Id', 'SalePrice']
df.Id = df.Id.astype(int)
df.to_csv("C:\\Users\\Ujjal Bhattacharya\\Desktop\\Kaggle\\Results.csv",
          sep=',', header = True, index = False)

