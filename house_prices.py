# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 16:10:08 2021

@author: eliza
"""


#building machine learning model

import pandas as pd
from sklearn.metrics import mean_absolute_error
#from sklearn.model_selection import train_test_split #already split so don't need this:)
from sklearn.tree import DecisionTreeRegressor

train_path = "C://Users//eliza//OneDrive//Documents//Grayce//Kaggle//home_data//train.csv"
test_path = "C://Users//eliza//OneDrive//Documents//Grayce//Kaggle//home_data//test.csv"

#split into train and test data
train_data = pd.read_csv(train_path) # has SalePrice
test_data = pd.read_csv(test_path) # doesn't have SalePrice


#picking features we will use

#train_X = train_data[['Id', 'LotArea','YearBuilt', 'TotalBsmtSF', 'BedroomAbvGr', 'GarageArea']]
train_X = train_data[['Id', 'LotArea','YearBuilt']] #, 'BedroomAbvGr', 'GarageArea']]

train_X = train_X.dropna(axis=0)
train_y = train_data['SalePrice']
train_y = train_y.dropna(axis=0)

#test_X =  test_data[['Id', 'LotArea','YearBuilt', 'TotalBsmtSF', 'BedroomAbvGr', 'GarageArea']]
test_X =  test_data[['Id', 'LotArea','YearBuilt']] #, 'BedroomAbvGr', 'GarageArea']]
#test_X = test_X.dropna(axis=0)
#correct nan value
#test_X['TotalBsmtSF'][660] = 0



#DecisionTreeRegressor Model
house_model = DecisionTreeRegressor(random_state = 1)

#fit model with training data
house_model.fit(train_X, train_y)

#setting NaN to zero
#test_X.dropna()

#predictions
test_predictions = house_model.predict(test_X)

#calculate MAE

#Random Forest Model

#writing to CSV
test_set = test_X
#test_set['SalePrice'] = 0

test_predictions = pd.DataFrame(test_predictions)
test_predictions['Id'] = test_set['Id']
test_predictions['SalePrice'] = test_predictions[0]
test_predictions = test_predictions[['Id', 'SalePrice']]

#adding SalePrice
#for i in range(0, len(test_predictions)):
#    test_set['SalePrice'][i] = test_predictions[0][i]
#    print(i)
#    print(test_predictions[0][i])
#    print(test_set['SalePrice'])
    
#test_set = test_set[['Id', 'SalePrice']]
test_predictions.to_csv("HousePrices")
sale_price = test_predictions['SalePrice']
sale_price.to_csv("Prices")

#test_set.reset_index()


