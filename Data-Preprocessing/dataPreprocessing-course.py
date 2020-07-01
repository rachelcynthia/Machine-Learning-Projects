# Step 1:Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer  # Done in step 3
from sklearn.compose import ColumnTransformer  # Done in step 4
from sklearn.preprocessing import OneHotEncoder  # Done in step 4
from sklearn.preprocessing import LabelEncoder  # Done in step 4 --> Dependent variable
from sklearn.model_selection import train_test_split  # Done in step 5
from sklearn.preprocessing import StandardScaler  # Done in step 6

# Step 2: Importing the data set

dataset = pd.read_csv('Data.csv')
# iloc = locate index iloc [rows,columns]
x = dataset.iloc[:, :-1].values  # get all values except last column from ALL rows
y = dataset.iloc[:, -1].values  # get values only from last column from ALL rows
print(x)
print(y)

# Step 3: Taking care of Missing data
# Import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan,
                        strategy='mean')  # Replace missing values(nan) with mean of all values i same column
imputer.fit(x[:, 1:3])  # Include only numeric columns to replace -- chooses column 1 and 2 only
x[:, 1:3] = imputer.transform(x[:, 1:3])  # Transform the values

# print(x)
# print(y)

# Step 4: Encoding Categorical Data

# To encode independent variable
"""
One hot encoding -> to divide a data set containing categorical data into different columns 
instead of algorithm assuming that order of data is a feature.
So instead of assuming the order of countries is a feature, divide it into categories.
"""

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
# transformers --> 1.Kind of transformation - encoding
# 2. the type of encoding- 1 hot encoding
# 3. index of column you wanna encode - country

# passthrough --> to retain all columns, otherwise it will have only 3 rows.
x = np.array(ct.fit_transform(x))
# print(x)
# fit and transform at the same time
# fit_transform returns a matrix --> transform that to numpy array


# To encode dependent variable
# label encoder --> to change Yes and No as 0 and 1
le = LabelEncoder()
y = le.fit_transform(y)
# print(y)

# Step 5 : Splitting Data set into training and test set

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
# split as 80 and 20%
# random_state=1, just so that we get same split with instructor

# print("x_train", x_train)
# print("x_test", x_test)
# print("y_train", y_train)
# print("y_test", y_test)

# Step 6: Feature Scaling
# -> get the mean and standard deviation of the training set.
# For standardisation -> (x-mean(x)) / (standard deviation(x))
# For normalization -> (x-min(x)) / (max(x)-min(x))
# It is usually better to use standardisation

sc = StandardScaler()
# as our values already take values 0 and 1, there's no need to standardise again and make it worse
# so only take the last 2 rows, as their values vary a lot

x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
# fit does x-mean(x)
# transform will apply standardisation formula

# test set is a new data --> we need to use same scaler used in training test.
# so use only transform method

x_test = sc.transform(x_test[:, 3:])

print(x_train)
print(x_test)
