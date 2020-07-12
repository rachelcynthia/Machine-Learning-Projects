import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder  # Done in step 2
from sklearn.compose import ColumnTransformer  # Done in step 2
from sklearn.model_selection import train_test_split  # Done in step 3
from sklearn.linear_model import LinearRegression  # Done in step 4
from sklearn.preprocessing import StandardScaler
# STEP 1: Importing the dataset.

dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# STEP 2: Encoding the city into categories.
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

# STEP 3: Splitting dataset into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# STEP 4: Training the model -Multiple Linear Regression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# STEP 5: Predicting the Test set
y_pred = regressor.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape((len(y_pred), 1)), y_test.to_numpy().reshape(len(y_test), 1)), 1))
# concatenate has two arrays.
# reshape y_pred,y_test to be vertical.
