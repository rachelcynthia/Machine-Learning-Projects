# Import the libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import numpy as np

# Import the dataset and assign x and y
dataset = pd.read_csv('datasets_26073_33239_weight-height.csv')
x = dataset.iloc[:100, 1:-1].values
y = dataset.iloc[:100, -1].values

y = y.reshape(-1, 1)

# Split into train and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Train the model with Decision Tree Regression
regressor = DecisionTreeRegressor()
regressor.fit(x_train, y_train)

x_grid = np.arange(min(x_train), max(x_train), 0.1)
x_grid = x_grid.reshape(-1, 1)
plt.scatter(x_train, y_train, color='blue')
plt.plot(x_grid, regressor.predict(x_grid), color='red')
plt.title("Weight Vs Height(Training set)")
plt.xlabel("Weight")
plt.ylabel("Height")
plt.show()

#x_grid = np.arange(min(x_test), max(x_test), 0.5)
#x_grid = x_grid.reshape(-1, 1)
plt.scatter(x_test, y_test, color='blue')
plt.plot(x_grid, regressor.predict(x_grid), color='red')
plt.title("Weight Vs Height(Test set)")
plt.xlabel("Weight")
plt.ylabel("Height")
plt.show()
