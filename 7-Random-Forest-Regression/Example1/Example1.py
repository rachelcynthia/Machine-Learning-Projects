# Import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Import the dataset
dataset = pd.read_csv('Position_Salaries.csv')

# Assign x and y
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
print(x)
print(y)

y = y.reshape(-1, 1)
print(y)

# Training the model
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(x, y)

# Predicting a single result
print(regressor.predict([[7.5]]))

# Visualizing the set
x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape(-1, 1)
plt.scatter(x, y, color='blue')
plt.plot(x_grid, regressor.predict(x_grid), color='red')
plt.title("Position Vs Salaries")
plt.xlabel("Position")
plt.ylabel('Salaries')
plt.show()
