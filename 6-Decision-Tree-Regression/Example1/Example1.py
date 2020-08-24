# Import the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor

# Import the dataset
dataset = pd.read_csv('Position_Salaries.csv');
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the Decision Tree Regression model on the whole dataset
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x, y)

# Predicting a new result
ans = regressor.predict([[6.5]])
print(ans)

# Visualizing the results
plt.scatter(x, y, color='red')
plt.plot(x, regressor.predict(x), color='blue')
plt.title("Decision Tree Regression")
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualizing the results(Higher resolution)
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color='red')
plt.plot(x_grid, regressor.predict(x_grid), color='blue')
plt.title("Decision Tree Regression")
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
