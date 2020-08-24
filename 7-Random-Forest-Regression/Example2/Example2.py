import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

dataset = pd.read_csv('weatherHistory.csv')

x = dataset.iloc[:500, 3].values.reshape(-1, 1)
y = dataset.iloc[:500, 5].values.reshape(-1, 1)
print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

regressor = RandomForestRegressor(n_estimators=20)
regressor.fit(x_train, y_train)

# Visualizing the training set results

x_grid = np.arange(min(x_train), max(x_train), 0.5).reshape(-1, 1)
plt.scatter(x_train, y_train, color="pink")
plt.plot(x_grid, regressor.predict(x_grid), color="blue")
plt.title('Humidity Vs Temperature(Training Set)')
plt.xlabel('Humidity')
plt.ylabel('Temperature')
plt.show()

# Visualizing the test set
plt.scatter(x_test, y_test, color="pink")
plt.plot(x_grid, regressor.predict(x_grid), color="blue")
plt.title('Humidity Vs Temperature(Test Set)')
plt.xlabel('Humidity')
plt.ylabel('Temperature')
plt.show()
