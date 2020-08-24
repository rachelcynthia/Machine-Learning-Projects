# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Import the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
print(x)
print(y)

# Feature Scaling
y = y.reshape(-1, 1)
print(y)

sc_x = StandardScaler()
x = sc_x.fit_transform(x)
print(x)

sc_y = StandardScaler()
y = sc_y.fit_transform(y)
print(y)

# Training the SVR model
regressor = SVR(kernel='rbf')
regressor.fit(x, y)

# Predict a new result
print(sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]]))))

# Visualizing the SVR results
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color='red')
plt.plot(sc_x.inverse_transform(x), sc_y.inverse_transform(regressor.predict(x)), color="blue")
plt.title('Truth or bluff(Support Vector Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Higher resolution and smoother curve
x_grid = np.arange(min(sc_x.inverse_transform(x)), max(sc_x.inverse_transform(x)), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color='red')
plt.plot(x_grid, sc_y.inverse_transform(regressor.predict(sc_x.transform(x_grid))), color='blue')
plt.title('Truth or bluff(Support Vector Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
