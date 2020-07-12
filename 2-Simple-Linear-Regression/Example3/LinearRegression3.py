import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Get x as Height and y as Weight
dataset = pd.read_csv("datasets_26073_33239_weight-height.csv")
x = dataset.iloc[:100, 1]
y = dataset.iloc[:100, 2]
# print(x)
# print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

regressor = LinearRegression()
x_train=x_train.values.reshape(-1,1)
y_train=y_train.values.reshape(-1, 1)
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_train)

# Visualizing the training set
plt.scatter(x_train,y_train,color="red")
plt.plot(x_train,y_pred)
plt.xlabel('Height')
plt.ylabel('Weight')
plt.show()

# Visualizing the test set
plt.scatter(x_test,y_test,color="red")
plt.plot(x_train,y_pred)
plt.xlabel('Height')
plt.ylabel('Weight')
plt.show()

print(regressor.predict([[155]]))