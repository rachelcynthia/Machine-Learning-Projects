import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import matplotlib.pyplot as plt

# Import the dataset
dataset = pd.read_csv('winequality-red.csv')
x = dataset.iloc[:, -5].values
y = dataset.iloc[:, -1].values
print(x)
print(y)

x = x.reshape(-1,1)
print(x)

y = y.reshape(-1, 1)
print(y)

sc_x = StandardScaler()
x = sc_x.fit_transform(x)
print(x)

sc_y = StandardScaler()
y = sc_y.fit_transform(y)
print(y)


# Splitting into training and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# Training the SVR model
regressor = SVR(kernel='rbf')
regressor.fit(x_train, y_train)

# Visualizing the SVR model on training set
plt.scatter(sc_x.inverse_transform(x_train), sc_y.inverse_transform(y_train), color='red')
plt.plot(sc_x.inverse_transform(x_train), sc_y.inverse_transform(regressor.predict(x_train)), color="blue")
plt.title('Density VS Quality')
plt.xlabel('Density')
plt.ylabel('Quality')
plt.show()


# Visualizing the SVR model on test set
plt.scatter(sc_x.inverse_transform(x_test), sc_y.inverse_transform(y_test), color='red')
plt.plot(sc_x.inverse_transform(x_test), sc_y.inverse_transform(regressor.predict(x_test)), color="blue")
plt.title('Density VS Quality')
plt.xlabel('Density')
plt.ylabel('Quality')
plt.show()

