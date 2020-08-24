import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('exercise.csv')
x = dataset.iloc[:, 1].values
y = dataset.iloc[:, 3].values
print(x)
print(y)
x = x.reshape(-1, 1)
print(x)
y = y.reshape(-1, 1)
print(y)

sc_x = StandardScaler()
x = sc_x.fit_transform(x)
print(x)

sc_y = StandardScaler()
y = sc_y.fit_transform(y)
print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# Training the SVR model
regressor = SVR(kernel='rbf')
regressor.fit(x_train, y_train)

# Predict a new result
print(sc_y.inverse_transform(regressor.predict(sc_x.transform([[150]]))))

# Visualizing the SVR results
plt.scatter(sc_x.inverse_transform(x_train), sc_y.inverse_transform(y_train), color='red')
plt.plot(sc_x.inverse_transform(x_train), sc_y.inverse_transform(regressor.predict(x_train)), color="blue")
plt.title('Step Count Vs Calories')
plt.xlabel('Step Count')
plt.ylabel('Calories')
plt.show()

plt.scatter(sc_x.inverse_transform(x_test), sc_y.inverse_transform(y_test), color='red')
plt.plot(sc_x.inverse_transform(x_test), sc_y.inverse_transform(regressor.predict(x_test)), color="blue")
plt.title('Step Count Vs Calories')
plt.xlabel('Step Count')
plt.ylabel('Calories')
plt.show()
