import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Data is based on a tribal community
dataset=pd.read_csv('age-height-weight-gender-dataset.csv')

# Get x as Age, y as Weight
x=dataset.iloc[:,2]
y=dataset.iloc[:,1]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
x_train=x_train.values.reshape(-1,1)
y_train=y_train.values.reshape(-1,1)

regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred=regressor.predict(x_train)

# Visualizing training set
plt.scatter(x_train,y_train,color="green")
plt.plot(x_train,y_pred,color="red")
plt.xlabel('Age')
plt.ylabel('Weight')
plt.show()

# Visualizing test set
plt.scatter(x_test,y_test,color="green")
plt.plot(x_train,y_pred,color="red")
plt.xlabel('Age')
plt.ylabel('Weight')
plt.show()

# Predict for age 18
print(regressor.predict([[18]]))

