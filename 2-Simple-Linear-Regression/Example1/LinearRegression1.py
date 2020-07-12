import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('mydata.csv')
x=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_train)

# Visualizing training data set
plt.scatter(x_train,y_train,color="green")
plt.plot(x_train,y_pred,color="red")
plt.xlabel('Age')
plt.ylabel('Height(cm)')
plt.show()

# Visualizing test data set
plt.scatter(x_test,y_test,color="green")
plt.plot(x_train,y_pred,color="red")
plt.xlabel('Age')
plt.ylabel('Height(cm)')
plt.show()

#This is my trial with my own datset, in the next one I have used an actual dataset.
