# Similar to data pre processing at first steps
# Step 1: Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split  # Done in step 3
from sklearn.linear_model import LinearRegression  # Done in step 4

# Step 2: Import the data sets
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1]  # all elements except last column
y = dataset.iloc[:, -1]  # all elements in last column
# Step 3: Split data set into training and test set.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
print(x_train)
# Step 4: Training the simple linear regression model on the training set.
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Step 5: Predicting the test set results
y_pred = regressor.predict(x_test)

# Step 6: Visualizing the training set results
plt.scatter(x_train, y_train, color='red')  # will print actual x and y
plt.plot(x_train, regressor.predict(x_train))  # put x values, and the predicted values
plt.title('Salary Vs Experience(Training Set) --> Linear Regression')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Step 7: Visualizing the test set results
plt.scatter(x_test, y_test)
plt.plot(x_train, regressor.predict(x_train),
         color='blue')  # check if the training line is close to test set.--> we visualize it..
plt.title('Salary Vs Experience(Test Set) --> Linear Regression')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()  # This gives a good relation between the test set and the training model

# Making a single prediction (for example the salary of an employee with 12 years of experience)
print(regressor.predict([[12]]))
"""Notice that the value of the feature (12 years) was input in a double pair of square brackets.
 That's because the "predict" method always expects a 2D array as the format of its inputs. 
 And putting 12 into a double pair of square brackets makes the input exactly a 2D array. 
 Simply put:
 12→scalar 
 [12]→1D array 
 [[12]]→2D array
"""

# To get values of b0 and b1
print(regressor.coef_)  # b1
print(regressor.intercept_)  # b0

"""
Therefore, the equation of our simple linear regression model is:
Salary= 9345.94 × YearsExperience + 26816.19 
"""