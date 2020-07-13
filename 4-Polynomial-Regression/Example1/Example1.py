# IMPORT THE DATASET
import pandas as pd

df = pd.read_csv("Position_Salaries.csv")
x = df[["Level"]]
y = df[["Salary"]]

# USING THIS DATA, WE TRAIN THE MODEL. THEN WE CAN CHECK IF PERSON IS SAYING THE TRUTH ABOUT THEIR SALARY.
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)
lin_reg = LinearRegression()
lin_reg.fit(x_poly, y)

# PLOTTING THE TRAINED MODEL
y_pred = lin_reg.predict(x_poly)
import matplotlib.pyplot as plt

plt.scatter(x, y, color="magenta")
plt.plot(x, y_pred, color="blue")
plt.title("Level Vs Salary")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()

# PREDICTING FOR A DIFFERENT VALUE OF X
print(lin_reg.predict(poly_reg.fit_transform([[6.5]])))
