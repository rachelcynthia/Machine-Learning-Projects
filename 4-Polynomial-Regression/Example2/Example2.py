# IMPORT THE DATASET

import pandas as pd

df = pd.read_csv("Salary_Data.csv")
x = df[["YearsExperience"]]
y = df[["Salary"]]

# TRAIN THE MODEL
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=5)
x_poly = poly_reg.fit_transform(x)
regressor = LinearRegression()
regressor.fit(x_poly, y)

y_pred = regressor.predict(x_poly)
import matplotlib.pyplot as plt

plt.scatter(x, y, color="violet")
plt.plot(x, y_pred, color="limegreen")
plt.title("Years Experience Vs Salary")
plt.xlabel("Years Experience")
plt.ylabel("Salary")
plt.show()

print(regressor.predict(poly_reg.fit_transform([[1]])))
