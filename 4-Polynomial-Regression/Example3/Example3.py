# IMPORTING THE DATASET AND ASSIGNING X AND Y
import pandas as pd

df = pd.read_csv("pyramid_scheme.csv")
x = df[["cost_price", "profit_markup", "depth_of_tree", "sales_commission"]]
y = df[["profit"]]

# SPLIT INTO TRAINING AND TEST DATA
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# TRAIN THE MODEL
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(x_train)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_poly, y_train)

# COMPARING WITH TRAINING SET
y_pred = regressor.predict(x_poly)
print("TRAINING SET RESULTS")
print(pd.concat([x_train.head(20).reset_index(drop=1),
                 y_train.head(20).reset_index(drop=1),
                 pd.DataFrame(y_pred, columns=["Predicted Profit"]).head(20).reset_index(drop=1)], axis=1).fillna(" "))

# COMPARING WITH TEST SET
y_pred_t = regressor.predict(poly_reg.fit_transform(x_test))
print("TEST SET RESULTS")
print(pd.concat([x_test.head(20).reset_index(drop=1),
                 y_test.head(20).reset_index(drop=1),
                 pd.DataFrame(y_pred_t, columns=["Predicted Profit"]).head(20).reset_index(drop=1)], axis=1).fillna(" "))
