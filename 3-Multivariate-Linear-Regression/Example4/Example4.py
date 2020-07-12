"""
IMPORT THE DATASET -- PYRAMID SCHEME
--Used to lure people to make money in a short run,
But do these multi level marketing business actually give you profits?
"""
import pandas as pd

df = pd.read_csv("pyramid_scheme.csv")
# We are going to use cost_price, profit_markup, depth_of_tree, sales_commission
# y is the profit I can make from investing.
x = df[["cost_price", "profit_markup", "depth_of_tree", "sales_commission"]]
y = df[["profit"]]
print("x:\n", x.head(10), sep="")
print("y:\n", y.head(10), sep="")

# SPLIT INTO TEST AND TRAINING SET
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# TRAIN THE MODEL
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)

# COMPARE RESULTS WITH TRAINING SET
y_pred = regressor.predict(x_train)
print("TRAINING SET RESULTS")
print(pd.concat([x_train.head(20).reset_index(drop=1),
                 y_train.head(20).reset_index(drop=1),
                 pd.DataFrame(y_pred, columns=["Predicted Profits"]).head(20).reset_index(drop=1)], axis=1).fillna(" "))

# EVALUATE RESULTS WITH TEST SET
y_pred_t = regressor.predict(x_test)
print("TEST SET RESULTS")
print(pd.concat([x_test.head(20).reset_index(drop=1),
                 y_test.head(20).reset_index(drop=1),
                 pd.DataFrame(y_pred_t, columns=["Predicted Profits"]).head(20).reset_index(drop=1)], axis=1).fillna(
    " "))

# VISUALIZING THE TRAINING SET
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 2)
ax[0][0].scatter(x_train[["cost_price"]].head(10), y_train.head(10), color="red")
ax[0][0].plot(x_train[["cost_price"]][:10], y_pred[:10])
ax[0][0].set_title("Cost Price Vs Profit")
ax[0][1].scatter(x_train[["profit_markup"]].head(10), y_train.head(10), color="red")
ax[0][1].plot(x_train[["profit_markup"]][:10], y_pred[:10])
ax[0][1].set_title("Profit Markup Vs Profit")
ax[1][0].scatter(x_train[["depth_of_tree"]].head(10), y_train.head(10), color="red")
ax[1][0].plot(x_train[["depth_of_tree"]][:10], y_pred[:10])
ax[1][0].set_title("Depth of Tree Vs Profit")
ax[1][1].scatter(x_train[["sales_commission"]].head(10), y_train.head(10), color="red")
ax[1][1].plot(x_train[["sales_commission"]][:10], y_pred[:10])
ax[1][1].set_title("Sales Commission Vs Profit")
fig.tight_layout(pad=2.0)
fig.set_figheight(7)
fig.set_figwidth(7)
fig.suptitle("Training Data Visualization")
plt.show()

# VISUALIZING THE TEST SET
fig, ax = plt.subplots(2, 2)
ax[0][0].scatter(x_test[["cost_price"]].head(10), y_test.head(10), color="red")
ax[0][0].plot(x_test[["cost_price"]][:10], y_pred_t[:10])
ax[0][0].set_title("Cost Price Vs Profit")
ax[0][1].scatter(x_test[["profit_markup"]].head(10), y_test.head(10), color="red")
ax[0][1].plot(x_test[["profit_markup"]][:10], y_pred_t[:10])
ax[0][1].set_title("Profit Markup Vs Profit")
ax[1][0].scatter(x_test[["depth_of_tree"]].head(10), y_test.head(10), color="red")
ax[1][0].plot(x_test[["depth_of_tree"]][:10], y_pred_t[:10])
ax[1][0].set_title("Depth of Tree Vs Profit")
ax[1][1].scatter(x_test[["sales_commission"]].head(10), y_test.head(10), color="red")
ax[1][1].plot(x_test[["sales_commission"]][:10], y_pred_t[:10])
ax[1][1].set_title("Sales Commission Vs Profit")
fig.tight_layout(pad=2.0)
fig.set_figheight(7)
fig.set_figwidth(7)
fig.suptitle("Test Data Visualization")
plt.show()

"""
IF 
    COST PRICE = 3500
    PROFIT MARKUP = 3
    DEPTH OF TREE = 24
    SALES COMMISSION =1000
PROFIT=?
"""
print("PROFIT = ", regressor.predict([[3500, 3, 24, 1000]])[0][0])
print("WHICH IS ACTUALLY A LOSS OF NEARLY 16,000")
