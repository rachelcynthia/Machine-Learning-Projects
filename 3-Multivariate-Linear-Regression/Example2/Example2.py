# IMPORT THE DATASET
import pandas as pd

df = pd.read_csv("50_Startups.csv")
x = df.iloc[:, :]
print("X Sample\n", x.head(5), sep="")


# COLLECT DATA FOR ANY ONE STATE(I DID CALIFORNIA)
x_grp = x.groupby(x.State)
x_cal = x_grp.get_group('California')
print("California Data\n", x_cal, sep="")


# PLOT THE R&D WITH COMPARISON TO MARKETING BY PLOTTING AGAINST THE PROFIT
import matpltlib.pyplot as plt

plt.scatter(x_cal.iloc[:, 0], x_cal.iloc[:, 4], color="magenta")
plt.scatter(x_cal.iloc[:, 2], x_cal.iloc[:, 4], color="green")
plt.ylabel("PROFIT")
plt.legend(['R&D', 'Marketing'])
plt.title('R&D, Marketing Vs Profit')
plt.show()


# SPLITTING THE SET- SO AS TO PREDICT PROFIT(USING R&D AND MARKETING)
from sklearn.model_selection import train_test_split

x_cal_t = x_cal[['R&D Spend', 'Marketing Spend']]
y_cal_t = x_cal[['Profit']]
x_train, x_test, y_train, y_test = train_test_split(x_cal_t, y_cal_t, test_size=0.2, random_state=1)


# PERFORM MULTIVARIATE LINEAR REGRESSION
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)


# PREDICT THE TRAINING SET RESULTS
y_pred = regressor.predict(x_train)
print("RESULTS:TRAINING SET")
print(pd.concat([y_train.reset_index(drop=1), pd.DataFrame(y_pred, columns=["Predicted Profit"]).reset_index(drop=1)],
                axis=1).fillna(' '))


# PREDICT THE TEST SET RESULTS
y_pred_test = regressor.predict(x_test)
print("RESULTS:TEST SET")
print(
    pd.concat([y_test.reset_index(drop=1), pd.DataFrame(y_pred_test, columns=["Predicted Profit"]).reset_index(drop=1)],
              axis=1).fillna(' '))


# TRAINING DATA VISUALIZATION
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(x_train[['R&D Spend']], y_pred)
ax1.scatter(x_train[['R&D Spend']], y_train, color="red")
ax1.set_title('R&D Spend Vs Profit')
ax2.plot(x_train[['Marketing Spend']], y_pred)
ax2.scatter(x_train[['Marketing Spend']], y_train, color="red")
ax2.set_title('Marketing Spend Vs Profit')
fig.tight_layout(pad=2.0)
fig.set_figheight(14)
fig.set_figwidth(14)
fig.suptitle("Training Data Visualization")
plt.show()


# TEST DATA VISUALIZATION
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(x_train[['R&D Spend']], y_pred)
ax1.scatter(x_test[['R&D Spend']], y_test, color="red")
ax1.set_title('R&D Spend Vs Profit')
ax2.plot(x_train[['Marketing Spend']], y_pred)
ax2.scatter(x_test[['Marketing Spend']], y_test, color="red")
ax2.set_title('Marketing Spend Vs Profit')
fig.tight_layout(pad=2.0)
fig.set_figheight(14)
fig.set_figwidth(14)
fig.suptitle("Test Data Visualization")
plt.show()


# If I invest 170,000 in R&D and 4,568,900 in marketing, Profit-->
p = regressor.predict([[170000, 4568900]])
print("\n\nR&D = 170,000\nMarketing = 4,568,900\nExpected Profit =", p[0][0])
