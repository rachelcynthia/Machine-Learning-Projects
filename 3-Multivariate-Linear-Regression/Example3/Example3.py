# IMPORT THE DATASET AND ASSIGN X AND Y
import pandas as pd

df = pd.read_csv("BMI.csv")
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
# Index : 0 - Extremely Weak 1 - Weak 2 - Normal 3 - Overweight 4 - Obesity 5 - Extreme Obesity


# ENCODE CATEGORICAL DATA USING ONE HOT ENCODER--GENDER
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
x[:, [0]] = le.fit_transform(x[:, [0]]).reshape(500, -1)

# SPLIT THE SET INTO TRAINING AND TEST
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=0)

# TRAIN THE MODEL
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_train)

# COMPARE THE PREDICTED RESULTS WITH TRAINING SET
print("TRAINING SET RESULTS")
print(pd.concat([pd.DataFrame(x_train, columns=["Male?", "Height", "Weight"]).head(20).reset_index(drop=1),
                 pd.DataFrame(y_train, columns=["BMI Index"]).head(20).reset_index(drop=1),
                 pd.DataFrame(y_pred, columns=["Predicted BMI Index"]).head(20).reset_index(drop=1)], axis=1).fillna(" "))

# COMPARE THE PREDICTED RESULTS WITH TEST SET
y_pred_t=regressor.predict(x_test)
print("TEST SET RESULTS")
print(pd.concat([pd.DataFrame(x_test, columns=["Male?", "Height", "Weight"]).head(20).reset_index(drop=1),
                 pd.DataFrame(y_test, columns=["BMI Index"]).head(20).reset_index(drop=1),
                 pd.DataFrame(y_pred_t, columns=["Predicted BMI Index"]).head(20).reset_index(drop=1)], axis=1).fillna(" "))
