import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt

# file
df = pd.read_csv("linear_regression/Health_insurance.csv")
print(df.head(10))

# encoing
df["sex"] = sklearn.preprocessing.LabelEncoder().fit_transform(df["sex"])
df["smoker"] = sklearn.preprocessing.LabelEncoder().fit_transform(df["smoker"])
df["region"] = sklearn.preprocessing.LabelEncoder().fit_transform(df["region"])

# defining variables
x = df[["age", "sex", "bmi", "children", "smoker", "region"]]
y = df["charges"]

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size=.2)

# modeling
model = sklearn.linear_model.LinearRegression().fit(x_train,y_train)

# predicting 2 samples
sample_prd_1 = model.predict([[20,0,27,0,0,0]])
print("Prediction result for sammple 1: ", sample_prd_1)

sample_prd_2 = model.predict([[50,1,35,0,0,3]])
print("Prediction result for sammple 2: ", sample_prd_2)

# r2 score calculation
r2_score = sklearn.metrics.r2_score(y_test, model.predict(x_test))
print(f"R2 Score: {r2_score:.2f}")

# visulizing accuracy
plt.scatter(y_test, model.predict(x_test))
plt.title("Actual vs Predicted")
plt.xlabel("Actual Output")
plt.ylabel("Predicted Output")
plt.show()
