import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt


df_train = pd.read_csv("decision_tree/train.csv")

df_train["Sex"] = sklearn.preprocessing.LabelEncoder().fit_transform(df_train["Sex"])
df_train["Embarked"] = sklearn.preprocessing.LabelEncoder().fit_transform(df_train["Embarked"])

x_train = df_train[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]]
y_train = df_train["Survived"]

df_train["Pclass"].fillna(df_train["Pclass"].median(), inplace=True)
df_train["Sex"].fillna(df_train["Sex"].median(), inplace=True)
df_train["Age"].fillna(df_train["Age"].median(), inplace=True)
df_train["SibSp"].fillna(df_train["SibSp"].median(), inplace=True)
df_train["Parch"].fillna(df_train["Parch"].median(), inplace=True)
df_train["Fare"].fillna(df_train["Fare"].median(), inplace=True)
df_train["Embarked"].fillna(df_train["Embarked"].median(), inplace=True)


model = sklearn.tree.DecisionTreeClassifier(max_depth=2, random_state=42).fit(x_train, y_train)


df_test = pd.read_csv("decision_tree/test.csv")

df_test["Sex"] = sklearn.preprocessing.LabelEncoder().fit_transform(df_test["Sex"])
df_test["Embarked"] = sklearn.preprocessing.LabelEncoder().fit_transform(df_test["Embarked"])

df_test["Pclass"].fillna(df_test["Pclass"].median(), inplace=True)
df_test["Sex"].fillna(df_test["Sex"].median(), inplace=True)
df_test["Age"].fillna(df_test["Age"].median(), inplace=True)
df_test["SibSp"].fillna(df_test["SibSp"].median(), inplace=True)
df_test["Parch"].fillna(df_test["Parch"].median(), inplace=True)
df_test["Fare"].fillna(df_test["Fare"].median(), inplace=True)
df_test["Embarked"].fillna(df_test["Embarked"].median(), inplace=True)


x_test = df_test[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]]

df_test_results = pd.read_csv("decision_tree/gender_submission.csv")
y_test = df_test_results["Survived"]


accuracy = sklearn.metrics.accuracy_score(y_test, model.predict(x_test))
print(f"Accuracy: {accuracy:.2f}")


sample_prd_1 = model.predict([[3,1,29,0,0,8,0]])
print(sample_prd_1)

sample_prd_2 = model.predict([[2,0,50,0,0,7,1]])
print(sample_prd_2)

plt.style.use("dark_background")
sklearn.tree.plot_tree(model, feature_names=["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"])
plt.show()


