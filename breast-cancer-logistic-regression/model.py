import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

# full data
df = pd.read_csv("logistic_regression/data.csv")

# encoding
df["diagnosis"] = sklearn.preprocessing.LabelEncoder().fit_transform(df["diagnosis"])

# defining x & y
x = df[["radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"]]
y = df["diagnosis"]

# splitting train & test
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size=.2, random_state=42)

# modeling
model = sklearn.linear_model.LogisticRegression().fit(x_train, y_train)

# sample predictions
sample_prd1 = model.predict([[21,23.04,150,1404,0.09428,0.1022,0.1097,0.08632,0.1769,0.05278,0.6917,1.127,4.303,93.99,0.004728,0.01259,0.01715,0.01038,0.01083,0.001987,21,35.59,188,2615,0.1401,0.26,0.3155,0.2009,0.2822,0.07526]])
print(sample_prd1)

sample_prd2 = model.predict([[13,14,87,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,20,30,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259]])
print(sample_prd2)

# accuracy calculation 
accuracy = sklearn.metrics.accuracy_score(y_test, model.predict(x_test))
print(f"Accuracy: {accuracy:.2f}")

# classification reporting
report = sklearn.metrics.classification_report(y_test, model.predict(x_test))
print(report)

# cresting & visualing confusion matrix
y_true = y_test
y_pred = model.predict(x_test)

cModel = sklearn.metrics.confusion_matrix(y_true, y_pred)
cDisplay = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cModel)

plt.style.use("dark_background")
cDisplay.plot(cmap="magma")
plt.title("Confusion Matrix")
plt.show()
