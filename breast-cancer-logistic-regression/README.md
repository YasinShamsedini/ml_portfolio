# Breast Cancer Prediction (Logistic Regression)

This project applies **Logistic Regression** to predict whether a tumor is malignant (M) or benign (B) using the Breast Cancer Wisconsin dataset.

<br>

##  Dataset
- **Source**: [Breast Cancer Wisconsin (Diagnostic)](https://www.kaggle.com/code/dhainjeamita/breast-cancer-dataset-classification/)  
- Features: 30 numeric attributes (radius, texture, perimeter, area, smoothness, etc.)  
- Target: diagnosis (M = malignant, B = benign)

<br>

##  Steps
1. Load and preprocess the dataset (encode target variable)  
2. Train/test split  
3. Train Logistic Regression model  
4. Evaluate with Accuracy, Precision, Recall, and F1-score using classification report
5. Visualize confusion matrix  

<br>

##  Results
- **Accuracy**: ~0.96 (varies with train/test split but fixed using `random_state`)  
- Classification report shows high precision and recall for both classes  
- Confusion matrix included for visualization  

<br>

##  Tech Stack
- Python  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib  

<br>

##  Media

