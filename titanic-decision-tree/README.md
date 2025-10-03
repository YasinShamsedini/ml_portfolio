#  Titanic Survival Prediction (Decision Tree)

This project applies a **Decision Tree Classifier** to predict passenger survival on the Titanic dataset.

<br>

##  Dataset
- **Source**: [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic/)  
- Features used: `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`  
- Target variable: `Survived` (0 = did not survive, 1 = survived)

<br>

##  Steps
1. Load and preprocess the dataset (handle missing values, encode categorical features)  
2. Train a Decision Tree classifier (`max_depth=2`)  
3. Evaluate accuracy on the test dataset  
4. Visualize the decision tree  

<br>

##  Results
- Accuracy: ~0.81 (depends on train/test split and preprocessing)  
- Sample predictions included  
- Tree visualization provided  

<br>

##  Tech Stack
- Python  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib  

<br>

✍️ *This project demonstrates classification, preprocessing, and model visualization on a classic Kaggle dataset.*
