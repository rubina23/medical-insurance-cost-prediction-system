# **Medical Insurance Cost Prediction System**

# Steps:

##**1. Data Loading**

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score


# Dataset load
df = pd.read_csv("insurance.csv")

# Display first few rows and shape
print(df.head())
print("Dataset shape:", df.shape)

## **2. Data Preprocessing**

# 1. Check Missing Values
print(df.isnull().sum())

# 2. Split Features and Target
X = df.drop('charges', axis=1)
y = df['charges']

# 3. OneHot Encoding for Categorical Variables
categorical_cols = ['sex', 'smoker', 'region']
numeric_cols  = ['age', 'bmi', 'children']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ])

# 4. Outlier Handling (optional, for charges)
Q1 = df['charges'].quantile(0.25)
Q3 = df['charges'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['charges'] >= lower_bound) & (df['charges'] <= upper_bound)]
X = df.drop('charges', axis=1)
y = df['charges']
# X, y = df.drop('charges', axis=1), df['charges']

# 5. Feature Engineering (BMI Category)
df["bmi_category"] = pd.cut(df["bmi"], bins=[0,18.5,25,30,100],
                            labels=["Underweight","Normal","Overweight","Obese"])
X["bmi_category"] = df["bmi_category"]
categorical_cols.append("bmi_category")

"""## **3. Pipeline Creation (10 Marks)**
Construct a standard Machine Learning pipeline that integrates preprocessing and the model
"""

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42))
])

## **5. Model Training**

# Train the pipeline on the training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

## **6. Cross-Validation (10 Marks)**

# Apply 5-fold cross-validation on the training set
scores = cross_val_score(pipeline, X, y, cv=5, scoring="r2")
print("CV Mean:", scores.mean())
print("CV Std:", scores.std())

## **7. Hyperparameter Tuning**

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5]
}

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring="r2", n_jobs=-1)
grid.fit(X_train, y_train)

print("Best Params:", grid.best_params_)
print("Best Score:", grid.best_score_)

## **8. Best Model Selection**

# Select the best model from GridSearchCV
best_model = grid.best_estimator_

# Display the chosen configuration
print("Final Best Model:", best_model)

## **9. Model Performance Evaluation**

# Predict on the test set
y_pred = best_model.predict(X_test)

# Print evaluation metrics
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
# print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))

## **Save Model**

import pickle
with open("insurance_model.pkl", "wb") as f:
    pickle.dump(grid.best_estimator_, f)


"""**See rest of task on app.py file**"""
