import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from utils import (
    load_data,
    fill_missing_data,
    encode_categorical_data,
    create_features,
    scale_data
)
import pickle
import numpy as np

train_set = load_data("data/train.csv")

float_columns = ["Age", "Annual Income", "Number of Dependents", "Health Score",
                 "Previous Claims", "Vehicle Age", "Credit Score", "Insurance Duration"]
categorical_columns = ["Gender", "Marital Status", "Education Level", "Occupation", "Location",
                       "Policy Type", "Customer Feedback", "Smoking Status", "Exercise Frequency", "Property Type"]

train_set = fill_missing_data(train_set, float_columns, categorical_columns)
train_set = encode_categorical_data(train_set, categorical_columns)
train_set = create_features(train_set)

X = train_set.drop(columns=["Premium Amount", "Policy Start Date"])
y = train_set["Premium Amount"]

X, scaler = scale_data(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# RMSLE hesabı
y_pred = model.predict(X_test)
rmsle = np.sqrt(((np.log1p(y_test) - np.log1p(y_pred)) ** 2).mean())

print("RMSLE:", rmsle)

with open("model.pkl", "wb") as file:
    pickle.dump((model, scaler), file)