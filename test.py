import pandas as pd
from utils import (
    load_data,
    fill_missing_data,
    encode_categorical_data,
    create_features,
    scale_data,
    load_model
)

test_set = load_data("data/test.csv")

float_columns = ["Age", "Annual Income", "Number of Dependents", "Health Score",
                 "Previous Claims", "Vehicle Age", "Credit Score", "Insurance Duration"]
categorical_columns = ["Gender", "Marital Status", "Education Level", "Occupation", "Location",
                       "Policy Type", "Customer Feedback", "Smoking Status", "Exercise Frequency", "Property Type"]

test_set = fill_missing_data(test_set, float_columns, categorical_columns)
test_set = encode_categorical_data(test_set, categorical_columns)
test_set = create_features(test_set, is_train=False)

X_test = test_set.drop(columns=["Policy Start Date"])

model, scaler = load_model("model.pkl")

X_test_scaled, _ = scale_data(X_test, scaler)

y_pred = model.predict(X_test_scaled)

# Sonuç kaydetme
output = pd.DataFrame({"ID": test_set["id"], "Predicted Premium Amount": y_pred})
output.to_csv("predictions.csv", index=False)

print("Tahminler 'predictions.csv' dosyasına kaydedildi.")