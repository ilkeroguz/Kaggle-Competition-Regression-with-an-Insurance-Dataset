import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

def load_data(filepath):
    return pd.read_csv(filepath)

def fill_missing_data(data, float_columns, categorical_columns):
    data[float_columns] = data[float_columns].apply(lambda col: col.fillna(col.mean()))
    data[categorical_columns] = data[categorical_columns].fillna(method="pad")
    return data

def encode_categorical_data(data, categorical_columns):
    encoder = LabelEncoder()
    for column in categorical_columns:
        if data[column].dtype == "object":
            data[column] = encoder.fit_transform(data[column].astype(str))
    return data

def create_features(df, is_train=True):
    # Future engineering
    df["Vehicle_Age_Squared"] = df["Vehicle Age"] ** 2
    if is_train:
        df["Insurance_Per_Year"] = df["Premium Amount"] / df["Insurance Duration"]
    else:
        df["Insurance_Per_Year"] = 0
    return df

def scale_data(X, scaler=None):
 # Skalerizasyon
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    return X_scaled, scaler

def load_model(filepath):
    with open(filepath, "rb") as file:
        model, scaler = pickle.load(file)
    return model, scaler