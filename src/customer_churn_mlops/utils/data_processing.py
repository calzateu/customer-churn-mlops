import pandas as pd
import pickle
import json
import os


def read_data(path):
    return pd.read_csv(path)

def prepare_data(df, preprocessor, features, target="Churn"):
    target_col = None
    if target in df.columns:
        target_col = df[target]
        df = df.drop(target, axis=1)

    # Transform the Yes/No columns into booleans
    yes_no_columns = ['PaperlessBilling', 'ContentType', 'MultiDeviceAccess', 'DeviceRegistered']
    for col in yes_no_columns:
        if df[col].nunique() == 2:
            df[col] = df[col].map({'Yes': True, 'No': False})

    customer_ids = df['CustomerID']

    x_scaled = preprocessor.transform(df)
    feature_names = preprocessor.get_feature_names_out()
    df_scaled = pd.DataFrame(x_scaled, columns=feature_names)

    return df_scaled[features], target_col, customer_ids


def save_pickle(model, filename, path="models", extension=".pkl") -> None:
    # Create the directory if it doesn't exist
    os.makedirs(path, exist_ok=True)

    filepath = os.path.join(path, f"{filename}{extension}")
    with open(filepath, 'wb') as file:
        pickle.dump(model, file)

    return filepath


def save_json(data, filename, path="models", extension=".json"):
    # Create the directory if it doesn't exist
    os.makedirs(path, exist_ok=True)

    filepath = os.path.join(path, f"{filename}{extension}")
    with open(filepath, 'w') as file:
        json.dump(data, file)

    return filepath
