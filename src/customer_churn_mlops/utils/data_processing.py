import pandas as pd
import pickle
import json
import os


def read_data(path):
    return pd.read_csv(path)

def prepare_data(df, scaler, features, target="Churn"):
    target_col = None
    if target in df.columns:
        target_col = df[target]
        df = df.drop(target, axis=1)
    df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns)
    return df_scaled[features], target_col


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
