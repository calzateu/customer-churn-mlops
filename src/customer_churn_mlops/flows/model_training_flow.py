import os
import logging
import warnings

import pandas as pd
from prefect import task, flow
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import mlflow

from customer_churn_mlops.training.model_trainer import train_model
from customer_churn_mlops.utils.mlflow_configs import setup_mlflow
from customer_churn_mlops.utils.data_processing import read_data, save_json, save_pickle


logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


@task(name="setup-mlflow", retries=3, retry_delay_seconds=2)
def setup_mlflow_task(tracking_uri: str, experiment_name: str):
    return setup_mlflow(tracking_uri, experiment_name)


@task(name="read-data", retries=3, retry_delay_seconds=2)
def read_data_task(path):
    return read_data(path)


@task(name="preprocessing", retries=3, retry_delay_seconds=2)
def preprocessing_task(df, features, target="Churn"):
    # Transform the Yes/No columns into booleans
    yes_no_columns = ['PaperlessBilling', 'ContentType', 'MultiDeviceAccess', 'DeviceRegistered']
    for col in yes_no_columns:
        if df[col].nunique() == 2:
            df[col] = df[col].map({'Yes': True, 'No': False})

    # Remove the CustomerID column
    df = df.drop(columns=['CustomerID'])

    # Detect numeric, categorical and binary features
    numeric_features = df.drop(columns=[target]).select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.drop(columns=[target]).select_dtypes(include=['object']).columns.tolist()
    binary_features = df.drop(columns=[target]).select_dtypes(include=['bool']).columns.tolist()

    logger.info(f"Total Numeric features in the dataset {len(numeric_features)} : {numeric_features}")
    logger.info(f"Total Categorical features in the dataset {len(categorical_features)} : {categorical_features}")
    logger.info(f"Total Binary features in the dataset {len(binary_features)} : {binary_features}")
    logger.info(f"Target: {target}")

    # Split features and target
    X = df.drop(target, axis=1).copy()
    y = df[target].copy()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', MinMaxScaler()) # Normalize the data using MinMaxScaler since the columns are not normally distributed, so we need to preserve the distribution
            ]), numeric_features),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features),
            ('bin', 'passthrough', binary_features)
        ],
        remainder='drop'
    )
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    # Convert to DataFrames
    X_train_preprocessed_df = pd.DataFrame(X_train_preprocessed, columns=feature_names)
    X_test_preprocessed_df = pd.DataFrame(X_test_preprocessed, columns=feature_names)

    # Select desired features (column subset)
    X_train_filtered = X_train_preprocessed_df[features]
    X_test_filtered = X_test_preprocessed_df[features]

    return (
        X_train_filtered, y_train,
        X_test_filtered, y_test,
        preprocessor
    )


@task(name="save-features", retries=3, retry_delay_seconds=2)
def save_features_task(features, features_path):
    return save_json(features, "features", features_path, ".json")


@task(name="save-preprocessor", retries=3, retry_delay_seconds=2)
def save_preprocessor_task(preprocessor, folder_path):
    return save_pickle(preprocessor, "preprocessor", folder_path, ".pkl")


@task(name="register-model-in-mlflow", retries=3, retry_delay_seconds=2)
def register_model_in_mlflow_task(client, experiment_name, run_id, model_name, stage):
    experiments = client.search_experiments()

    experiment_id = -1
    for experiment in experiments:
        if experiment.name == experiment_name:
            experiment_id = experiment.experiment_id
            break
    
    if experiment_id == -1:
        raise Exception(f"Experiment {experiment_name} not found")
        
    model_uri = f"runs:/{run_id}/model"
    model_version = mlflow.register_model(
        model_uri,
        model_name
    )

    client.set_registered_model_alias(
        name=model_name,
        version=model_version.version,
        alias=stage
    )


@flow(name="customer-churn-model-training", log_prints=True)
def train_model_flow(
    data_path: str = "data/train.csv",
    model_name: str = "customer-churn-model",
    stage: str = "production",
    experiment_name: str = "customer-churn-experiment",
    features: list = [
        "num__AccountAge", "num__MonthlyCharges", "num__ViewingHoursPerWeek", "num__AverageViewingDuration", "num__ContentDownloadsPerMonth", "num__SupportTicketsPerMonth"
    ],
):
    tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5050") # Not 5000 because it's already in use in my Mac
    client = setup_mlflow_task(tracking_uri, experiment_name)

    df = read_data_task(data_path)
    features_path = save_features_task(features, "../models")

    X_train_filtered, y_train, X_test_filtered, y_test, preprocessor = preprocessing_task(df, features)
    preprocessor_path = save_preprocessor_task(preprocessor, "../models")

    _, run_id = train_model(
        X_train_filtered, y_train, X_test_filtered, y_test,
        preprocessor_path, features_path
    )

    register_model_in_mlflow_task(client, experiment_name, run_id, model_name, stage)


if __name__ == "__main__":
    train_model_flow()
