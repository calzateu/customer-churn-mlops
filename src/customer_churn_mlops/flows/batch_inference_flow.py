
from prefect import task, flow
from prefect.tasks import NO_CACHE

import pandas as pd

import os
from mlflow.tracking import MlflowClient
from customer_churn_mlops.inference.model_loader import load_sklearn_model, load_preprocessor, load_features
from customer_churn_mlops.inference.batch_predictor import predict_and_save
from customer_churn_mlops.utils.data_processing import read_data, prepare_data
from customer_churn_mlops.utils.mlflow_configs import setup_mlflow

from sqlalchemy import create_engine
from customer_churn_mlops.utils.config import DB_URL

@task(name="setup-mlflow", retries=3, retry_delay_seconds=2)
def setup_mlflow_task(tracking_uri: str, experiment_name: str):
    return setup_mlflow(tracking_uri, experiment_name)

@task(name="get-model-version", retries=3, retry_delay_seconds=2)
def get_model_version_task(client: MlflowClient, model_name: str, stage: str):
    return client.get_model_version_by_alias(model_name, stage)

@task(name="load-model", retries=3, retry_delay_seconds=2, cache_policy=NO_CACHE)
def load_model_task(model_name, model_version):
    return load_sklearn_model(model_name, model_version)

@task(name="load-preprocessor", retries=3, retry_delay_seconds=2)
def load_preprocessor_task(client, run_id):
    return load_preprocessor(client, run_id)

@task(name="load-features", retries=3, retry_delay_seconds=2)
def load_features_task(client, run_id):
    return load_features(client, run_id)

@task(name="read-data", retries=3, retry_delay_seconds=2)
def read_data_task(path):
    return read_data(path)

@task(name="prepare-data")
def prepare_data_task(df, preprocessor, features, target="target"):
    return prepare_data(df, preprocessor, features, target)

@task(name="batch-inference", cache_policy=NO_CACHE)
def batch_inference_task(
    model, X, customer_ids, output_file
):
    return predict_and_save(model, X, customer_ids, output_file)

@task(name="insert-predictions-into-db")
def insert_predictions_into_db(pred_df: pd.DataFrame):
    engine = create_engine(DB_URL)
    subset = pred_df[["customer_id", "churn", "churn_probability"]].copy()
    subset["churn"] = subset["churn"].astype(bool)
    subset.to_sql("predictions_history", engine, if_exists="append", index=False)
    print(f"✅ Inserted {len(subset)} predictions into 'predictions_history'")


@flow(name="customer-churn-batch-inference", log_prints=True)
def batch_inference_flow(
    input_file: str = "data/test.csv",
    output_file: str = "output/predictions.csv",
    model_name: str = "customer-churn-model",
    stage: str = "production",
    experiment_name: str = "customer-churn-experiment",
):
    tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5050") # Not 5000 because it's already in use in my Mac
    os.makedirs("output", exist_ok=True)

    client = setup_mlflow_task(tracking_uri, experiment_name)
    version = get_model_version_task(client, model_name, stage)
    model = load_model_task(model_name, version)
    preprocessor = load_preprocessor_task(client, version.run_id)
    features = load_features_task(client, version.run_id)

    df = read_data_task(input_file)
    X, _, customer_ids = prepare_data_task(df, preprocessor, features)

    pred_df = batch_inference_task(model, X, customer_ids, output_file)
    insert_predictions_into_db(pred_df)

if __name__ == "__main__":
    batch_inference_flow()
