import pickle
import json
import mlflow
from mlflow.tracking import MlflowClient

def load_model(model_name, model_version):
    model_uri = f"models:/{model_name}/{model_version.version}"
    return mlflow.pyfunc.load_model(model_uri)

def load_preprocessor(client: MlflowClient, run_id: str, path='preprocessor/preprocessor.pkl'):
    scaler_path = client.download_artifacts(run_id=run_id, path=path)
    with open(scaler_path, 'rb') as f:
        return pickle.load(f)

def load_features(client: MlflowClient, run_id: str, path='features/features.json'):
    local_path = client.download_artifacts(run_id=run_id, path=path)
    with open(local_path, 'r') as f:
        return json.load(f)
