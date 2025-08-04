import logging
import warnings
import os
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import text
#import kaggle
from prefect import task, flow

from customer_churn_mlops.utils.config import DB_URL
from customer_churn_mlops.utils.db_configs import camel_to_snake

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


@task(name="truncate-users-and-churn-data")
def truncate_tables(db_url: str):
    engine = create_engine(db_url)
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM users_churn_data"))
        conn.execute(text("DELETE FROM users"))
        conn.commit()
    logger.info("🧹 Tables 'users' and 'users_churn_data' truncated")

@task(name="download-from-kaggle")
def kaggle_to_local(dataset: str, download_path: str, filename: str, using_kaggle: bool) -> str:
    """Downloads dataset from Kaggle if not present"""
    os.makedirs(download_path, exist_ok=True)
    full_path = os.path.join(download_path, filename)

    files_exist = os.path.exists(full_path)

    if not files_exist:
        if not using_kaggle:
            raise ValueError("You are not using Kaggle, please set using_kaggle to True and use kaggle api key or download the dataset manually.")

        #kaggle.api.dataset_download_files(dataset, path=download_path, unzip=True)
        logger.info(f"✅ Dataset downloaded to: {download_path}")
    else:
        logger.info(f"📁 Dataset already exists at: {full_path}")

    return full_path

@task(name="load-csv")
def load_dataframe(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)

@task(name="insert-into-users")
def insert_into_users(df: pd.DataFrame, db_url: str):
    users_df = df[["CustomerID", "Churn"]].copy()
    users_df.columns = ["customer_id", "churn"]
    users_df["churn"] = users_df["churn"].astype(bool)

    engine = create_engine(db_url)
    users_df.to_sql("users", engine, if_exists="append", index=False)
    logger.info(f"✅ Inserted {len(users_df)} rows into 'users'")

@task(name="insert-into-users-churn-data")
def insert_into_users_churn_data(df: pd.DataFrame, db_url: str):
    features_df = df.drop(columns=["Churn"])
    features_df.columns = [camel_to_snake(c) for c in features_df.columns]

    yes_no_cols = ["paperless_billing", "multi_device_access", "parental_control", "subtitles_enabled"]
    for col in yes_no_cols:
        if col in features_df.columns:
            features_df[col] = features_df[col].map({"Yes": True, "No": False})

    engine = create_engine(db_url)
    features_df.to_sql("users_churn_data", engine, if_exists="append", index=False)
    logger.info(f"✅ Inserted {len(features_df)} rows into 'users_churn_data'")

# ---------- FLOW ----------

@flow(name="load-kaggle-customer-churn", log_prints=True)
def load_churn_data_flow(
    kaggle_dataset: str = "safrin03/predictive-analytics-for-customer-churn-dataset",
    kaggle_path: str = "data/",
    csv_filename: str = "train.csv",
    using_kaggle: bool = False,
    reset_db: bool = False,
):
    if reset_db:
        truncate_tables(DB_URL)

    path = kaggle_to_local(dataset=kaggle_dataset, download_path=kaggle_path, filename=csv_filename,
                           using_kaggle=using_kaggle)
    df = load_dataframe(path)
    insert_into_users(df, DB_URL)
    insert_into_users_churn_data(df, DB_URL)

if __name__ == "__main__":
    load_churn_data_flow()
