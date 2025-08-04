import pandas as pd
import os
import json
from sqlalchemy import create_engine
from prefect import flow, task
from evidently import Dataset, DataDefinition, Report
from evidently.metrics import (
    Accuracy, F1Score, Precision, Recall,
    DatasetMissingValueCount, EmptyColumnsCount,
    DuplicatedColumnsCount, AlmostDuplicatedColumnsCount, AlmostConstantColumnsCount,
    ColumnCount, RowCount, DuplicatedRowCount, EmptyRowsCount
)
from evidently import BinaryClassification
from evidently.presets import DataDriftPreset

from customer_churn_mlops.utils.config import DB_URL
from customer_churn_mlops.utils.db_configs import camel_to_snake


@task
def load_reference(path="data/train_reference.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [camel_to_snake(c) for c in df.columns]


    df.drop(columns=["customer_id"], inplace=True, errors="ignore")
    return df


@task
def load_evaluation_dataset(db_url: str = DB_URL) -> pd.DataFrame:
    query = """
    SELECT ucd.*, 
           u.churn AS actual,
           p.churn AS prediction,
           p.churn_probability
    FROM users_churn_data ucd
    JOIN users u ON ucd.customer_id = u.customer_id
    JOIN (
        SELECT DISTINCT ON (customer_id) *
        FROM predictions_history
        ORDER BY customer_id, predicted_at DESC
    ) p ON p.customer_id = ucd.customer_id;
    """
    engine = create_engine(db_url)
    df = pd.read_sql(query, engine)

    df.drop(columns=['customer_id'], inplace=True)

    print("Loaded df")
    print(df.head())

    return df


@task
def build_data_definition(df: pd.DataFrame) -> DataDefinition:
    numerical = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical = df.select_dtypes(include=["object"]).columns.tolist()
    boolean = df.select_dtypes(include=["bool"]).columns.tolist()

    return DataDefinition(
        numerical_columns=numerical,
        categorical_columns=categorical + boolean,
        classification=[
            BinaryClassification(
                target="actual",
                prediction_labels="prediction",
                pos_label=True
            )
        ]
    )


@task
def generate_reports(reference_df: pd.DataFrame, current_df: pd.DataFrame, data_def: DataDefinition):
    # Evidently Datasets
    ref_dataset = Dataset.from_pandas(reference_df, data_definition=data_def)
    curr_dataset = Dataset.from_pandas(current_df, data_definition=data_def)

    # Drift report
    drift_report = Report([
        ColumnCount(),
        RowCount(),
        DataDriftPreset()
    ])
    drift_eval = drift_report.run(curr_dataset, ref_dataset)

    # Performance report
    perf_report = Report([
        Accuracy(),
        Precision(),
        Recall(),
        F1Score(),
    ])
    perf_eval = perf_report.run(curr_dataset, ref_dataset)

    # Data Quality report
    quality_report = Report([
        ColumnCount(),
        RowCount(),
        EmptyRowsCount(),
        EmptyColumnsCount(),
        DuplicatedRowCount(),
        DuplicatedColumnsCount(),
        DatasetMissingValueCount(), 
        AlmostConstantColumnsCount(),
        AlmostDuplicatedColumnsCount()
    ])
    quality_eval = quality_report.run(curr_dataset)

    return drift_eval, perf_eval, quality_eval, drift_report, perf_report, quality_report


@task
def save_reports(reports: list, path: str = "reports/"):
    os.makedirs(path, exist_ok=True)
    names = ["drift", "performance", "quality"]
    for name, report in zip(names, reports):
        report.save_json(os.path.join(path, f"{name}.json"))
        report.save_html(os.path.join(path, f"{name}.html"))
    print(f"✅ Saved reports to {path}")


@task(name="save-reports-to-db")
def save_reports_to_db(reports: list, db_url: str = DB_URL):
    engine = create_engine(db_url)
    names = ["drift", "performance", "quality"]
    rows = []

    for name, report in zip(names, reports):
        report_dict = report.dict()
        rows.append({
            "report_type": name,
            "report_json": json.dumps(report_dict),
            "generated_at": pd.Timestamp.utcnow()
        })

    df = pd.DataFrame(rows)
    df.to_sql("evidently_reports", engine, if_exists="append", index=False)
    print(f"✅ Inserted {len(df)} reports into 'evidently_reports'")


@flow(name="monitoring-with-evidently-v2", log_prints=True)
def monitoring_flow():
    reference = load_reference()
    current = load_evaluation_dataset()
    data_def = build_data_definition(current)

    drift_eval, performance_eval, quality_eval, drift_report, performance_report, quality_report = generate_reports(reference, current, data_def)

    save_reports([drift_eval, performance_eval, quality_eval])
    save_reports_to_db([drift_eval, performance_eval, quality_eval])


if __name__ == "__main__":
    monitoring_flow()
