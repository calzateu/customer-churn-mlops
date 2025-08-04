from prefect import serve

from customer_churn_mlops.flows.batch_inference_flow import batch_inference_flow
from customer_churn_mlops.flows.model_training_flow import train_model_flow
from customer_churn_mlops.flows.kaggle_data_to_db_flow import load_churn_data_flow
from customer_churn_mlops.flows.monitoring_flow import monitoring_flow

if __name__ == "__main__":
    load_churn_data_deployment = load_churn_data_flow.to_deployment(
        name="Load Kaggle Data",
        tags=["data", "kaggle"],
    )

    monitoring_deployment = monitoring_flow.to_deployment(
        name="Monitoring",
        tags=["monitoring", "ml"],
        schedule={"cron": "0 0 * * *", "timezone": "America/Bogota"}, # every day at 00:00
    )

    train_model_deployment = train_model_flow.to_deployment(
        name="Model Training",
        tags=["training", "ml"],
    )

    batch_inference_deployment = batch_inference_flow.to_deployment(
        name="Batch Inference",
        tags=["batch", "ml"],
        schedule={"cron": "0 23 * * *", "timezone": "America/Bogota"}, # every day at 23:00
    )

    serve(
        load_churn_data_deployment,
        monitoring_deployment,
        batch_inference_deployment,
        train_model_deployment
    )
