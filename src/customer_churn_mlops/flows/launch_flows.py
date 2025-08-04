from prefect import serve

from customer_churn_mlops.flows.batch_inference_flow import batch_inference_flow
from customer_churn_mlops.flows.model_training_flow import train_model_flow

if __name__ == "__main__":
    batch_inference_deployment = batch_inference_flow.to_deployment(
        name="Batch Inference",
        tags=["batch", "ml"],
        schedule={"cron": "0 23 * * *", "timezone": "America/Bogota"}, # every day at 23:00
    )

    train_model_deployment = train_model_flow.to_deployment(
        name="Model Training",
        tags=["training", "ml"],
    )

    serve(
        batch_inference_deployment,
        train_model_deployment
    )
