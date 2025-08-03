import logging
import warnings

from sklearn.base import BaseEstimator
import mlflow
from mlflow.tracking import MlflowClient

from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, precision_recall_fscore_support,
    roc_auc_score
    )

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


def setup_mlflow(tracking_uri: str, experiment_name: str):
    """
    Set up MLflow tracking.

    Parameters
    ----------
    tracking_uri : str
        URI of the MLflow tracking server.
    experiment_name : str
        Name of the MLflow experiment.

    Returns
    -------
    client : MlflowClient
        A client for the MLflow tracking server.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return MlflowClient(tracking_uri)


def display_and_log_metrics(model, preprocessor_path, name_model, developer, X_train, y_train, X_test, y_test, training_time, features_path, use_cv=False):
    """
    Display and log metrics for the model in MLflow. Also, logs
    the preprocessor and selected features.

    Parameters
    ----------
    model : Estimator
        A scikit-learn estimator.
    preprocessor_path : str
        Path to the preprocessor.
    name_model : str
        Name of the model.
    developer : str
        Name of the developer.
    X_train : array-like of shape (n_samples, n_features)
        Training data.
    y_train : array-like of shape (n_samples,)
        Target values for the training data.
    X_test : array-like of shape (n_samples, n_features)
        Test data.
    y_test : array-like of shape (n_samples,)
        Target values for the test data.
    training_time : float
        Time taken for training the model.
    features_path : str
        Path to the selected features.
    use_cv : bool
        Whether to use cross-validation or not.

    """
    with mlflow.start_run() as run:
        mlflow.set_tag("model", name_model)
        mlflow.set_tag("developer", developer)

        mlflow.log_metric("training_time", training_time)

        y_train_pred_proba = model.predict_proba(
            X_train)[:, 1] # As the model is a binary classifier, we only need the positive class
        y_test_pred_proba = model.predict_proba(X_test)[:, 1]

        roc_auc_score_train = round(
            roc_auc_score(y_train, y_train_pred_proba, average="weighted"), 2
        )
        roc_auc_score_test = round(
            roc_auc_score(y_test, y_test_pred_proba, average="weighted"), 2
        )

        logger.info("ROC AUC Score Train: %s", roc_auc_score_train)
        logger.info("ROC AUC Score Test: %s", roc_auc_score_test)
        mlflow.log_metric("roc_auc_train", roc_auc_score_train)
        mlflow.log_metric("roc_auc_test", roc_auc_score_test)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        (
            precision_train,
            recall_train,
            fscore_train,
            support_train,
        ) = precision_recall_fscore_support(y_train, y_pred_train, average="weighted")
        (
            precision_test,
            recall_test,
            fscore_test,
            support_test,
        ) = precision_recall_fscore_support(y_test, y_pred_test, average="weighted")

        (
            positive_precision_test,
            positive_recall_test,
            positive_fscore_test,
            positive_support_test,
        ) = precision_recall_fscore_support(y_test, y_pred_test, average="binary", pos_label=1)
        (
            negative_precision_test,
            negative_recall_test,
            negative_fscore_test,
            negative_support_test,
        ) = precision_recall_fscore_support(y_test, y_pred_test, average="binary", pos_label=0)

        mlflow.log_metric("precision_train", precision_train)
        mlflow.log_metric("precision_test", precision_test)
        mlflow.log_metric("recall_train", recall_train)
        mlflow.log_metric("recall_test", recall_test)
        mlflow.log_metric("fscore_train", fscore_train)
        mlflow.log_metric("fscore_test", fscore_test)

        mlflow.log_metric("positive_precision_test", positive_precision_test)
        mlflow.log_metric("positive_recall_test", positive_recall_test)
        mlflow.log_metric("positive_fscore_test", positive_fscore_test)

        mlflow.log_metric("negative_precision_test", negative_precision_test)
        mlflow.log_metric("negative_recall_test", negative_recall_test)
        mlflow.log_metric("negative_fscore_test", negative_fscore_test)

        if support_train:
            mlflow.log_metric("support_train", support_train)
        if support_test:
            mlflow.log_metric("support_test", support_test)
        if positive_support_test:
            mlflow.log_metric("positive_support_test", positive_support_test)
        if negative_support_test:
            mlflow.log_metric("negative_support_test", negative_support_test)

        logger.info("Precision Train: %s", precision_train)
        logger.info("Precision Test: %s", precision_test)
        logger.info("Recall Train: %s", recall_train)
        logger.info("Recall Test: %s", recall_test)
        logger.info("Fscore Train: %s", fscore_train)
        logger.info("Fscore Test: %s", fscore_test)
        logger.info("Support Train: %s", support_train)
        logger.info("Support Test: %s", support_test)

        logger.info("Positive Precision Test: %s", positive_precision_test)
        logger.info("Positive Recall Test: %s", positive_recall_test)
        logger.info("Positive Fscore Test: %s", positive_fscore_test)
        logger.info("Positive Support Test: %s", positive_support_test)

        logger.info("Negative Precision Test: %s", negative_precision_test)
        logger.info("Negative Recall Test: %s", negative_recall_test)
        logger.info("Negative Fscore Test: %s", negative_fscore_test)
        logger.info("Negative Support Test: %s", negative_support_test)

        try:
            if use_cv:
                best_params = model.best_params_
            else:
                best_params = model.get_params()

            mlflow.log_params(best_params)

        except AttributeError as e:
            print(e)

        if isinstance(model, BaseEstimator):
            mlflow.sklearn.log_model(model, name="model",
                                     input_example=X_train.iloc[0].values.reshape(1, -1))
        else:
            raise ValueError("Unsupported model type for MLflow logging.")

        mlflow.log_artifact(preprocessor_path, artifact_path="preprocessor")

        # Log selected features
        mlflow.log_artifact(features_path, artifact_path="features")


        model_report_train = classification_report(y_train, y_pred_train)
        model_report_test = classification_report(y_test, y_pred_test)

        print("Classification Report for Train:\n", model_report_train)
        print("Classification Report for Test:\n", model_report_test)

        # Plot the confusion matrix
        _, ax = plt.subplots(figsize=(12, 8))

        cm = confusion_matrix(y_test, y_pred_test)
        cmp = ConfusionMatrixDisplay(
            cm, display_labels=["Not Churn", "Churn"] # TODO: Review labels
        )
        cmp.plot(ax=ax)

        plt.xticks(rotation=80)
        plt.show()

        return run.info.run_id
