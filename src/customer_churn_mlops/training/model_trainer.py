
from sklearn.base import clone
import time

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from skopt import BayesSearchCV
from skopt.space import Real, Integer

from customer_churn_mlops.utils.mlflow_configs import display_and_log_metrics


def print_best_score_params(model):
    """
    Prints the best score and hyperparameters of a fitted model.

    Parameters
    ----------
    model : object
        The model object, which must have `best_score_` and `best_params_` attributes.
    """
    print("Best Score: ", model.best_score_)
    print("Best Hyperparameters: ", model.best_params_)


def optimize_and_log_model(model, search_space, model_name, X_train, y_train,
                           X_test, y_test, developer, scaler_path, features_path,
                           n_iter=25, cv=5, scoring=["f1_weighted", "roc_auc", "recall"]):
    """
    Perform hyperparameter optimization and log the best model and metrics.

    Parameters
    ----------
    model : sklearn.base.BaseEstimator
        The model to optimize.
    search_space : dict
        The hyperparameter search space.
    model_name : str
        The name of the model.
    X_train : pandas.DataFrame
        The training data.
    y_train : pandas.Series
        The training labels.
    X_test : pandas.DataFrame
        The test data.
    y_test : pandas.Series
        The test labels.
    developer : str
        The name of the developer.
    scaler_path : str
        The path to the scaler.
    features_path : str
        The path to the features.
    n_iter : int
        The number of iterations to run the optimization.
    cv : int
        The number of folds for the cross-validation.
    scoring : list of str
        The metrics to use for optimization.

    Returns
    -------
    best_model : sklearn.base.BaseEstimator
        The best model.
    search : skopt.BayesSearchCV
        The optimization object.
    """
    search = BayesSearchCV(
        estimator=clone(model),
        search_spaces=search_space,
        scoring=scoring,
        n_iter=n_iter,
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    search.fit(X_train, y_train)

    print_best_score_params(search)

    best_model = search.best_estimator_


    start_time = time.time()
    best_model.fit(X_train, y_train)
    end_time = time.time()

    trainging_time = end_time - start_time

    run_id = display_and_log_metrics(
        best_model,
        scaler_path,
        model_name,
        developer,
        X_train, y_train,
        X_test, y_test,
        trainging_time, features_path
    )

    return best_model, run_id


def train_model(X_train_filtered, y_train, X_test_filtered, y_test, final_scaler_path, best_features_path,
                developer="Cristian"):
    lr = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('clf', LogisticRegression())
    ])

    lr_search_space = {
        "clf__C": Real(1e-4, 1e3, prior="log-uniform"),
        "clf__max_iter": Integer(100, 1000), 
        "clf__tol": Real(1e-5, 1e-1, prior="log-uniform")
    }

    best_lrr, run_id = optimize_and_log_model(
        lr,
        lr_search_space,
        "LogisticRegression",
        X_train_filtered, y_train,
        X_test_filtered, y_test,
        developer,
        final_scaler_path,
        best_features_path,
        n_iter=25,
        cv=5,
        scoring="recall"
    )

    return best_lrr, run_id
