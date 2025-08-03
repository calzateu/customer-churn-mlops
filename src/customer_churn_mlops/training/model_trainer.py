
from sklearn.base import clone
import time

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from skopt import BayesSearchCV
from skopt.space import Real, Integer

from customer_churn_mlops.utils.mlflow_configs import display_and_log_metrics


def print_best_score_params(model):
    print("Best Score: ", model.best_score_)
    print("Best Hyperparameters: ", model.best_params_)


def optimize_model(model, search_space, X_train, y_train,
                           n_iter=25, cv=5, scoring=["f1_weighted", "roc_auc", "recall"]):
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

    return best_model


def fit_final_model_without_smote(best_pipeline, X_train, y_train):
    # 1. Extract the best hyperparameters
    clf_params = {
        k.replace("clf__", ""): v
        for k, v in best_pipeline.get_params().items()
        if k.startswith("clf__")
    }

    # 2. Clone the best classifier and set the hyperparameters
    final_clf = clone(best_pipeline.named_steps["clf"])
    final_clf.set_params(**clf_params)

    # 3. Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # 4. Train the final model
    final_clf.fit(X_train_resampled, y_train_resampled)

    return final_clf


def train_model(X_train_filtered, y_train, X_test_filtered, y_test, final_scaler_path, best_features_path,
                developer="Cristian"):
    # 1. Training pipeline with SMOTE
    lr = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('clf', LogisticRegression())
    ])

    lr_search_space = {
        "clf__C": Real(1e-4, 1e3, prior="log-uniform"),
        "clf__max_iter": Integer(100, 1000), 
        "clf__tol": Real(1e-5, 1e-1, prior="log-uniform")
    }

    # 2. Optimize with SMOTE
    best_lrr = optimize_model(
        lr,
        lr_search_space,
        X_train_filtered, y_train,
        n_iter=25,
        cv=5,
        scoring="recall"
    )

    # 3. Train the final model
    final_model = fit_final_model_without_smote(best_lrr, X_train_filtered, y_train)

    # 4. Final model logging (without SMOTE)
    import time
    start_time = time.time()
    end_time = time.time()
    training_time = end_time - start_time

    run_id = display_and_log_metrics(
        final_model,
        final_scaler_path,
        "LogisticRegression",
        developer,
        X_train_filtered, y_train,
        X_test_filtered, y_test,
        training_time,
        best_features_path
    )

    return final_model, run_id
