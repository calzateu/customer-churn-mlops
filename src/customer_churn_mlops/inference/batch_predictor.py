import pandas as pd



def predict_and_save(model, X, customer_ids, output_path):
    predictions = model.predict(X)
    probs = model.predict_proba(X)[:, 1]  # churn probability (positive class)

    pred_df = pd.DataFrame({
        "customer_id": customer_ids,
        "churn": predictions,
        "churn_probability": probs
    })

    pred_df.to_csv(output_path, index=False)
    return pred_df
