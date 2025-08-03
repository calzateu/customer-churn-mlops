import pandas as pd


def predict_and_save(model, X, customer_ids, output_path):
    predictions = model.predict(X)
    pred_df = pd.DataFrame({
        "customerID": customer_ids,
        "Churn": predictions
    })
    pred_df.to_csv(output_path, index=False)
    return pred_df
