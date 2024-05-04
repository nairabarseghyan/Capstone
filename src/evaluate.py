from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd

from config import DATA_PATH

from gluonts.evaluation import Evaluator

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def evaluate_model(y_true_train, y_pred_train, y_true_val, y_pred_val, model_name):
    # Calculate evaluation metrics
    mae_train = mean_absolute_error(y_true_train, y_pred_train)
    mae_val = mean_absolute_error(y_true_val, y_pred_val)
    
    mape_train = mean_absolute_percentage_error(y_true_train, y_pred_train)
    mape_val = mean_absolute_percentage_error(y_true_val, y_pred_val)
    
    rmse_train = np.sqrt(mean_squared_error(y_true_train, y_pred_train))
    rmse_val = np.sqrt(mean_squared_error(y_true_val, y_pred_val))
    
    r2_train = r2_score(y_true_train, y_pred_train)
    r2_val = r2_score(y_true_val, y_pred_val)
    
    
    metrics = {
        "model_name": model_name,
        "mae_train": mae_train,
        "mae_val": mae_val,
        "mape_train": mape_train,
        "mape_val": mape_val,
        "rmse_train": rmse_train,
        "rmse_val": rmse_val,
        "r2_train": r2_train,
        "r2_val": r2_val
    }
    
    # Convert dictionary to DataFrame
    metrics_df = pd.DataFrame([metrics])
    
    metrics_df.to_csv(f'{DATA_PATH}results/Models/{model_name}.csv',index = False)
    
    print(f"Results of model {model_name}: ")
    # Print evaluation metrics
    print("MAE (Mean Absolute Error):")
    print(f"Train Set: {mae_train}")
    print(f"Validation Set: {mae_val}\n")
    
    print("MAPE (Mean Absolute Percentage Error):")
    print(f"Train Set: {mape_train}%")
    print(f"Validation Set: {mape_val}%\n")
    
    print("RMSE (Root Mean Squared Error):")
    print(f"Train Set: {rmse_train}")
    print(f"Validation Set: {rmse_val}\n")
    
    print("R-squared (R2) Score:")
    print(f"Train Set: {r2_train}")
    print(f"Validation Set: {r2_val}\n")
    
    
    
def evaluate_lag_llama_model(tss_train, forecasts_train, tss_val, forecasts_val,model_name):
    evaluator = Evaluator()

    agg_metrics_train, ts_metrics_train = evaluator(iter(tss_train), iter(forecasts_train))
    agg_metrics_val, ts_metrics_val = evaluator(iter(tss_val), iter(forecasts_val))
    
    mae_train = mean_absolute_error(tss_train[0][0], forecasts_train[0].quantile(0.5))
    mae_val = mean_absolute_error(tss_val[0][0], forecasts_val[0].quantile(0.5))
    
    mae_train = agg_metrics_train
    mae_val = agg_metrics_val
    
    mape_train = agg_metrics_train
    mape_val = agg_metrics_val
    
    rmse_train = agg_metrics_train
    rmse_val = agg_metrics_val
    
    r2_train = r2_score(tss_train[0][0], forecasts_train[0].quantile(0.5))
    r2_val = r2_score(tss_val[0][0], forecasts_val[0].quantile(0.5))
    

    
    metrics = {
        "model_name": model_name,
        "mae_train": mae_train,
        "mae_val": mae_val,
        "mape_train": mape_train,
        "mape_val": mape_val,
        "rmse_train": rmse_train,
        "rmse_val": rmse_val,
        "r2_train": r2_train,
        "r2_val": r2_val
    }
    
    # Convert dictionary to DataFrame
    metrics_df = pd.DataFrame([metrics])
    
    metrics_df.to_csv(f'{DATA_PATH}results/Models/{model_name}.csv',index = False)
    
    print(f"Results of model {model_name}: ")
    # Print evaluation metrics
    print("MAE (Mean Absolute Error):")
    print(f"Train Set: {mae_train}")
    print(f"Validation Set: {mae_val}\n")
    
    print("MAPE (Mean Absolute Percentage Error):")
    print(f"Train Set: {mape_train}%")
    print(f"Validation Set: {mape_val}%\n")
    
    print("RMSE (Root Mean Squared Error):")
    print(f"Train Set: {rmse_train}")
    print(f"Validation Set: {rmse_val}\n")
    
    print("R-squared (R2) Score:")
    print(f"Train Set: {r2_train}")
    print(f"Validation Set: {r2_val}\n")