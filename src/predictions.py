import pandas as pd
import numpy as np
import torch


from config import lag_llama_environment, find_device



# =====================================================================================================================================
#                                                   Machine Learning Models
# =====================================================================================================================================



def retrieve_predictions(split_df, model):
    df_copy = split_df.copy()
    X = df_copy.drop(columns=['Revenue','day_name','Date','Branch'])
    ref = df_copy[['Date', 'Branch']]
    y_pred = model.predict(X)
    y_pred = pd.DataFrame({
    'Pred': y_pred,
    'Date': ref['Date'].values,  # Use .values to ignore index alignment issues
    'Branch': ref['Branch'].values
})
    
    return y_pred



# =====================================================================================================================================
#                                                   LSTM NN Models
# =====================================================================================================================================

def lstm_model_predictions(model, scaler, device, X, Y):

    #model.eval()
    X_predict = model(X.to(device))
    X_predict = X_predict.cpu().data.numpy()
    Y = Y.cpu().data.numpy()

    ## Inverse Normalize 
    X_predict_actual = scaler.inverse_transform(X_predict)
    Y_actual = scaler.inverse_transform(Y)

    
    
    return X_predict_actual, Y_actual



# =====================================================================================================================================
#                                                   Transformer Lag-Llama Model
# =====================================================================================================================================



from gluonts.evaluation import make_evaluation_predictions

llama_device = torch.device(find_device())



def get_lag_llama_predictions(
    dataset, prediction_length, LagLlamaEstimator, num_samples=100, predictor=None, 
):
    ckpt = torch.load(
        "./lag-llama/lag-llama.ckpt", map_location=llama_device
    )
    estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

    estimator = LagLlamaEstimator(
        ckpt_path="./lag-llama/lag-llama.ckpt",
        prediction_length=prediction_length,

        # pretrained length
        context_length=32,

        # estimator args
        input_size=estimator_args["input_size"],
        n_layer=estimator_args["n_layer"],
        n_embd_per_head=estimator_args["n_embd_per_head"],
        n_head=estimator_args["n_head"],
        scaling=estimator_args["scaling"],
        time_feat=estimator_args["time_feat"],
        batch_size=1,
        num_parallel_samples=100,
        device = llama_device,
    )

    lightning_module = estimator.create_lightning_module()
    transformation = estimator.create_transformation()
    predictor = (
        predictor
        if predictor is not None
        else estimator.create_predictor(transformation, lightning_module)
    )

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset, predictor=predictor, num_samples=num_samples
    )
    forecasts = list(forecast_it)
    tss = list(ts_it)

    return forecasts, tss


def transfrom_lag_llama_predictions(forecasts, tss, prediction_length):
    all_preds = list()
    for item in forecasts:
        family = item.item_id
        p = np.median(item.samples, axis=0)
        p10 = np.percentile(item.samples, 10, axis=0)
        p90 = np.percentile(item.samples, 90, axis=0)
        dates = pd.date_range(start=item.start_date.to_timestamp(), periods=len(p), freq='D')
        family_pred = pd.DataFrame({'date': dates, 'pred': p, 'p10': p10, 'p90': p90})
        all_preds += [family_pred]
    all_preds = pd.concat(all_preds, ignore_index=True)
    
    valid = tss[0][-prediction_length:]
    valid.index = pd.to_datetime(valid.index)
    
    all_preds.merge(valid, left_on=['date'], right_index=True, how='left')
    
    return all_preds