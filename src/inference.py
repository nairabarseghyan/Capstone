import pickle
import numpy as np
import pandas as pd
import os
import time
import torch

from config import MODEL_PATH, DATA_PATH, find_device
from data_prep import load_and_preprocess_data, split_data, general_revenue_dataframe, split_univariate_data, arrays_to_tensors, get_lag_llama_dataset
from features import add_calendar_features, add_sales_proxies,  full_data_target_encoder
from predictions import retrieve_predictions, lstm_model_predictions, get_lag_llama_predictions, transfrom_lag_llama_predictions
from models import sliding_windows, LSTM, LSTM2
from evaluate import evaluate_model

start_time = time.time()

# =====================================================================================================================================
#                                                   Ensemble Models
# =====================================================================================================================================

#Importing trained model

filename = 'ensemble_model.sav'
model_path = os.path.join(MODEL_PATH, filename) 
with open(model_path, 'rb') as file:
    loaded_model = pickle.load(file)



# Importing encoder
encoder_name = 'ml_target_encoder.pkl'
encoder_path = os.path.join(MODEL_PATH, encoder_name)

with open(encoder_path, 'rb') as file:
    encoder = pickle.load(file)



# Importing data

initial_data = load_and_preprocess_data(encoding=True)
features_data_calendar = add_calendar_features(initial_data)
features_data_proxies = add_sales_proxies(features_data_calendar)

#spliting data
train_data, val_data, test_data = split_data(features_data_proxies)

#Encoding full data for predictions
train_data_encoded, val_data_encoded, test_data_encoded = full_data_target_encoder(train_data, val_data, test_data, encoder)



# Making Predictions
train_predictions = retrieve_predictions(train_data, loaded_model)
val_predictions = retrieve_predictions(val_data, loaded_model)
test_predictions = retrieve_predictions(test_data, loaded_model)


#Saving Predictions
train_predictions.to_csv(f"{DATA_PATH}results/Predictions/SR_train_predictions.csv")
val_predictions.to_csv(f"{DATA_PATH}results/Predictions/SR_val_predictions.csv")
test_predictions.to_csv(f"{DATA_PATH}results/Predictions/SR_test_predictions.csv")




# =====================================================================================================================================
#                                                       LSTM MODELS
# =====================================================================================================================================

seq_length = 28 #length of the sliding window
device  = find_device()

#Importing trained models

lstm_model_filename = os.path.join(MODEL_PATH, '1layer_lstm_model.pth')

loaded_model_info = torch.load(lstm_model_filename)
lstm = LSTM(loaded_model_info['num_classes'],
                   loaded_model_info['input_size'],
                   loaded_model_info['hidden_size'],
                   loaded_model_info['num_layers'])
lstm.load_state_dict(loaded_model_info['model_state_dict'])
lstm.to(device)                                         # Make sure to send the model to the appropriate device
lstm.eval()



lstm2_model_filename = os.path.join(MODEL_PATH, 'multi_layer_lstm_model.pth')
loaded_model_info2 = torch.load(lstm2_model_filename)
multiple_lstm = LSTM2(loaded_model_info2['num_classes'],
                   loaded_model_info2['input_size'],
                   loaded_model_info2['hidden_size'],
                   loaded_model_info2['num_layers'])
multiple_lstm.load_state_dict(loaded_model_info2['model_state_dict'])
multiple_lstm.to(device)                                         # Make sure to send the model to the appropriate device
multiple_lstm.eval()
    

# Importing scaler
scaler_filename = os.path.join(MODEL_PATH, 'lstm_standard_scaler.pkl')

with open(scaler_filename, 'rb') as file:
    scaler = pickle.load(file)
    

# Importing data

#loading data and adding necessary features
initial_data = load_and_preprocess_data()
univariate_data = general_revenue_dataframe(initial_data)
univariate_data_array = np.array(univariate_data)

#spliting data
train_data, test_data = split_univariate_data(univariate_data_array)


# Scaling (Normalizing) data
train_data_normalized, test_data_normalized, full_data_normalized = scaler.transform(train_data),scaler.transform(test_data),scaler.transform(univariate_data_array)

#Applying sliding window tp the normlized data 
trainX, trainY = sliding_windows(train_data_normalized, seq_length)
testX, testY = sliding_windows(test_data_normalized, seq_length)
fullX, fullY = sliding_windows(full_data_normalized, seq_length)

#getting tensors of data

trainX_tensor, trainY_tensor, testX_tensor, testY_tensor, dataX, dataY = arrays_to_tensors(trainX, trainY, testX, testY, fullX, fullY)

#get dates
date_index = pd.Series(univariate_data.index, name='date')
dates = date_index[0:1080]
dates_list = [date for date in dates]


### Getting the prediction and saving it

##One layer LSTM###

lstm1_prediction_X, lstm1_actual_Y = lstm_model_predictions(lstm, scaler, device, dataX, dataY)
prediction_df_lstm1 = pd.DataFrame({'Prediction': lstm1_prediction_X[:, 0], 'Actual': lstm1_actual_Y[:, 0],})
prediction_df_lstm1.index = pd.to_datetime(dates_list[:-29])

prediction_df_lstm1.to_csv(f"{DATA_PATH}results/Predictions/LSTM_one_predictions.csv")

##Two layer LSTM###
lstm_multi_prediction_X, lstm_multi_actual_Y = lstm_model_predictions(multiple_lstm, scaler, device, dataX, dataY)
prediction_df_lstm_multi = pd.DataFrame({'Prediction': lstm_multi_prediction_X[:, 0], 'Actual': lstm_multi_actual_Y[:, 0]})
prediction_df_lstm_multi.index = pd.to_datetime(dates_list[:-29])

prediction_df_lstm_multi.to_csv(f"{DATA_PATH}results/Predictions/LSTM_multi_predictions.csv")


###### Evaluating the models based on predictions.

### One layer LSTM evaluation ###
lstm1_X_predict_train, lstm1_Y_train= lstm_model_predictions(lstm, scaler, device, trainX_tensor, trainY_tensor)
lstm1_X_predict_test, lstm1_Y_test= lstm_model_predictions(lstm, scaler, device, testX_tensor, testY_tensor)

evaluate_model(lstm1_Y_train, lstm1_X_predict_train, lstm1_Y_test, lstm1_X_predict_test, "One_layer_LSTM")



### Multi layer LSTM evaluation ###
lstm_multi_X_predict_train, lstm_multi_Y_train = lstm_model_predictions(multiple_lstm, scaler, device, trainX_tensor, trainY_tensor)
lstm_multi_X_predict_test, lstm_multi_Y_test = lstm_model_predictions(multiple_lstm, scaler, device, testX_tensor, testY_tensor)

evaluate_model(lstm_multi_Y_train, lstm_multi_X_predict_train, lstm_multi_Y_test, lstm_multi_X_predict_test, "Multi_layer_LSTM")



print("Inference for all models was completed in --- %s seconds ---" % (time.time() - start_time))