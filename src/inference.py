#IMPORTS 
## Libraries
import pickle
import numpy as np
import pandas as pd
import os
import time
import torch

# Import custom functionalities 
from config import MODEL_PATH, DATA_PATH, find_device
from data_prep import load_and_preprocess_data, split_data, general_revenue_dataframe, split_univariate_data, arrays_to_tensors, get_x_y
from features import add_calendar_features, add_sales_proxies,  full_data_target_encoder
from predictions import retrieve_predictions, lstm_model_predictions
from models import sliding_windows, LSTM, LSTM2
from evaluate import evaluate_model

start_time = time.time()

# =====================================================================================================================================
#                                                   ML MODELS: Final Model
# =====================================================================================================================================

# Loading the trained Random Forest Regressor model

filename_final = 'RFR_model.sav'
model_path = os.path.join(MODEL_PATH, filename_final) 
with open(model_path, 'rb') as file:
    loaded_model = pickle.load(file)



# Loading the encoder for data preprocessing
encoder_name = 'ml_target_encoder.pkl'
encoder_path = os.path.join(MODEL_PATH, encoder_name)

with open(encoder_path, 'rb') as file:
    encoder = pickle.load(file)



# Loading and preparing data
initial_data = load_and_preprocess_data(encoding=True)
features_data_calendar = add_calendar_features(initial_data)
features_data_proxies = add_sales_proxies(features_data_calendar)
train_data, val_data, test_data = split_data(features_data_proxies)


# Encoding data for model input
train_data_encoded, val_data_encoded, test_data_encoded = full_data_target_encoder(train_data, val_data, test_data, encoder)

# Making predictions with the final trained model
train_predictions = retrieve_predictions(train_data, loaded_model)
val_predictions = retrieve_predictions(val_data, loaded_model)
test_predictions = retrieve_predictions(test_data, loaded_model)

# Making Evaluations of saved model 
evaluate_model(train_predictions['True_value'], train_predictions['Pred'], val_predictions['True_value'], val_predictions['Pred'], "RandomForestRegressor")

# Saving model predictions to CSV files
train_predictions.to_csv(f"{DATA_PATH}results/Predictions/SR_train_predictions.csv")
val_predictions.to_csv(f"{DATA_PATH}results/Predictions/SR_val_predictions.csv")
test_predictions.to_csv(f"{DATA_PATH}results/Predictions/SR_test_predictions.csv")

# =====================================================================================================================================
#                                                   ML MODELS: Intermediary Models
# =====================================================================================================================================


################################### Ensemble Model ###################################

# Loading the trained Ensemble model

filename_ensemble = 'testing_models/Stack_model.sav'
model_path_ens = os.path.join(MODEL_PATH, filename_ensemble) 
with open(model_path_ens, 'rb') as file:
    loaded_model_ens = pickle.load(file)
    
# Making predictions with the final trained model
train_predictions_ens = retrieve_predictions(train_data, loaded_model_ens)
val_predictions_ens = retrieve_predictions(val_data, loaded_model_ens)

# Making Evaluations of saved model 
evaluate_model(train_predictions_ens['True_value'], train_predictions_ens['Pred'], val_predictions_ens['True_value'], val_predictions_ens['Pred'], "Stack")


################################### Linear Regression ###################################

# Loading the trained Linear Regression model
filename_LR = 'testing_models/LR_model.sav'
model_path_LR = os.path.join(MODEL_PATH, filename_LR) 
with open(model_path_LR, 'rb') as file:
    loaded_model_LR = pickle.load(file)

# Making predictions with the final trained model
train_predictions_LR = retrieve_predictions(train_data, loaded_model_LR)
val_predictions_LR = retrieve_predictions(val_data, loaded_model_LR)


# Making Evaluations of saved model 
evaluate_model(train_predictions_LR['True_value'], train_predictions_LR['Pred'], val_predictions_LR['True_value'], val_predictions_LR['Pred'], "LinearRegression")

################################### Decision Tree Regressor ###################################

# Loading the trained Decision Tree model
filename_DTR = 'testing_models/DTR_model.sav'
model_path_DT = os.path.join(MODEL_PATH, filename_DTR) 
with open(model_path_DT, 'rb') as file:
    loaded_model_DT = pickle.load(file)

# Making predictions with the final trained model
train_predictions_DT = retrieve_predictions(train_data, loaded_model_DT)
val_predictions_DT = retrieve_predictions(val_data, loaded_model_DT)


# Making Evaluations of saved model 
evaluate_model(train_predictions_DT['True_value'], train_predictions_DT['Pred'], val_predictions_DT['True_value'], val_predictions_DT['Pred'], "DecisionTree")

################################### Gradient Boosting Regressor ###################################

# Loading the trained Gradient Boosting model
filename_GBR = 'testing_models/GBR_model.sav'
model_path_GBR = os.path.join(MODEL_PATH, filename_GBR) 
with open(model_path_GBR, 'rb') as file:
    loaded_model_GBR = pickle.load(file)

# Making predictions with the final trained model
train_predictions_GBR = retrieve_predictions(train_data, loaded_model_GBR)
val_predictions_GBR = retrieve_predictions(val_data, loaded_model_GBR)


# Making Evaluations of saved model 
evaluate_model(train_predictions_GBR['True_value'], train_predictions_GBR['Pred'], val_predictions_GBR['True_value'], val_predictions_GBR['Pred'], "GradientBoosting")

################################### XGBoost ###################################

# Loading the trainedXGBoost model
filename_XGB = 'testing_models/XGB_model.sav'
model_path_XGB = os.path.join(MODEL_PATH, filename_XGB) 
with open(model_path_XGB, 'rb') as file:
    loaded_model_XGB = pickle.load(file)

# Making predictions with the final trained model
train_predictions_XGB = retrieve_predictions(train_data, loaded_model_XGB)
val_predictions_XGB = retrieve_predictions(val_data, loaded_model_XGB)


# Making Evaluations of saved model 
evaluate_model(train_predictions_XGB['True_value'], train_predictions_XGB['Pred'], val_predictions_XGB['True_value'], val_predictions_XGB['Pred'], "XGBoost")

# =====================================================================================================================================
#                                                       LSTM MODELS
# =====================================================================================================================================

seq_length = 28 #length of the sliding window
device  = find_device()

# Loading trained LSTM models
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
    


# Loading and preparing univariate data for LSTM models
initial_data = load_and_preprocess_data()
univariate_data = general_revenue_dataframe(initial_data)
univariate_data_array = np.array(univariate_data)

train_data, test_data = split_univariate_data(univariate_data_array)


# Normalizing data for LSTM processing
train_data_normalized, test_data_normalized, full_data_normalized = scaler.transform(train_data),scaler.transform(test_data),scaler.transform(univariate_data_array)

# Applying sliding window to normalized data
trainX, trainY = sliding_windows(train_data_normalized, seq_length)
testX, testY = sliding_windows(test_data_normalized, seq_length)
fullX, fullY = sliding_windows(full_data_normalized, seq_length)


# Converting arrays to tensors for LSTM input
trainX_tensor, trainY_tensor, testX_tensor, testY_tensor, dataX, dataY = arrays_to_tensors(trainX, trainY, testX, testY, fullX, fullY)

# Get dates
date_index = pd.Series(univariate_data.index, name='date')
dates = date_index[0:1080]
dates_list = [date for date in dates]


# Generating predictions with LSTM models and saving results

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


# ===================================== Evaluating LSTM models =====================================

### One layer LSTM evaluation ###
lstm1_X_predict_train, lstm1_Y_train= lstm_model_predictions(lstm, scaler, device, trainX_tensor, trainY_tensor)
lstm1_X_predict_test, lstm1_Y_test= lstm_model_predictions(lstm, scaler, device, testX_tensor, testY_tensor)

evaluate_model(lstm1_Y_train, lstm1_X_predict_train, lstm1_Y_test, lstm1_X_predict_test, "One_layer_LSTM")



### Multi layer LSTM evaluation ###
lstm_multi_X_predict_train, lstm_multi_Y_train = lstm_model_predictions(multiple_lstm, scaler, device, trainX_tensor, trainY_tensor)
lstm_multi_X_predict_test, lstm_multi_Y_test = lstm_model_predictions(multiple_lstm, scaler, device, testX_tensor, testY_tensor)

evaluate_model(lstm_multi_Y_train, lstm_multi_X_predict_train, lstm_multi_Y_test, lstm_multi_X_predict_test, "Multi_layer_LSTM")



print("Inference for all models was completed in --- %s seconds ---" % (time.time() - start_time))