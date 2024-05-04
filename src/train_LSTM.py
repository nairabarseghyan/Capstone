# import modules 
import pickle
import os
import numpy as np
from fastprogress import progress_bar
import torch
import time

from data_prep import load_and_preprocess_data, general_revenue_dataframe, split_univariate_data, arrays_to_tensors
from features import scaler
from models import sliding_windows, LSTM, LSTM2, init_weights
from config import MODEL_PATH, SEED, seed_everything, find_device

os.environ['KMP_DUPLICATE_LIB_OK']='True'
seed_everything(SEED)

# HYPERPARMAETERS
seq_length = 28 #length of the sliding window
device  = find_device()

#For First NN
num_epochs1 = 500
learning_rate1 = 1e-3
input_size1 = 1
hidden_size1 = 512
num_layers1 = 1
num_classes1 = 1

#For Second NN
num_epochs2 = 700
learning_rate2 = 1e-3
input_size2 = 1
hidden_size2 = 512
num_layers2 = 2

num_classes2 = 1

#loading data and adding necessary features
initial_data = load_and_preprocess_data()
univariate_data = general_revenue_dataframe(initial_data)
univariate_data_array = np.array(univariate_data)


#spliting data
train_data, test_data = split_univariate_data(univariate_data_array)


# Scaling (Normalizing) data
train_data_normalized, test_data_normalized, full_data_normalized, fitted_scaler = scaler(train_data, test_data, univariate_data_array)

#Applying sliding window tp the normlized data 
trainX, trainY = sliding_windows(train_data_normalized, seq_length)
testX, testY = sliding_windows(test_data_normalized, seq_length)
fullX, fullY = sliding_windows(full_data_normalized, seq_length)

#getting tensors of data

trainX_tensor, trainY_tensor, testX_tensor, testY_tensor, dataX, dataY = arrays_to_tensors(trainX, trainY, testX, testY, fullX, fullY)


# Check shapes of tensors 

print("train shape is:",trainX_tensor.size())
print("train label shape is:",trainY_tensor.size())
print("test shape is:",testX_tensor.size())
print("test label shape is:",testY_tensor.size())


# =====================================================================================================================================
#                                 Instantiate the first LSTM model and move it to the appropriate device (GPU or CPU)
# =====================================================================================================================================

start_time_lstm1 = time.time()
lstm = LSTM(num_classes1, input_size1, hidden_size1, num_layers1)
lstm.to(device)

##### Set Criterion Optimzer and scheduler ####################
criterion = torch.nn.MSELoss().to(device)    # mean-squared error for regression
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate1,weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,  patience=500,factor =0.5 ,min_lr=1e-7, eps=1e-08)

#optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)

# Train the model

for epoch in progress_bar(range(num_epochs1)): 
    lstm.train()
    outputs = lstm(trainX_tensor.to(device))
    optimizer.zero_grad()
    
    # obtain the loss function
    loss = criterion(outputs, trainY_tensor.to(device))
    
    loss.backward()
    
    
    optimizer.step()
    
    #Evaluate on test     
    lstm.eval()
    valid = lstm(testX_tensor.to(device))
    vall_loss = criterion(valid, testY_tensor.to(device))
    scheduler.step(vall_loss)
    
    if epoch % 50 == 0:
      print("Epoch: %d, loss: %1.5f valid loss:  %1.5f " %(epoch, loss.cpu().item(),vall_loss.cpu().item()))
      
lstm_model_filename = os.path.join(MODEL_PATH, '1layer_lstm_model.pth')

torch.save({
            'model_state_dict': lstm.state_dict(),
            'num_classes': num_classes1,
            'input_size': input_size1,
            'hidden_size': hidden_size1,
            'num_layers': num_layers1
           }, lstm_model_filename)
      
print("First LSTM model training process is completed in --- %s seconds ---" % (time.time() - start_time_lstm1))


# =====================================================================================================================================
#                                 Instantiate the Second LSTM model with multiple LSTM layers
# =====================================================================================================================================

start_time_lstm2 = time.time()
multiple_lstm = LSTM2(num_classes2, input_size2, hidden_size2, num_layers2)
multiple_lstm.to(device)


multiple_lstm.apply(init_weights)

criterion = torch.nn.MSELoss().to(device)    # mean-squared error for regression
optimizer = torch.optim.Adam(multiple_lstm.parameters(), lr=learning_rate2,weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,  patience=100, factor =0.5 ,min_lr=1e-7, eps=1e-08)
#optimizer = torch.optim.SGD(multiple_lstm.parameters(), lr=learning_rate)

# Train the model

for epoch in progress_bar(range(num_epochs2)): 
    multiple_lstm.train()
    outputs = multiple_lstm(trainX_tensor.to(device))
    optimizer.zero_grad()
    torch.nn.utils.clip_grad_norm_(multiple_lstm.parameters(), 1)
    # obtain the loss function
    loss = criterion(outputs, trainY_tensor.to(device))
    
    loss.backward()
    
    scheduler.step(loss)
    optimizer.step()
    multiple_lstm.eval()
    valid = multiple_lstm(testX_tensor.to(device))
    vall_loss = criterion(valid, testY_tensor.to(device))
    scheduler.step(vall_loss)
    
    if epoch % 50 == 0:
      print("Epoch: %d, loss: %1.5f valid loss:  %1.5f " %(epoch, loss.cpu().item(),vall_loss.cpu().item()))
      
      
lstm2_model_filename = os.path.join(MODEL_PATH, 'multi_layer_lstm_model.pth')

torch.save({
            'model_state_dict': multiple_lstm.state_dict(),
            'num_classes': num_classes2,
            'input_size': input_size2,
            'hidden_size': hidden_size2,
            'num_layers': num_layers2
           }, lstm2_model_filename)

print("Seconf LSTM model training process is completed in --- %s seconds ---" % (time.time() - start_time_lstm2))


scaler_filename = os.path.join(MODEL_PATH, 'lstm_standard_scaler.pkl')

with open(scaler_filename, 'wb') as file:
    pickle.dump(fitted_scaler, file)

print("The traing process of LSTM models is completed.")