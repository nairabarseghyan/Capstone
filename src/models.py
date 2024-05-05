# Importing libraries

#### ML libraries
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import StackingRegressor
import time


#### LSTM LIBRARIES 
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from itertools import cycle
from fastprogress import progress_bar

    

from evaluate import evaluate_model
from config import SEED, seed_everything, find_device

seed_everything(SEED)

device  = find_device()
    
# =====================================================================================================================================
#                                                   Machine Learning Models
# =====================================================================================================================================


# Benchmarking function for model testing
def ML_model_test(X_train, y_train, X_val, y_val):
    models = {
        "MLR": LinearRegression(),
        "DT": DecisionTreeRegressor(random_state=SEED),
        "RF": RandomForestRegressor(random_state=SEED, n_jobs=-1),
        "GB": GradientBoostingRegressor(random_state=SEED),
        "XGB": XGBRegressor(random_state=SEED)
    }
    
    for model_name, model in models.items():
    
        start = time.time()
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        
        evaluate_model(y_train, y_pred_train, y_val, y_pred_val, model_name)
        end = time.time()
        print(f"The computational time is {end - start} seconds")
        print("\n")
        
        
# Benchmarking
def ML_model_ensemble(X_train, y_train, X_val, y_val):
    models = {}
    models["RF"] = RandomForestRegressor(random_state=SEED, n_jobs=-1)
    models["XGB"] = XGBRegressor(random_state=SEED)

    estimators = [
        ('rf', RandomForestRegressor(random_state=SEED, n_jobs=-2)),
        ('xgb', XGBRegressor(random_state=SEED))
    ]
    stack_reg = StackingRegressor(
        estimators=estimators,
        final_estimator=LinearRegression()
    )
    models["stack"] = stack_reg

    return_models = []

    for model_name in models.keys():
    
        start = time.time()
        cur_model = models[model_name]
        cur_model.fit(X_train, y_train)
        y_pred_train = cur_model.predict(X_train)
        y_pred_val = cur_model.predict(X_val)
        
        evaluate_model(y_train, y_pred_train, y_val, y_pred_val, model_name)

        end = time.time()
        print(f"the computational time is {end-start} ")
        print("\n")
        return_models.append(cur_model)

    return return_models





# =====================================================================================================================================
#                                         Long Short-Term Memory Neural Network Model
# =====================================================================================================================================

###  This function creates a sliding window or sequences of 28 days and one day label ####
def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))

class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        #self.seq_length = seq_length
        self.dropout = nn.Dropout(p=0.2)
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,dropout = 0.25)
        
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size).to(device))
        
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size).to(device))
        
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        
        h_out = h_out.view(-1, self.hidden_size)
        
        out = self.fc(h_out)
        out = self.dropout(out)
       
        return out


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)




class LSTM2(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM2, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.batch_size = 1
        #self.seq_length = seq_length
        
        self.LSTM2 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,batch_first=True,dropout = 0.25)
       
        
        
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(p=0.2)
    def forward(self, x):
        h_1 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size).to(device))
         
        
        c_1 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size).to(device))
        
       
        _, (hn, cn) = self.LSTM2(x, (h_1, c_1))
     
        #print("hidden state shpe is:",hn.size())
        y = hn.view(-1, self.hidden_size)
        
        final_state = hn.view(self.num_layers, x.size(0), self.hidden_size)[-1]
        #print("final state shape is:",final_state.shape)
        out = self.fc(final_state)
        #out = self.dropout(out)
        #print(out.size())
        return out



# =====================================================================================================================================
#                                           Lag-Llama Model Estiamtor For Fine-tuning 
# =====================================================================================================================================



def lag_llama_estiamtor(device, prediction_length,  num_samples, LagLlamaEstimator):
    ckpt = torch.load("./lag-llama.ckpt", map_location=device)
    estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

    estimator = LagLlamaEstimator(
        ckpt_path="./lag-llama.ckpt",
        prediction_length=prediction_length,
        context_length=32,

        # adjust as needed

        # scaling="mean",
        nonnegative_pred_samples=True,
        aug_prob=0,
        lr=5e-4,

        # estimator args
        input_size=estimator_args["input_size"],
        n_layer=estimator_args["n_layer"],
        n_embd_per_head=estimator_args["n_embd_per_head"],
        n_head=estimator_args["n_head"],
        time_feat=estimator_args["time_feat"],
        device = device,


        batch_size=64,
        num_parallel_samples=num_samples,
        trainer_kwargs={
            "max_epochs": 500,
        },  # <- lightning trainer arguments
    )
    
    return estimator