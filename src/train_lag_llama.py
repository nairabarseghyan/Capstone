# import modules 
import pickle
import os
import time 
import torch


from data import load_and_preprocess_data, general_revenue_dataframe, split_univariate_data, get_lag_llama_dataset
from config import MODEL_PATH, SEED, seed_everything, lag_llama_environment, find_device



# SET UP ENVIRONMENT
os.environ['KMP_DUPLICATE_LIB_OK']='True'
seed_everything(SEED)

start_time = time.time()

#lag_llama_environment()


from models import lag_llama_estiamtor




llama_device = torch.device(find_device())



#loading data and adding necessary features
initial_data = load_and_preprocess_data()
univariate_data = general_revenue_dataframe(initial_data)

train_data_raw, test_data = split_univariate_data(univariate_data)


# Tranfrom dataset for lag_llama input

prediction_length = len(test_data)  # prediction length
num_samples = 1000 # sampled from the distribution for each timestep

#train on first 
train_data = get_lag_llama_dataset(train_data_raw,frequency = "1D")


estimator = lag_llama_estiamtor(llama_device, prediction_length, num_samples)

predictor = estimator.train(
    train_data,
    cache_data=True,
    shuffle_buffer_length=1000,
)
print("Fine - Tuning process of Lag-Llama model is completed in --- %s seconds ---" % (time.time() - start_time))


## Saving the model

lag_llama_model_filename = os.path.join(MODEL_PATH, 'fine_tuned_lag_llama_model.pth')

torch.save(predictor, lag_llama_model_filename)



print("The Fine - Tuning process of Lag-Llama model is completed.")