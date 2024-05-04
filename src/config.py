import os
import random
import torch
import numpy as np
import subprocess
import sys

# =====================================================================================================================================
#                                                   Main Paths of Project
#
#                                               CHANGE PATHS ACCORDING TO YOUR SYSTEM
# =====================================================================================================================================


################### Operating System MACOS  ##################


CODE_PATH = '/Users/nairabarseghyan/Desktop/project_name/src/'

DATA_PATH = '/Users/nairabarseghyan/Desktop/project_name/data/'

MODEL_PATH = '/Users/nairabarseghyan/Desktop/project_name/models/'



SEED = 42

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    

def find_device():
    # Check if MPS is available
    if torch.backends.mps.is_available():
        print("Using MPS device")
        return "mps"
    # Check if CUDA is available
    elif torch.cuda.is_available():
        print("Using CUDA device")
        return "cuda"
    # Default to CPU if neither MPS nor CUDA is available
    else:
        print("Using CPU device")
        return "cpu"
    
    
def lag_llama_environment(already_installed = False):
    
    os.environ['KMP_DUPLICATE_LIB_OK']='True' #DO NOT CHANGE THIS 
    
    os.chdir(CODE_PATH)
    
    if already_installed:
        
        current_dir = os.getcwd()
        model_repo_name = os.path.join(current_dir, 'lag-llama')
        os.chdir(model_repo_name)
        sys.path.append(os.getcwd())  # Add this line to ensure Python knows where to look
        print(f"Working in directory {os.getcwd()}")
    
        
        
    else:
        git_donwload_command = ['git', 'clone', 'https://github.com/time-series-foundation-models/lag-llama/']
        
        result = subprocess.run(git_donwload_command, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Successfully cloned the repository.")
            
                        
            model_reqs_download_command = ["pip", "install", "-r"  "./lag-llama/requirements.txt", "--quiet", "--use-deprecated=legacy-resolver"]
            result_reqs = subprocess.run(model_reqs_download_command, capture_output=True, text=True)
            
            if result_reqs.returncode == 0:
                print("Successfully downloaded lag-llama requirments")
                
                model_chkpt_download_command =["huggingface-cli", "download", "time-series-foundation-models/Lag-Llama", "lag-llama.ckpt", "--local-dir", "./lag-llama"]
                result_chkpt = subprocess.run(model_chkpt_download_command, capture_output=True, text=True)
                
                
                if result_chkpt.returncode == 0:
                    print("Successfully downloaded lag-llama checkpoints")
                    
                else:
                    print("Failed to donwload lag-llama checkpoints.")
                    print(result_chkpt.stderr)
                    
                
            else:
                print("Failed to donwload lag-llama requirments.")
                print(result_reqs.stderr)
        else:
            print("Failed to clone the repository.")
            print(result.stderr)


