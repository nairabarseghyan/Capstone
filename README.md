# Capstone Project: Analysis and Time Series Forecasting of Sales for "NOR TUN" Chain Stores

## Author
**Naira Maria Barseghyan**  
American University of Armenia

## Description
This Capstone project is conducted for the American University of Armenia and is divided into two main parts:

1. **Sales Analysis:** Analyzing the sales data of "NOR TUN" chain stores to uncover trends and patterns that could inform strategic decisions.
2. **Time Series Forecasting:** Developing and implementing time series forecasting models to predict future sales of "NOR TUN" chain stores, aiding in better inventory and business management.

## Installation Instructions

### Prerequisites
- Python 3.11

### Setting up a Python Virtual Environment
To run this project, you will need Python 3.11. It is recommended to use a virtual environment to avoid conflicts with other packages. Follow these steps to set up your environment:

1. **Install Python 3.11**  
   Ensure that Python 3.11 is installed on your system. You can download it from [python.org](https://www.python.org/downloads/release/python-3110/).

2. **Create a Virtual Environment**  
   Open a terminal and run the following command:
   ```bash
   python3.11 -m venv venv
   ```
3. **Create a Virtual Environment**  
  * On Windows 
  ```bash
  venv\Scripts\activate
  ```
  
  * On MacOS/Linux:
  ```bash
  source venv/bin/activate
  ```
  
4. **Install Required Packages**  
    ```bash
    pip install -r requirements.txt
    ```
    
## Project Structure
```
Capstone/
   │
   ├── data/                            # Data files and scripts
   │   ├── raw/                         # Unprocessed data
   │   ├── processed/                   # Cleaned and preprocessed data
   │   ├── external/                    # Data from external sources
   │   └── results/                     # Final analysis results and outputs
   ├── models/                          # Trained model files and scripts
   │   └── *.pkl                        # Saved models in Pickle format
   ├── notebooks/                       # Jupyter notebooks for exploration and presentation
   │   ├── Jupyter/
   │   │   ├── data_preprocessing.ipynb # Notebook for data preprocessing
   │   │   └── translation_library.py   # Script for data translation features
   │   └── rmd/
   │       ├── graph_tests.Rmd          # R Markdown for visual testing
   │       └── preprocessing.R          # R script for data preprocessing
   ├── src/                             # Source code for this project
   │   ├── config.py                    # Configuration settings and constants
   │   ├── data_prep.py                 # Data preparation utilities
   │   ├── evaluate.py                  # Evaluation metrics and functions
   │   ├── features.py                  # Feature engineering scripts
   │   ├── models.py                    # Machine learning model definitions
   │   ├── inference.py                 # Scripts for making inferences
   │   ├── train_ML.py                  # Script for training ML models
   │   ├── train_LSTM.py                # Script for training LSTM models
   │   ├── predictions.py               # Script for generating predictions
   │   ├── lag_llama_inference.ipynb    # Notebook for Lag-Llama model inference
   │   └── fine_tune_lag_llama.ipynb    # Notebook for fine-tuning Lag-Llama model
   ├── reports/                         # Generated analysis reports and summaries
   ├── README.md
   └── requirements.txt                 # Python dependencies for the project

```
