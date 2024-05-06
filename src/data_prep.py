#IMPORTS 
## Libraries
import pandas as pd
import numpy as np
import torch
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName

# Import custom functionalities 
from config import DATA_PATH

def load_and_preprocess_data(PATH = DATA_PATH, encoding= False):
    """
    Load sales and holiday data, preprocess, and merge into a single DataFrame.

    Parameters:
    - PATH (str): Base directory for data files.
    - encoding (bool): If True, add a column for branch encoding.

    Returns:
    - final_df (DataFrame): Preprocessed and merged data frame.
    """
    sales = pd.read_csv(f'{PATH}processed/sales_whole.csv')
    holidays = pd.read_csv(f'{PATH}/external/armenian_holidays.csv', encoding="UTF-8")
    holidays.rename(columns={'Date': 'sale_date'}, inplace=True)
    sales = sales.merge(holidays, on='sale_date', how='left')
    sales['IsHoliday'] = sales['Holiday_name'].notna().astype(int)
    sales.drop(columns=['Holiday_name'], inplace=True)
    sales['sale_date'] = pd.to_datetime(sales['sale_date'])
    sales = sales.sort_values(by='sale_date')
    sales = sales.dropna(subset=['Revenue+VAT'])
    sales.rename(columns={"Revenue+VAT": "Revenue", "sale_date": "Date"}, inplace=True)
    final_df = sales[['Date', 'Revenue', 'Quantity', 'product_type', 'Brand', 'manufacture_country', 'Branch', 'City', 'IsHoliday']]
    if encoding == True:
        final_df['Branch_for_encoding'] = final_df["Branch"]
    else: 
        pass
    return final_df


def split_data(data, test_size=0.10, val_size=0.10):
    """
    Split data into training, validation, and testing sets.

    Parameters:
    - data (DataFrame): The input data set.
    - test_size (float): The proportion of the dataset to include in the test split.
    - val_size (float): The proportion of the dataset to include in the validation split.

    Returns:
    - train_data, val_data, test_data (DataFrames): Training testing and validation sets.
    """
    
    test_val_ratio = test_size + val_size
    val_ratio = val_size
    train_split_index = int(len(data) * (1 - test_val_ratio))
    test_split_index = int(len(data) * (1 - val_ratio))
    
    
    train_data = data[:train_split_index]
    val_data = data[train_split_index:test_split_index]
    test_data = data[test_split_index:]
    
    return train_data, val_data, test_data



def split_univariate_data(data):
    """
    Splits univariate data set into training and testing sets.

    Parameters:
    - data (DataFrame): The data set to split.

    Returns:
    - train_data, test_data (DataFrames): Training and testing sets.
    """
    
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    return train_data, test_data
    
    
def get_x_y(train_data, val_data, test_data, removing = ['Revenue']):
    """
    Prepare feature matrices (X) and target vectors (y) for training, validation, and testing.

    Parameters:
    - train_data, val_data, test_data (DataFrame): Input data sets.
    - removing (list): Columns to exclude from feature matrices.

    Returns:
    - Feature matrices and target vectors for all data sets.
    """
    
    train_X = train_data.drop(columns = removing)
    validate_X = val_data.drop(columns = removing)
    test_X = test_data.drop(columns = removing)

    train_y = train_data["Revenue"]
    validate_y = val_data["Revenue"]
    test_y = test_data["Revenue"]
    
    return train_X, train_y, validate_X, validate_y, test_X, test_y


def general_revenue_dataframe(data):
    """
    Aggregate revenue data by date.

    Parameters:
    - data (DataFrame): Input data frame with 'Date' and 'Revenue' columns.

    Returns:
    - daily_revenue_df (DataFrame): Aggregated daily revenue data.
    """

    df = data[['Date', 'Revenue']].copy()
    df['Date'] = pd.to_datetime(df['Date'])
    
    daily_revenue_df = df.groupby([df['Date'].dt.date, ])['Revenue'].sum().reset_index()
    daily_revenue_df = daily_revenue_df.set_index('Date')
    
    return daily_revenue_df


def arrays_to_tensors(trainX, trainY, testX, testY, fullX, fullY):
    """
    Convert arrays to PyTorch tensors for training and testing.

    Parameters:
    - trainX, trainY, testX, testY, fullX, fullY (array-like): Arrays to be converted.

    Returns:
    - Corresponding PyTorch tensors for all inputs.
    """
    trainX_tensor = torch.Tensor(trainX)
    trainY_tensor = torch.Tensor(trainY)
    testX_tensor = torch.Tensor(testX)
    testY_tensor = torch.Tensor(testY)
    dataX = torch.Tensor(fullX)
    dataY = torch.Tensor(fullY)
    
    return trainX_tensor, trainY_tensor, testX_tensor, testY_tensor, dataX, dataY


def get_lag_llama_dataset(dataset, frequency):
    """
    Prepare a dataset for the lag-llama model.

    Parameters:
    - dataset (DataFrame): The input data frame.
    - frequency (str): The frequency of the time series data.

    Returns:
    - backtest_dataset (ListDataset): Dataset formatted for GluonTS models.
    """
    
    # avoid mutations
    dataset = dataset.copy()

    # convert numerical columns to `float32`
    for col in dataset.columns:
        if dataset[col].dtype != "object" and not pd.api.types.is_string_dtype(
            dataset[col]
        ):
            dataset[col] = dataset[col].astype("float32")

    #freq = "1D"

    # Ensure timestamps are in pandas Timestamp format
    dataset.index = pd.to_datetime(dataset.index)

    # Create a ListDataset with proper frequency and data format
    backtest_dataset = ListDataset([
        {
            FieldName.START: dataset.index[0],  # Start timestamp of the time series
            FieldName.TARGET: dataset[col].values,  # Target values
            FieldName.FEAT_DYNAMIC_REAL: [dataset[col].values],  # Dynamic features
            
        } 
        for col in dataset.columns
    ], freq=frequency)

    # Return the dataset for further processing
    return backtest_dataset