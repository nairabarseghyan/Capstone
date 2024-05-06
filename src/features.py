#IMPORTS 
## Libraries
import pandas as pd 
import category_encoders as ce
from sklearn.preprocessing import StandardScaler



def add_calendar_features(df):
    """
    Add calendar-based features to the DataFrame.

    Parameters:
    - df (DataFrame): DataFrame containing a 'Date' column.

    Returns:
    - DataFrame: Original DataFrame merged with calendar features.
    """
    
    calendar_df = pd.DataFrame(df['Date'])
    calendar_df['Date'] = pd.to_datetime(calendar_df['Date'])
    calendar_df['month'] = calendar_df['Date'].dt.month
    calendar_df['year'] = calendar_df['Date'].dt.year
    calendar_df['day'] = calendar_df['Date'].dt.day
    calendar_df['day_of_week'] = calendar_df['Date'].dt.dayofweek
    calendar_df['day_name'] = calendar_df['Date'].dt.day_name()
    calendar_df['quarter'] = calendar_df['Date'].dt.quarter
    calendar_df['is_weekend'] = (calendar_df['day_of_week'] >= 5).astype(int)
    calendar_df["week_of_year"] = calendar_df["Date"].dt.isocalendar().week
    calendar_df = calendar_df.drop_duplicates(subset=['Date'])
    
    return df.merge(calendar_df, how='left', on='Date')



def add_sales_proxies(data):
    """
    Add sales proxy features like weekly and monthly averages to the DataFrame.

    Parameters:
    - data (DataFrame): Original DataFrame.

    Returns:
    - DataFrame: Modified DataFrame with added sales proxy features.
    """
    
    
    data = data.copy() # Make a copy of the data to avoid modifying the original

    # Compute and merge weekly average sales
    week_average = data.groupby(["Branch", "week_of_year"])["Revenue"].mean().reset_index()
    week_average.rename(columns={"Revenue": "sales_proxy_week"}, inplace=True)
    data = data.merge(week_average, on=["Branch", "week_of_year"], how="left").fillna(0)

    # Compute and merge monthly average sale
    month_average = data.groupby(["Branch", "month"])["Revenue"].mean().reset_index()
    month_average.rename(columns={"Revenue": "sales_proxy_month"}, inplace=True)
    data = data.merge(month_average, on=["Branch", "month"], how="left").fillna(0)

    return data


def feuture_scaling(X_train_raw, X_val_raw, X_test_raw):
    """
    Apply standard scaling to feature sets.

    Parameters:
    - X_train_raw, X_val_raw, X_test_raw (DataFrame or ndarray): Raw feature sets.

    Returns:
    - Tuple of scaled feature sets: (X_train, X_val, X_test).
    """
    
    scaler = StandardScaler()
    scaler.fit(X_train_raw) #Fit scaler to training data

    X_train = scaler.transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)
    
    return X_train, X_val, X_test


def X_y_target_encoder(train_X_raw, train_y, validate_X_raw, test_X_raw, categorical_vars = ['product_type','Brand','manufacture_country','Branch_for_encoding','City'], return_encoder = False):
    """
    Encode categorical variables using target encoding.

    Parameters:
    - train_X_raw, validate_X_raw, test_X_raw (DataFrame): Raw feature DataFrames.
    - train_y (Series): The target variable corresponding to train_X_raw.
    - categorical_vars (list): List of categorical variables to encode.
    - return_encoder (bool): Flag to return the fitted encoder.

    Returns:
    - Encoded feature sets and optionally the encoder.
    """
    
    encoder = ce.TargetEncoder(cols=categorical_vars)
   
    encoder.fit(train_X_raw, train_y) # Fit encoder using training data
    
    # Transform feature sets
    train_X_encoded = encoder.transform(train_X_raw)
    validate_X_encoded =encoder.transform(validate_X_raw)
    test_X_encoded = encoder.transform(test_X_raw)
    
    # Drop unnecessary columns
    train_X_encoded = train_X_encoded.drop(columns = ['day_name','Date','Branch'])
    validate_X_encoded = validate_X_encoded.drop(columns = ['day_name','Date','Branch'])
    test_X_encoded = test_X_encoded.drop(columns = ['day_name','Date','Branch'])
    
    if return_encoder == True: 
        return train_X_encoded, validate_X_encoded, test_X_encoded, encoder
    else: 
        return train_X_encoded, validate_X_encoded, test_X_encoded



def full_data_target_encoder(train_data,val_data, test_data, encoder):
    """
    Apply target encoding to the full dataset including training, validation, and test sets.

    Parameters:
    - train_data, val_data, test_data (DataFrame): Data sets to encode.
    - encoder (TargetEncoder): Pre-fitted TargetEncoder.

    Returns:
    - Tuple of encoded DataFrames: (train_data, val_data, test_data).
    """
    
    columns = ['Revenue', 'Quantity', 'product_type', 'Brand',
       'manufacture_country', 'Branch', 'City', 'IsHoliday', 'month', 'year',
       'day', 'day_of_week', 'day_name', 'quarter', 'is_weekend',
       'week_of_year', 'sales_proxy_week', 'sales_proxy_month','Branch_for_encoding']
    
    train_data[columns] = encoder.transform(train_data[columns])
    val_data[columns] = encoder.transform(val_data[columns])
    test_data[columns] = encoder.transform(test_data[columns])
    
    return train_data, val_data, test_data

def scaler(train_data, test_data, original_data):
    """
    Apply standard scaling to given data sets.

    Parameters:
    - train_data, test_data, original_data (ndarray): Data arrays to scale.

    Returns:
    - Scaled data arrays and the scaler object.
    """
    
    scaler = StandardScaler()
    
    # Reshape data for scaling and apply transformation
    train_data_normalized = scaler.fit_transform(train_data.reshape(-1, 1))
    test_data_normalized = scaler.transform(test_data.reshape(-1, 1))
    full_data_normalized = scaler.transform(original_data.reshape(-1, 1))
    
    return train_data_normalized, test_data_normalized, full_data_normalized, scaler

