import pandas as pd 
import category_encoders as ce
from sklearn.preprocessing import StandardScaler




def add_calendar_features(df):
    
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
    # Make a copy of the data to avoid modifying the original
    data = data.copy()

    # Calculate weekly average
    week_average = data.groupby(["Branch", "week_of_year"])["Revenue"].mean().reset_index()
    week_average.rename(columns={"Revenue": "sales_proxy_week"}, inplace=True)
    # Merge weekly average
    data = data.merge(week_average, on=["Branch", "week_of_year"], how="left").fillna(0)

    # Calculate monthly average
    month_average = data.groupby(["Branch", "month"])["Revenue"].mean().reset_index()
    month_average.rename(columns={"Revenue": "sales_proxy_month"}, inplace=True)
    # Merge monthly average
    data = data.merge(month_average, on=["Branch", "month"], how="left").fillna(0)

    return data


def feuture_scaling(X_train_raw, X_val_raw, X_test_raw):
    
    scaler = StandardScaler()
    
    scaler.fit(X_train_raw)

    X_train = scaler.transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)
    
    return X_train, X_val, X_test


def X_y_target_encoder(train_X_raw, train_y, validate_X_raw, test_X_raw, categorical_vars = ['product_type','Brand','manufacture_country','Branch_for_encoding','City'], return_encoder = False):
    
    encoder = ce.TargetEncoder(cols=categorical_vars)
   
    encoder.fit(train_X_raw, train_y)
    
    train_X_encoded = encoder.transform(train_X_raw)
    validate_X_encoded =encoder.transform(validate_X_raw)
    test_X_encoded = encoder.transform(test_X_raw)
    
    train_X_encoded = train_X_encoded.drop(columns = ['day_name','Date','Branch'])
    validate_X_encoded = validate_X_encoded.drop(columns = ['day_name','Date','Branch'])
    test_X_encoded = test_X_encoded.drop(columns = ['day_name','Date','Branch'])
    
    if return_encoder == True: 
        return train_X_encoded, validate_X_encoded, test_X_encoded, encoder
    else: 
        return train_X_encoded, validate_X_encoded, test_X_encoded



def full_data_target_encoder(train_data,val_data, test_data, encoder):
    columns = ['Revenue', 'Quantity', 'product_type', 'Brand',
       'manufacture_country', 'Branch', 'City', 'IsHoliday', 'month', 'year',
       'day', 'day_of_week', 'day_name', 'quarter', 'is_weekend',
       'week_of_year', 'sales_proxy_week', 'sales_proxy_month','Branch_for_encoding']
    
    train_data[columns] = encoder.transform(train_data[columns])
    val_data[columns] = encoder.transform(val_data[columns])
    test_data[columns] = encoder.transform(test_data[columns])
    
    return train_data, val_data, test_data

def scaler(train_data, test_data, original_data):

    scaler = StandardScaler()
    
    train_data_normalized = scaler.fit_transform(train_data.reshape(-1, 1))
    test_data_normalized = scaler.transform(test_data.reshape(-1, 1))
    full_data_normalized = scaler.transform(original_data.reshape(-1, 1))
    
    return train_data_normalized, test_data_normalized, full_data_normalized, scaler

