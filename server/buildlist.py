import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

# [[C_CPR, C_CPI, C_CEM, C_GDP, C_IP, U_CIV,], ...] 

# Load individual economic factor datasets
def load_economic_factor(file_path):
    data = pd.read_csv(file_path, header=0, names=['timestamp', 'value'])
    return data['value']

# Preprocess datasets for individual analysis
def preprocess_data(canada_folder, us_folder, currency_rate_file):
    # Load economic factors for both countries
    canada_factors = {}
    us_factors = {}

    for file_name in os.listdir(canada_folder):
        if file_name.endswith('.csv'):
            factor_name = os.path.splitext(file_name)[0]
            file_path = os.path.join(canada_folder, file_name)
            canada_factors[factor_name] = load_economic_factor(file_path)

    for file_name in os.listdir(us_folder):
        if file_name.endswith('.csv'):
            factor_name = os.path.splitext(file_name)[0]
            file_path = os.path.join(us_folder, file_name)
            us_factors[factor_name] = load_economic_factor(file_path)

    # Load currency rate data
    currency_rate_data = pd.read_csv(currency_rate_file, header=None, names=['timestamp', 'currency_rate'])

    # Shift the target column to predict the next month's currency rate
    currency_rate = currency_rate_data['currency_rate'].shift(-1)

    # Drop the last row since the shifted target will be NaN
    currency_rate = currency_rate[:-1]

    features = {}

    # Align data and prepare features for each economic factor individually
    for factor_name, values in {**canada_factors, **us_factors}.items():
        features[factor_name] = values[:-1]  # Drop the last row to match the shifted target

    feature_dfs = {name: pd.DataFrame({name: features[name]}) for name in features}

    # Standardize each factor independently
    scalers = {}
    for name, df in feature_dfs.items():
        scaler = StandardScaler()
        feature_dfs[name] = scaler.fit_transform(df)
        scalers[name] = scaler

    return feature_dfs, currency_rate



