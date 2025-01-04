from numpy import concatenate
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Preprocess datasets for individual analysis
def preprocess_data(canada_data, us_data):
    # Load economic factor data from csv files for both countries and combine them into a dataframe
    canada_numpy = pd.read_csv(canada_data, header=0).to_numpy()
    us_numpy = pd.read_csv(us_data, header=0).to_numpy()

    data_merged = concatenate((canada_numpy, us_numpy), axis=1)

    canada_header = pd.read_csv(canada_data, header=0).columns
    us_header = pd.read_csv(us_data, header=0).columns

    headers_merged = concatenate((canada_header, us_header), axis=0)

    df = pd.DataFrame(data_merged, columns=headers_merged)

    # Standardize each factor independently
    # scalers = {}
    # for name, df in feature_dfs.items():
    #     scaler = StandardScaler()
    #     feature_dfs[name] = scaler.fit_transform(df)
    #     scalers[name] = scaler

    return df



