from numpy import concatenate
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Preprocess datasets for individual analysis
def preprocess_data(canada_data, us_data):
    # Load economic factors for both countries
    canada_factors = pd.read_csv(canada_data, header=0).to_numpy()
    us_factors = pd.read_csv(us_data, header=0).to_numpy()

    feature_dfs = {name: pd.DataFrame(concatenate((canada_factors, us_factors), axis=1)) for name in ["Canada", "US"]}

    # Standardize each factor independently
    # scalers = {}
    # for name, df in feature_dfs.items():
    #     scaler = StandardScaler()
    #     feature_dfs[name] = scaler.fit_transform(df)
    #     scalers[name] = scaler

    return feature_dfs



