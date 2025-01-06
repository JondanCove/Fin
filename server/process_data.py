from numpy import concatenate, array
import pandas as pd


def preprocess_data(canada_data, us_data):
    """
    Loads economic factor data from CSV files for Canada and the US, combines them into a single
    DataFrame, and returns the result. The method merges the data by concatenating the numpy arrays
    of both datasets and re-constructs a DataFrame by combining their column headers.
    This function prepares the datasets for further data processing and analysis.

    :param canada_data: File path to the CSV dataset containing Canada's economic factor data.
    :type canada_data: str
    :param us_data: File path to the CSV dataset containing the United States' economic factor
        data.
    :type us_data: str
    :return: A pandas DataFrame combining the economic factors from both countries
    :rtype: pandas.DataFrame
    """
    # Load economic factor data from csv files for both countries and combine them into a dataframe
    canada_numpy = pd.read_csv(canada_data, header=0).to_numpy()
    us_numpy = pd.read_csv(us_data, header=0).to_numpy()

    data_merged = concatenate((canada_numpy, us_numpy), axis=1)

    canada_header = pd.read_csv(canada_data, header=0).columns
    us_header = pd.read_csv(us_data, header=0).columns

    headers_merged = concatenate((canada_header, us_header), axis=0)

    df = pd.DataFrame(data_merged, columns=headers_merged)

    return df


def create_sequences(df_input, df_output, window_size):
    """
    Convert input and output DataFrames into sequences for LSTM.

    Args:
        df_input (pd.DataFrame): Input data with shape (num_timesteps, num_features).
        df_output (pd.DataFrame): Output data with shape (num_timesteps, 1).
        window_size (int): Number of time steps per sequence.

    Returns:
        X (np.ndarray): Input sequences of shape (num_samples, window_size, num_features).
        y (np.ndarray): Output values of shape (num_samples, 1).
    """
    X, y = [], []

    for i in range(len(df_input) - window_size):
        X.append(df_input.iloc[i:i + window_size].values)
        y.append(df_output.iloc[i + window_size].values)

    return array(X), array(y)
