######################################
# FOR TESTING PURPOSES               #
# Not sure if these are reproducible #
# with individual inputs             #
######################################

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('normalized_large_merged_factors.csv')

# Initialize scalers
min_max_scaler = MinMaxScaler()
standard_scaler = StandardScaler()

# Iterate through each column and normalize individually
for column in df.columns:
    if df[column].max() - df[column].min() > 1e6:
        # For columns with very large magnitudes, use Min-Max Scaling
        df[column] = min_max_scaler.fit_transform(df[[column]])
        # Comment: Applied Min-Max Scaling to column '{}' to scale values to the range [0, 1]
    elif df[column].std() > 1e2:
        # For columns with high standard deviation, use Standard Scaling for zero-mean and unit variance
        df[column] = standard_scaler.fit_transform(df[[column]])
        # Comment: Applied Standard Scaling to column '{}' to standardize values to zero mean and unit variance
    else:
        # For other columns, normalize each value independently to unit scale
        df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
        # Comment: Scaled column '{}' to a [0, 1] range by its min and max values

# Save the normalized DataFrame to a new CSV
df.to_csv('normalized_large_merged_factors.csv', index=False)