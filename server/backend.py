import pandas as pd
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras import Sequential
from keras import layers

# Load individual economic factor datasets
def load_economic_factor(file_path):
    data = pd.read_csv(file_path, header=None, names=['timestamp', 'value'])
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

# Build the neural network model
def build_model():
    model = Sequential([
        layers.Dense(64, activation='relu', input_dim=1),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1)  # Single output for currency rate prediction
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Train the model for each factor individually
def train_model_for_factors(feature_dfs, target):
    models = {}

    for factor_name, features in feature_dfs.items():
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        # Build and train the model
        model = build_model()
        model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=50,
            batch_size=32,
            verbose=1
        )

        models[factor_name] = model
        print(f"Model training complete for factor: {factor_name}")

    return models

# Main execution
if __name__ == "__main__":
    # Paths to the data folders and currency rate file
    canada_folder = "./Data/CanadaData"
    us_folder = "./Data/USData"
    currency_rate_file = "./Data/OutputData/canada_to_us_exchange_rate.csv"

    # Preprocess the data
    feature_dfs, target = preprocess_data(canada_folder, us_folder, currency_rate_file)

    # Train models for each factor
    models = train_model_for_factors(feature_dfs, target)

    # Save the trained models
    for factor_name, model in models.items():
        model.save(f"{factor_name}_model.h5")

    print("All models trained and saved.")
