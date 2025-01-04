import pandas as pd
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras import layers

# Build the neural network model
def build_model():
    model = Sequential([
        layers.Dense(64, activation='leaky_relu', input_dim=1),
        layers.Dropout(0.2),
        layers.Dense(32, activation='leaky_relu'),
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

    target = pd.read_csv(currency_rate_file, index_col=0)
    feature_dfs = {} # TODO

    # Train models for each factor
    models = train_model_for_factors(feature_dfs, target)

    # Save the trained models
    for factor_name, model in models.items():
        model.save(f"{factor_name}_model.h5")

    print("All models trained and saved.")
