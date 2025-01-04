import pandas as pd
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras import layers
from process_data import preprocess_data

# Build the neural network model
def build_model():
    model = Sequential([
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1)  # Single output for currency rate prediction
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Train the model for each factor individually
def train_model_for_factors(feature_dfs, target):

    # Split data into train and test sets
    input_train, input_test, output_train, output_test = train_test_split(feature_dfs, target, test_size=0.2, random_state=42)

    # Build and train the model
    model = build_model()
    model.fit(
        input_train, output_train,
        validation_data=(input_test, output_test),
        epochs=50,
        batch_size=32,
        verbose=1
    )


    return model

# Main execution
if __name__ == "__main__":
    # Paths to the data folders and currency rate data
    canada_data = "./Data/CanadaData/canada_merged.csv"
    us_data = "./Data/USData/us_merged.csv"
    currency_rate_data = "./Data/OutputData/canada_to_us_exchange_rate.csv"

    target = pd.DataFrame(pd.read_csv(currency_rate_data, index_col=0), columns=['exchange_rate'])
    # pandas dataframe for both us and canada data
    feature_dfs = preprocess_data(canada_data, us_data)

    # Train models for each factor
    models = train_model_for_factors(feature_dfs, target)

    # Save the trained models
    for factor_name, model in models.items():
        model.save(f"{factor_name}_model.h5")

    print("All models trained and saved.")
