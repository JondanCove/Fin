import requests
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout



# Step 1: Fetch Data from APIs
def fetch_financial_data(financial_api_url, currency_api_url):
    # Fetch financial index data
    financial_response = requests.get(financial_api_url)
    financial_data = financial_response.json()

    # Fetch currency data
    currency_response = requests.get(currency_api_url)
    currency_data = currency_response.json()

    # Process and return data (Assume the response contains time-series data)
    return financial_data, currency_data

# Step 2: Preprocess Data
def preprocess_data(financial_data, currency_data):
    # Convert API data into Pandas DataFrames
    financial_df = pd.DataFrame(financial_data)
    currency_df = pd.DataFrame(currency_data)

    # Merge the dataframes on a time column (adjust based on API response structure)
    combined_df = pd.merge(financial_df, currency_df, on="timestamp")

    # Scale data (Normalize between 0 and 1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(combined_df.drop("timestamp", axis=1))

    # Prepare data for LSTM (look back 30 timesteps to predict the next timestep)
    X, y = [], []
    look_back = 30
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back:i])
        y.append(scaled_data[i, -1])  # Assuming the last column is the target variable

    X, y = np.array(X), np.array(y)
    return X, y, scaler

# Step 3: Build Neural Network Model
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # Output layer for regression
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Step 4: Train and Evaluate Model
def train_model(X, y):
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build the model
    model = build_model(X_train.shape[1:])
    
    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
    
    return model

# Step 5: Predict Future Trends
def predict_future(model, recent_data, scaler):
    # Prepare recent data for prediction
    recent_data_scaled = scaler.transform(recent_data)
    recent_data_scaled = np.array([recent_data_scaled])  # Add batch dimension

    # Predict the future trend
    prediction = model.predict(recent_data_scaled)
    return scaler.inverse_transform(prediction)  # Return prediction in original scale

# Main Workflow
if __name__ == "__main__":
    financial_api_url = "https://api.example.com/financial-index"
    currency_api_url = "https://api.example.com/currency"

    # Fetch and preprocess data
    financial_data, currency_data = fetch_financial_data(financial_api_url, currency_api_url)
    X, y, scaler = preprocess_data(financial_data, currency_data)

    # Train the model
    model = train_model(X, y)

    # Predict future trends
    recent_data = X[-1]  # Use the most recent data for prediction
    prediction = predict_future(model, recent_data, scaler)

    print("Predicted future trend:", prediction)
