import pandas as pd
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras import layers
from keras import models
from process_data import preprocess_data
import matplotlib.pyplot as plt


def build_model():
    """
    Builds and compiles a Sequential model for currency rate prediction.

    :return: Compiled Keras Sequential model.
    :rtype: keras.engine.sequential.Sequential
    """
    model = Sequential([
        layers.Input(shape=(11,)),
        layers.BatchNormalization(),
        layers.Dense(64, activation=None, kernel_initializer='he_normal'),
        layers.LeakyReLU(negative_slope=0.1),
        layers.Dropout(0.1),
        layers.Dense(32, activation=None, kernel_initializer='he_normal'),
        layers.LeakyReLU(negative_slope=0.1),
        layers.Dropout(0.1),
        layers.Dense(16, activation=None, kernel_initializer='he_normal'),
        layers.LeakyReLU(negative_slope=0.1),
        layers.Dropout(0.1),
        layers.Dense(1, activation=None)  # Single output for currency rate prediction
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def train_model_for_factors(feature, target):
    """
    Trains a machine learning model using the given feature sets and targets. This function splits
    the input data into training and testing sets, builds a model, and trains it. The trained
    model is returned as the output.

    :param feature: A DataFrame containing the feature variables used to predict the target variable.
    :type feature: pandas.DataFrame or numpy.ndarray
    :param target: A DataFrame containing the target variable to be predicted.
    :type target: pandas.DataFrame, pandas.Series, or numpy.ndarray
    :return: The trained machine learning model after fitting.
    :rtype: Model
    """
    # Split data into train and test sets
    input_train, input_test, output_train, output_test = train_test_split(feature, target, test_size=0.2, random_state=42)

    # Build and train the model
    model = build_model()
    history = model.fit(
        input_train, output_train,
        validation_data=(input_test, output_test),
        epochs=1000,
        batch_size=16,
        verbose=1
    )

    # Retrieve loss values from history
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']  # Validation loss, if validation data is provided
    # Plot the loss graph
    plt.figure(figsize=(5, 3))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid()
    plt.show()

    return model

# Main execution
if __name__ == "__main__":
    # Paths to the data folders and currency rate data
    current_folder = os.getcwd()
    canada_data = current_folder + "/Data/CanadaData/canada_merged.csv"
    us_data = current_folder + "/Data/USData/us_merged.csv"
    currency_rate_data = current_folder + "/Data/OutputData/canada_to_us_exchange_rate.csv"

    decision = input("What do you want to do? (1 = train new, 2 = test existing)")

    if decision == '1':
        target_df = pd.DataFrame(pd.read_csv(currency_rate_data, header=0).to_numpy(), columns=['exchange_rate'])
        # pandas dataframe for both us and canada data
        feature_df = preprocess_data(canada_data, us_data)

        # normalize_data(feature_df)
        # normalize_data(target_df)

        # Train models for each factor
        final_model = train_model_for_factors(feature_df, target_df)

        final_model.save('currency_rate_predictor.keras')
        print('Model saved successfully')
    elif decision == '2':
        while True:
            model = models.load_model('currency_rate_predictor.keras')
            print('Model loaded successfully')

            print('Enter the following information, separated by commas:\nprime_rate_CA,unemployment_CA,consumer_price_index_CA,GDP_CA,industrial_price_index_CA,labor_participation_US,consumer_price_index_US,population_US,price_per_commodity_US,unemployment_US,prime_rate_US')
            user_input = input().split(',')
            user_input = [float(value.strip()) for value in user_input]

            user_input_df = pd.DataFrame([user_input])


            print('model prediction: ')
            print(model.predict(user_input_df))

