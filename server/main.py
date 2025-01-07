import os

import pandas as pd
from keras.api import Sequential, activations, regularizers, optimizers, losses
from keras.api.callbacks import EarlyStopping
from keras.api.callbacks import ReduceLROnPlateau
from keras.api.layers import Dense, Dropout, Normalization, Input, LSTM
from keras.api.models import load_model

os.environ["KERAS_BACKEND"] = "tensorflow"
from sklearn.model_selection import train_test_split
from process_data import preprocess_data, create_sequences
import matplotlib.pyplot as plt

WINDOW_SIZE = 12


def build_model(input_shape):
    """
    Builds and compiles a sequential model.

    :param input_shape: Tuple containing the shape of the input data.
    :type input_shape: tuple
    :return: Compiled LSTM-based model.
    :rtype: keras.models.Sequential
    """
    lstm_model = Sequential([
        Input(shape=input_shape),
        # Normalization(),
        LSTM(128, return_sequences=True, recurrent_dropout=0.2),
        Dropout(0.2),
        LSTM(64, recurrent_dropout=0.2),
        Dropout(0.2),
        Dense(32, activation=activations.leaky_relu, kernel_regularizer=regularizers.l1(l1=0.01)),
        Dense(1, activation=activations.linear)
    ])

    lstm_model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss=losses.mean_squared_error, metrics=[losses.mean_absolute_error])
    return lstm_model


def train_model_for_factors(feature, target, show_plot=True):
    """
    Train a model using given features and target values, with an option to plot the training
    and validation loss curves. This function includes data splitting, training, validation,
    and optional visualization of the loss curves over epochs.

    :param feature: Input features for the model.
    :type feature: numpy.ndarray

    :param target: Corresponding target values for the features.
    :type target: numpy.ndarray

    :param show_plot: Boolean flag to indicate whether to display the loss curve plot.
                      Defaults to True.
    :type show_plot: bool

    :return: Trained machine learning model after fitting on the provided data.
    :rtype: keras.models.Model
    """
    # Split data into train and test sets
    input_train, input_test, output_train, output_test = train_test_split(feature, target, test_size=0.2, random_state=41)

    # Build and train the model
    model = build_model((WINDOW_SIZE,11))
    early_stop = EarlyStopping(monitor='val_loss', patience=25, verbose=1)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.00001, verbose=1)
    history = model.fit(
        input_train, output_train,
        validation_data=(input_test, output_test),
        epochs=10000,
        callbacks=[early_stop, lr_scheduler],
        batch_size=16,
        verbose=1
    )

    if show_plot:
        # Retrieve loss values from history
        train_loss = history.history['mean_absolute_error']
        val_loss = history.history['val_mean_absolute_error']  # Validation loss, if validation data is provided
        # Plot the loss graph
        plt.figure(figsize=(10, 6))
        plt.plot(train_loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss', linestyle='--')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()
        plt.grid()
        plt.yscale('log')
        plt.show()

    return model

# Main execution
if __name__ == "__main__":
    # Paths to the data folders and currency rate data
    current_folder = os.getcwd()
    input_data = current_folder + "/Data/normalized_large_merged_factors.csv"
    currency_rate_data = current_folder + "/Data/OutputData/canada_to_us_exchange_rate.csv"

    decision = input("What do you want to do? (1 = train new, 2 = test existing)")

    if decision == '1':
        target_df = pd.DataFrame(pd.read_csv(currency_rate_data, header=0).to_numpy(), columns=['exchange_rate'])
        # pandas dataframe for both us and canada data
        feature_df = pd.DataFrame(pd.read_csv(input_data, header=0).to_numpy(), columns=[
            'prime_rate_CA', 'unemployment_CA', 'consumer_price_index_CA', 'GDP_CA', 'industrial_price_index_CA',
            'labor_participation_US', 'consumer_price_index_US', 'population_US', 'price_per_commodity_US',
            'unemployment_US', 'prime_rate_US'
        ])

        input_seq, output_seq = create_sequences(feature_df, target_df, WINDOW_SIZE)

        # Train models for each factor
        final_model = train_model_for_factors(input_seq, output_seq, True)

        predictions = pd.DataFrame(final_model.predict(input_seq), columns=['exchange_rate'])

        print(predictions)
        
        plt.figure(figsize=(20, 6))
        plt.plot(predictions, label='Prediction')
        plt.plot(target_df.to_numpy()[-len(input_seq):], label='Actual')
        plt.legend(loc='upper right')
        plt.xlabel('Time')
        plt.ylabel('Exchange Rate')
        plt.title('Exchange Rate Vs Prediction')
        plt.grid()
        plt.show()

        want_to_save = input("Do you want to save the model? (y/n)")
        if want_to_save == 'y':
            final_model.save('currency_rate_predictor.keras')
            print('Model saved successfully')
    elif decision == '2':
        while True:
            model = load_model('currency_rate_predictor.keras')
            print('Model loaded successfully')

            print('Enter the following information, separated by commas:\nprime_rate_CA,unemployment_CA,consumer_price_index_CA,GDP_CA,industrial_price_index_CA,labor_participation_US,consumer_price_index_US,population_US,price_per_commodity_US,unemployment_US,prime_rate_US')
            user_input = input().split(',')
            user_input = [float(value.strip()) for value in user_input]

            user_input_df = pd.DataFrame([user_input])


            print('model prediction: ')
            print(model.predict(user_input_df))

