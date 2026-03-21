# Import libraries
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler

def run(df_cleaned_stores, dir_figures):
    """
    Trains forecasting models (ARIMA, Prophet, XGBoost, LSTM) for each store and generates forecasts.

    Args:
        df_cleaned_stores (dict): Dictionary of cleaned DataFrames for each store.
        dir_figures (str): Path to save visualizations.

    Returns:
        dict: Dictionary of forecasts for each store.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    forecasts = {}

    for store_id, df_store in df_cleaned_stores.items():
        logging.info(f"\nProcessing Store {store_id}...")

        try:
            # Split the dataset into training (80%) and testing (20%) sets
            logging.info("Splitting dataset into training (80%) and testing (20%) sets...")
            split_index = int(len(df_store) * 0.8)
            train_df = df_store.iloc[:split_index]
            test = df_store.iloc[split_index:]
            logging.info(f"Train set size: {train_df.shape[0]}, Test set size: {test.shape[0]}")

            # Train ARIMA model
            logging.info("Training ARIMA model...")
            train_sales = train_df.set_index('Date').sort_index()['Sales'].asfreq('D').ffill()
            test_sales = test["Sales"]
            arima_model = ARIMA(train_sales, order=(5, 1, 0))
            arima_model_fit = arima_model.fit()
            logging.info("ARIMA model trained!")

            # Walk-Forward Validation for ARIMA
            logging.info("Performing walk-forward validation for ARIMA...")
            forecast = arima_model_fit.forecast(steps=len(test_sales))
            logging.info(f"ARIMA walk-forward validation completed for {len(test_sales)} periods!")

            # Plot ARIMA forecast
            logging.info("Plotting ARIMA forecast...")
            plt.figure(figsize=(12, 6))
            plt.plot(train_sales.index, train_sales, label="Train")
            plt.plot(test_sales.index, test_sales, label="Real")
            plt.plot(test_sales.index, forecast, label="Forecast")
            plt.title(f"ARIMA Forecast vs Actual for Store {store_id}")
            plt.xlabel("Date")
            plt.ylabel("Sales")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{dir_figures}store_{store_id}_arima_forecast.png", dpi=300)
            plt.close()
            logging.info("ARIMA forecast plot saved!")

            # Train Prophet model
            logging.info("Training Prophet model...")
            prophet_model = Prophet()
            prophet_train = train_df[["Date", "Sales"]].rename(columns={"Date": "ds", "Sales": "y"})
            prophet_model.fit(prophet_train)
            logging.info("Prophet model trained!")

            # Generate Prophet forecast
            logging.info("Generating Prophet forecast...")
            future_dates = pd.DataFrame({'ds': test["Date"]})
            forecast_prophet = prophet_model.predict(future_dates)
            prophet_predictions = forecast_prophet['yhat'].values

            # Plot Prophet forecast
            logging.info("Plotting Prophet forecast...")
            prophet_model.plot(forecast_prophet)
            plt.title(f"Prophet Forecast for Store {store_id}")
            plt.tight_layout()
            plt.savefig(f"{dir_figures}store_{store_id}_prophet_forecast.png", dpi=300)
            plt.close()
            logging.info("Prophet forecast plot saved!")

            # Train XGBoost model
            logging.info("Training XGBoost model...")

            # Create time features
            df_store["Day"] = df_store["Date"].dt.day
            df_store["Weekday"] = df_store["Weekday"].astype("category")
            df_store["Weekday_code"] = df_store["Weekday"].cat.codes

            features = [
                "DayOfWeek",
                "DayOfYear",
                "IsWeekend",
                "IsPromo",
                "IsStateHoliday",
                "IsSchoolHoliday",
                "CompetitionOpen",
                "CompetitionDistance"
            ]

            X = df_store[features]
            y = df_store["Sales"]

            # Train/test split
            X_train = X.iloc[:split_index]
            X_test = X.iloc[split_index:]
            y_train = y.iloc[:split_index]
            y_test = y.iloc[split_index:]

            logging.info(f"Training XGBoost on {len(X_train)} samples, testing on {len(X_test)} samples...")

            # Test two simple configurations
            configs = [
                {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1},
                {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1},
            ]

            best_rmse = float('inf')
            best_config = {}

            for config in configs:
                logging.info(f"Testing configuration: {config}")
                model = xgb.XGBRegressor(**config, random_state=42)
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, predictions))
                logging.info(f"RMSE: {rmse:.2f}")

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_config = config

            logging.info(f"Best configuration: {best_config} with RMSE: {best_rmse:.2f}")

            # Train the final XGBoost model
            xgb_model = xgb.XGBRegressor(**best_config, random_state=42)
            xgb_model.fit(X_train, y_train)
            xgb_predictions = xgb_model.predict(X_test)
            logging.info("XGBoost model trained!")

            # Train LSTM model
            logging.info("Training LSTM model...")

            # Scale sales
            scaler = MinMaxScaler()
            sales_scaled = scaler.fit_transform(train_df[["Sales"]])
            test_sales_scaled = scaler.transform(test[["Sales"]])

            # Create sequences
            seq_length = 30

            def create_sequences(data, seq_length):
                X, y = [], []
                for i in range(len(data) - seq_length):
                    X.append(data[i:i + seq_length])
                    y.append(data[i + seq_length])
                return np.array(X), np.array(y)

            X_lstm, y_lstm = create_sequences(sales_scaled, seq_length)

            # Train-test split
            split_lstm = int(len(X_lstm) * 0.8)
            X_train_lstm, X_test_lstm = X_lstm[:split_lstm], X_lstm[split_lstm:]
            y_train_lstm, y_test_lstm = y_lstm[:split_lstm], y_lstm[split_lstm:]

            # Build and train LSTM model
            model = Sequential()
            model.add(Input(shape=(seq_length, 1)))
            model.add(LSTM(50))
            model.add(Dense(1))
            model.compile(optimizer="adam", loss="mse")
            model.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=32)

            # Predict and inverse scale
            lstm_predictions = model.predict(X_test_lstm)
            lstm_predictions = scaler.inverse_transform(lstm_predictions)
            y_test_unscaled = scaler.inverse_transform(y_test_lstm.reshape(-1, 1))

            # Calculate metrics
            lstm_rmse = np.sqrt(mean_squared_error(y_test_unscaled, lstm_predictions))
            lstm_mape = np.mean(
                np.abs((y_test_unscaled.flatten() - lstm_predictions.flatten()) / y_test_unscaled.flatten())) * 100
            logging.info(f"LSTM model trained...\nRMSE: {lstm_rmse:.2f}, MAPE: {lstm_mape:.2f}%")

            # Plot LSTM forecast
            lstm_dates = df_store["Date"].iloc[split_lstm + seq_length:split_lstm + seq_length + len(lstm_predictions)]
            logging.info("Plotting LSTM forecast...")
            plt.figure(figsize=(12, 6))
            plt.plot(lstm_dates, y_test_unscaled, label="Actual")
            plt.plot(lstm_dates, lstm_predictions, label="LSTM Prediction")
            plt.title(f"LSTM Forecast vs Actual for Store {store_id}")
            plt.xlabel("Date")
            plt.ylabel("Sales")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{dir_figures}store_{store_id}_lstm_forecast.png", dpi=300)
            plt.close()
            logging.info("LSTM forecast plot saved!")

            # Predict next 30 days demand using Prophet
            logging.info("Predicting next 30 days demand using Prophet...")
            future_30 = prophet_model.make_future_dataframe(periods=30, freq='D', include_history=False)
            forecast_30 = prophet_model.predict(future_30)

            # Store forecasts
            forecasts[store_id] = {
                "forecast": forecast,
                "xgb_predictions": xgb_predictions,
                "prophet_predictions": prophet_predictions,
                "y_test": y_test,
                "test_sales": test_sales,
                "lstm_rmse": lstm_rmse,
                "lstm_mape": lstm_mape,
                "forecast_30": forecast_30
            }

            logging.info("All forecasting models trained for Store {store_id}!\n")

        except Exception as e:
            logging.error(f"Failed to process Store {store_id}: {e}", exc_info=True)
            raise

    return forecasts
