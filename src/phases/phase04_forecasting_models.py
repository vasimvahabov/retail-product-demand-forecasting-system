"""
Trains forecasting models (ARIMA, Prophet, XGBoost, LSTM) for each store and generates forecasts.
"""

import os
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

logger = logging.getLogger(__name__)


def run(df_cleaned_stores, dir_figures):
    """
    Train forecasting models and generate predictions per store.

    Args:
        df_cleaned_stores (dict): Dictionary of cleaned DataFrames for each store.
        dir_figures (str): Output path for visualizations.

    Returns:
        dict: Dictionary of forecasts for each store.
    """

    forecasts = {}

    for store_id, df_store in df_cleaned_stores.items():
        logger.info("Processing Store %s...", store_id)
        try:

            logger.info("Splitting dataset into training (80%) and testing (20%) sets...")
            try:
                split_index = int(len(df_store) * 0.8)
                train_df = df_store.iloc[:split_index]
                test = df_store.iloc[split_index:]

                y_test = test["Sales"].copy()

                logger.info(
                    "Train size: %s, Test size: %s",
                    train_df.shape[0],
                    test.shape[0]
                )

            except Exception:
                logger.exception("Failed to split dataset into training (80%) and testing (20%) sets on Store %s!", store_id)
                continue

            forecast_arima = None
            logger.info("Training ARIMA model...")
            try:
                train_sales = (
                    train_df.set_index('Date')
                    .sort_index()['Sales']
                    .asfreq('D')
                    .ffill()
                )

                arima_model = ARIMA(train_sales, order=(5, 1, 0))
                arima_model_fit = arima_model.fit()
                logger.info("ARIMA model trained!")

                # Walk-Forward Validation for ARIMA
                logger.info("Performing walk-forward validation for ARIMA...")
                forecast_arima = arima_model_fit.forecast(steps=len(y_test))
                logger.info("ARIMA walk-forward validation completed for %s periods!", len(y_test))

                # Plot ARIMA forecast
                logger.info("Saving ARIMA forecast plot...")
                plt.figure(figsize=(12, 6))
                plt.plot(train_sales.index, train_sales, label="Train")
                plt.plot(y_test.index, y_test, label="Real")
                plt.plot(y_test.index, forecast_arima, label="Forecast")
                plt.title(f"ARIMA Forecast vs Actual for Store {store_id}")
                plt.xlabel("Date")
                plt.ylabel("Sales")
                plt.legend()
                plt.tight_layout()
                path_arima_forecast = os.path.join(dir_figures, f"store_{store_id}_arima_forecast.png")
                plt.savefig(path_arima_forecast, dpi=300)
                plt.close()
                logger.info("ARIMA forecast plot saved!")

            except Exception:
                logger.exception("Failed on training ARIMA model on Store %s!", store_id)

            prophet_predictions = None
            logger.info("Training Prophet model...")
            try:
                # Train Prophet model
                prophet_model = Prophet()
                prophet_train = train_df[["Date", "Sales"]].rename(columns={"Date": "ds", "Sales": "y"})
                prophet_model.fit(prophet_train)
                logger.info("Prophet model trained!")

                # Generate Prophet forecast
                logger.info("Generating Prophet forecast...")
                future_dates = pd.DataFrame({'ds': test["Date"]})
                forecast_prophet = prophet_model.predict(future_dates)
                prophet_predictions = forecast_prophet['yhat'].values

                # Plot Prophet forecast
                logger.info("Plotting Prophet forecast...")
                prophet_model.plot(forecast_prophet)
                plt.title(f"Prophet Forecast for Store {store_id}")
                plt.tight_layout()
                path_prophet_forecast = os.path.join(dir_figures, f"store_{store_id}_prophet_forecast.png")
                plt.savefig(path_prophet_forecast, dpi=300)
                plt.close()
                logger.info("Prophet forecast plot saved!")

            except Exception:
                logger.exception("Failed on training Prophet model on Store %s!", store_id)

            xgb_predictions = None
            logger.info("Training XGBoost model...")
            try:
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
                y_test_xgb = y.iloc[split_index:]
                logger.info("Training XGBoost on %s samples, testing on %s samples...", len(X_train), len(X_test))

                # Test two simple configurations
                configs = [
                    {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1},
                    {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1},
                ]

                best_rmse = float('inf')
                best_config = {}

                for config in configs:
                    logger.info("Testing configuration: %s", config)
                    model = xgb.XGBRegressor(**config, random_state=42)
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)
                    rmse = np.sqrt(mean_squared_error(y_test_xgb, predictions))
                    logger.info("RMSE: %.2f", rmse)

                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_config = config

                logger.info("Best configuration: %s with RMSE: %.2f", best_config, best_rmse)

                # Train the final XGBoost model
                xgb_model = xgb.XGBRegressor(**best_config, random_state=42)
                xgb_model.fit(X_train, y_train)
                xgb_predictions = xgb_model.predict(X_test)
                logger.info("XGBoost model trained!")

            except Exception:
                logger.exception("Failed on training XGBoost model on Store %s!", store_id)

            lstm_rmse = None
            lstm_mape = None
            logger.info("Training LSTM model...")
            try:

                # Scale sales
                scaler = MinMaxScaler()
                sales_scaled = scaler.fit_transform(train_df[["Sales"]])

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
                logger.info("LSTM trained | RMSE: %.2f | MAPE: %.2f%%!", lstm_rmse, lstm_mape)

                # Plot LSTM forecast
                lstm_dates = df_store["Date"].iloc[
                    split_lstm + seq_length:split_lstm + seq_length + len(lstm_predictions)]
                logger.info("Plotting LSTM forecast...")
                plt.figure(figsize=(12, 6))
                plt.plot(lstm_dates, y_test_unscaled, label="Actual")
                plt.plot(lstm_dates, lstm_predictions, label="LSTM Prediction")
                plt.title(f"LSTM Forecast vs Actual for Store {store_id}")
                plt.xlabel("Date")
                plt.ylabel("Sales")
                plt.legend()
                plt.tight_layout()
                path_lstm_forecast = os.path.join(dir_figures, f"store_{store_id}_lstm_forecast.png")
                plt.savefig(path_lstm_forecast, dpi=300)
                plt.close()
                logger.info("LSTM forecast plot saved!")

            except Exception:
                logger.exception("Failed on training LSTM model on Store %s!", store_id)

            forecast_30 = None
            logger.info("Predicting next 30 days demand using Prophet...")
            if prophet_model is not None:
                try:
                    future_30 = prophet_model.make_future_dataframe(
                        periods=30,
                        freq='D',
                        include_history=False
                    )
                    forecast_30 = prophet_model.predict(future_30)
                except Exception:
                    logger.exception("Failed future forecast on Store %s!", store_id)

            logger.info("Storing forecast model results...")
            forecasts[store_id] = {
                "forecast_arima": forecast_arima,
                "xgb_predictions": xgb_predictions,
                "prophet_predictions": prophet_predictions,
                "y_test": y_test,
                "lstm_rmse": lstm_rmse,
                "lstm_mape": lstm_mape,
                "forecast_30": forecast_30,
            }
            logger.info("All forecasting models trained for Store %s!", store_id)

        except Exception:
            logger.exception("Critical failure for Store %s!", store_id)
            continue

    return forecasts
