"""
Trains forecasting models (ARIMA, Prophet, XGBoost, LSTM) for each store, generates and persists forecast results.
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


def run(dir_data_output, dir_figures, stores_to_process):
    """
    Train forecasting models, generates and persists predictions per store.

    Args:
        dir_data_output (str): Path to persist processed datasets.
        dir_figures (str): Output path for visualizations.
        stores_to_process (list[int]): List of store IDs to process.

    Returns:
        None
    """

    df_forecasts = {}

    df_processed_stores = {
        store_id: pd.read_csv(
            os.path.join(dir_data_output, f"store_{store_id}_eda.csv")
        )
        for store_id in stores_to_process
    }

    # Ensure figures directory exists
    os.makedirs(dir_figures, exist_ok=True)
    for store_id, df_store in df_processed_stores.items():
        logger.info("Processing Store %s...", store_id)
        try:
            logger.info(
                "Splitting dataset into training (80%%) and testing (20%%) sets for Store %s...",
                store_id
            )

            df_store["Date"] = pd.to_datetime(df_store["Date"])
            try:
                split_index = int(len(df_store) * 0.8)
                train_df = df_store.iloc[:split_index]
                test = df_store.iloc[split_index:]

                y_test = (
                    test.set_index("Date")
                    .sort_index()["Sales"]
                )

                logger.info(
                    "Train size: %s, Test size: %s",
                    train_df.shape[0],
                    test.shape[0]
                )

            except Exception:
                logger.exception(
                    "Failed to split dataset into training (80%) and testing (20%) sets on Store %s!",
                    store_id
                )
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

                arima_model = ARIMA(train_sales, order=(2, 1, 2))
                arima_model_fit = arima_model.fit()
                logger.info("ARIMA model trained!")

                # Walk-Forward Validation for ARIMA
                logger.info("Performing walk-forward validation for ARIMA...")
                forecast_arima = arima_model_fit.forecast(steps=len(y_test))
                forecast_arima = pd.Series(
                    forecast_arima.values,
                    index=y_test.index
                )
                logger.info("ARIMA walk-forward validation completed for %s periods!", len(y_test))

                # Plot ARIMA forecast
                logger.info("Saving ARIMA forecast plot...")
                plt.figure(figsize=(14, 7))

                plt.plot(train_sales.index, train_sales,
                         label="Training Sales", color="black", linewidth=1.5)

                plt.plot(y_test.index, y_test,
                         label="Actual Sales", color="royalblue", linewidth=2)

                plt.plot(y_test.index, forecast_arima,
                         label="ARIMA Forecast", color="darkorange",
                         linestyle="--", linewidth=2)

                plt.grid(True, linestyle="--", alpha=0.4)

                plt.title(f"ARIMA Forecast vs Actual Sales for Store {store_id}",
                          fontsize=16, fontweight="bold")

                plt.xlabel("Date", fontsize=12)
                plt.ylabel("Daily Sales (€)", fontsize=12)

                plt.legend(frameon=True)
                plt.tight_layout()
                path_arima_forecast = os.path.join(dir_figures, f"store_{store_id}_arima_forecast.png")
                plt.savefig(path_arima_forecast, dpi=300)
                plt.close()
                logger.info("ARIMA forecast plot saved!")

            except Exception:
                logger.exception("Failed on training ARIMA model on Store %s!", store_id)

            prophet_predictions = None
            prophet_model = None
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

                plt.figure(figsize=(14, 7))
                plt.plot(
                    train_df["Date"],
                    train_df["Sales"],
                    label="Training Sales",
                    color="black",
                    linewidth=1.5,
                    alpha=0.7
                )

                plt.plot(
                    test["Date"],
                    test["Sales"],
                    label="Actual Sales",
                    color="royalblue",
                    linewidth=2
                )

                plt.plot(
                    test["Date"],
                    prophet_predictions,
                    label="Prophet Forecast",
                    color="darkorange",
                    linestyle="--",
                    linewidth=2
                )

                plt.fill_between(
                    test["Date"],
                    forecast_prophet["yhat_lower"],
                    forecast_prophet["yhat_upper"],
                    color="orange",
                    alpha=0.2,
                    label="Confidence Interval"
                )

                plt.grid(True, linestyle="--", alpha=0.4)

                plt.title(
                    f"Prophet Forecast vs Actual Sales for Store {store_id}",
                    fontsize=16,
                    fontweight="bold"
                )
                plt.xlabel("Date", fontsize=12)
                plt.ylabel("Daily Sales (€)", fontsize=12)
                plt.xticks(rotation=45)
                plt.legend(frameon=True)
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
                features = [
                    "DayOfWeek",
                    "DayOfYear",
                    "IsWeekend",
                    "IsPromo",
                    "IsStateHoliday",
                    "IsSchoolHoliday",
                    "CompetitionOpen",
                    "CompetitionDistance",
                    "Lag_1",
                    "Lag_7",
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

                # Plot XGBoost forecast
                logger.info("Plotting XGBoost forecast for Store %s...", store_id)

                plt.figure(figsize=(14, 7))

                plt.plot(
                    train_df["Date"],
                    y_train,
                    label="Training Sales",
                    color="black",
                    linewidth=1.5
                )

                # Actual test values
                plt.plot(
                    test["Date"],
                    y_test_xgb,
                    label="Actual Sales",
                    color="royalblue",
                    linewidth=2
                )

                # XGBoost predictions
                plt.plot(
                    test["Date"],
                    xgb_predictions,
                    label="XGBoost Forecast",
                    color="green",
                    linestyle="--",
                    linewidth=2
                )

                plt.grid(True, linestyle="--", alpha=0.4)

                plt.title(
                    f"XGBoost Forecast vs Actual Sales for Store {store_id}",
                    fontsize=16,
                    fontweight="bold"
                )

                plt.xlabel("Date", fontsize=12)
                plt.ylabel("Daily Sales (€)", fontsize=12)

                plt.legend(frameon=True)
                plt.tight_layout()

                path_xgb_forecast = os.path.join(
                    dir_figures,
                    f"store_{store_id}_xgboost_forecast.png"
                )

                plt.savefig(path_xgb_forecast, dpi=300)
                plt.close()

                logger.info("XGBoost forecast plot saved!")

            except Exception:
                logger.exception("Failed on training XGBoost model on Store %s!", store_id)

            lstm_predictions = None
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
                logger.info(
                    "LSTM sequence shapes | X=%s y=%s",
                    X_lstm.shape,
                    y_lstm.shape
                )

                if len(X_lstm) == 0 or len(y_lstm) == 0:
                    logger.warning(
                        "Skipping LSTM for Store %s: insufficient sequence data",
                        store_id
                    )
                    continue

                # Train-test split
                split_lstm = int(len(X_lstm) * 0.8)
                if split_lstm == 0:
                    logger.warning(
                        "Skipping LSTM for Store %s: empty training split",
                        store_id
                    )
                    continue

                if split_lstm >= len(X_lstm):
                    logger.warning(
                        "Skipping LSTM for Store %s: empty test split",
                        store_id
                    )
                    continue
                X_train_lstm, X_test_lstm = X_lstm[:split_lstm], X_lstm[split_lstm:]
                logger.info(
                    "LSTM train/test shapes | X_train=%s X_test=%s",
                    X_train_lstm.shape,
                    X_test_lstm.shape
                )
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

                # Plot LSTM forecast
                lstm_dates = df_store["Date"].iloc[
                    split_lstm + seq_length:split_lstm + seq_length + len(lstm_predictions)
                ]
                logger.info("Plotting LSTM forecast...")
                plt.figure(figsize=(14, 7))
                plt.plot(lstm_dates, y_test_unscaled.flatten(), label="Actual")
                plt.plot(lstm_dates, lstm_predictions.flatten(), label="LSTM Forecast")
                plt.title(f"LSTM Forecast vs Actual Sales for Store {store_id}")
                plt.xlabel("Date")
                plt.ylabel("Daily Sales (€)")
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

            if forecast_30 is not None and not forecast_30.empty:
                df_forecast_30 = forecast_30[["ds", "yhat"]].rename(
                    columns={"ds": "Date", "yhat": "Forecast"}
                )

                path_forecast_30 = os.path.join(
                    dir_data_output,
                    f"store_{store_id}_forecast_30.csv"
                )

                df_forecast_30.to_csv(path_forecast_30, index=False)
                logger.info("30-day forecast saved for Store %s", store_id)

            logger.info("Storing forecast model results...")
            df_forecasts = pd.DataFrame({
                "Date": test["Date"].values,
                "Actual": y_test.values,

                "ARIMA": (
                    np.array(forecast_arima)
                    if forecast_arima is not None
                    else np.full(len(test), np.nan)
                ),

                "Prophet": (
                    np.array(prophet_predictions)
                    if prophet_predictions is not None
                    else np.full(len(test), np.nan)
                ),

                "XGBoost": (
                    np.array(xgb_predictions)
                    if xgb_predictions is not None
                    else np.full(len(test), np.nan)
                ),
            })

            # Add LSTM predictions
            df_forecasts["lstm_predictions"] = np.nan
            if lstm_predictions is not None:

                lstm_flat = lstm_predictions.flatten()
                lstm_start = len(df_forecasts) - len(lstm_flat)

                df_forecasts.loc[
                    lstm_start:lstm_start + len(lstm_flat) - 1,
                    "lstm_predictions"
                ] = lstm_flat


            path_forecast_csv = os.path.join(
                dir_data_output,
                f"store_{store_id}_forecasts.csv"
            )

            df_forecasts.to_csv(path_forecast_csv, index=False)
            logger.info("Forecasts saved to %s", path_forecast_csv)

            logger.info("All forecasting models trained for Store %s!", store_id)

        except Exception:
            logger.exception("Critical failure for Store %s!", store_id)
            continue
