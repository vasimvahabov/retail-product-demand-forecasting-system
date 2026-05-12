"""
Compares forecasting models (ARIMA, Prophet, XGBoost, LSTM) per store using RMSE and MAPE.

Generates comparison tables, selects best model, and persists plots and CSV outputs.
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)


def run(dir_data_output, dir_figures, stores_to_process):
    """
    Evaluates and compares forecasting models for each store, then persists the results.

    Args:
        dir_data_output (str): Path to save CSV results.
        dir_figures (str): Path to save plots.
        stores_to_process (list[int]): List of store IDs to process.

    Returns:
        None
    """

    df_forecasts = {
        store_id: pd.read_csv(
            os.path.join(dir_data_output, f"store_{store_id}_forecasts.csv")
        )
        for store_id in stores_to_process
    }

    # Ensure figures directory exists
    os.makedirs(dir_figures, exist_ok=True)
    for store_id, data in df_forecasts.items():
        logger.info("Evaluating models for Store %s...", store_id)

        try:
            forecast_arima = data.get("ARIMA")
            xgb_predictions = data.get("XGBoost")
            prophet_predictions = data.get("Prophet")
            y_test = data.get("Actual")
            lstm_predictions = data.get("lstm_predictions")

            # Define MAPE with zero-safe handling
            def mape(actual, predicted):
                if actual is None or predicted is None:
                    return None
                actual = np.asarray(actual)
                predicted = np.asarray(predicted)
                return np.mean(np.abs((actual - predicted) / (actual + 1e-10))) * 100

            if lstm_predictions is not None and y_test is not None:
                valid_lstm = ~pd.isna(lstm_predictions)

                lstm_rmse = np.sqrt(
                    mean_squared_error(
                        y_test[valid_lstm],
                        lstm_predictions[valid_lstm]
                    )
                )

                lstm_mape = mape(
                    y_test[valid_lstm],
                    lstm_predictions[valid_lstm]
                )
            else:
                lstm_rmse = None
                lstm_mape = None

            # Calculate performance metrics
            logger.info("Calculating performance metrics...")
            if forecast_arima is not None and y_test is not None:
                arima_rmse = np.sqrt(mean_squared_error(y_test, forecast_arima))
            else:
                arima_rmse = None

            if xgb_predictions is not None and y_test is not None:
                xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_predictions))
            else:
                xgb_rmse = None

            if prophet_predictions is not None and y_test is not None:
                prophet_rmse = np.sqrt(mean_squared_error(y_test, prophet_predictions))
            else:
                prophet_rmse = None

            arima_mape = mape(y_test, forecast_arima) if forecast_arima is not None else None
            prophet_mape = mape(y_test, prophet_predictions) if prophet_predictions is not None else None
            xgb_mape = mape(y_test, xgb_predictions) if xgb_predictions is not None else None

            results = pd.DataFrame({
                "Model": ["ARIMA", "Prophet", "XGBoost", "LSTM"],
                "RMSE": [arima_rmse, prophet_rmse, xgb_rmse, lstm_rmse],
                "MAPE": [arima_mape, prophet_mape, xgb_mape, lstm_mape]
            })
            logger.info("Model Performance Comparison:\n%s", results)

            # Plot and save RMSE comparison
            plot_df = results.dropna(subset=["RMSE"])
            if plot_df.empty:
                logger.warning("No RMSE values to plot for Store %s", store_id)
            else:
                logger.info("Plotting RMSE comparison...")
                plt.figure(figsize=(8, 5))

                plt.bar(plot_df["Model"], plot_df["RMSE"], color='skyblue')
                plt.title(f"Forecasting Model Comparison — RMSE (Store {store_id})")
                plt.xlabel("Model")
                plt.ylabel("RMSE")
                plt.tight_layout()
                path_mode_comparison = os.path.join(
                    dir_figures,
                    f"store_{store_id}_model_comparison.png"
                )
                plt.savefig(path_mode_comparison, dpi=300)
                plt.close()
                logger.info("RMSE comparison plot saved!")

            # Find the best model
            if results["RMSE"].dropna().empty:
                logger.warning("No valid RMSE values available!")
            else:
                best_model = results.loc[results["RMSE"].idxmin()]
                logger.info("Best Model is %s (RMSE=%s)!", best_model["Model"], best_model["RMSE"])

            # Save comparison results
            logger.info("Writing comparison results to disk...")
            path_comparison_result = os.path.join(
                dir_data_output,
                f"store_{store_id}_evaluation.csv"
            )
            results.to_csv(path_comparison_result, index=False)
            logger.info("Model comparison results written to %s! for Store %s", path_comparison_result, store_id)

            logger.info("All models evaluated, compared, and results persisted for Store %s!", store_id)

        except Exception:
            logger.exception("Failed to evaluate models for Store %s!", store_id)
            continue
