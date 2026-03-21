# Import libraries
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def run(forecasts, dir_data_output, dir_figures):
    """
    Evaluates and compares forecasting models for each store, then saves the results.

    Args:
        forecasts (dict): Dictionary of forecasts for each store.
        dir_data_output (str): Path to save comparison results.
        dir_figures (str): Path to save visualizations.

    Returns:
        None
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    for store_id, data in forecasts.items():
        logging.info(f"\nEvaluating models for Store {store_id}...")

        try:
            forecast = data["forecast"]
            xgb_predictions = data["xgb_predictions"]
            prophet_predictions = data["prophet_predictions"]
            y_test = data["y_test"]
            test_sales = data["test_sales"]
            lstm_rmse = data["lstm_rmse"]
            lstm_mape = data["lstm_mape"]

            # Define MAPE with zero-safe handling
            def mape(actual, predicted):
                actual, predicted = np.array(actual), np.array(predicted)
                return np.mean(np.abs((actual - predicted) / (actual + 1e-10))) * 100

            # Calculate performance metrics
            logging.info("Calculating performance metrics...")
            arima_rmse = np.sqrt(mean_squared_error(test_sales, forecast))
            xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_predictions))
            prophet_rmse = np.sqrt(mean_squared_error(test_sales, prophet_predictions))

            arima_mape = mape(test_sales, forecast)
            prophet_mape = mape(test_sales, prophet_predictions)
            xgb_mape = mape(y_test, xgb_predictions)

            results = pd.DataFrame({
                "Model": ["ARIMA", "Prophet", "XGBoost", "LSTM"],
                "RMSE": [arima_rmse, prophet_rmse, xgb_rmse, lstm_rmse],
                "MAPE": [arima_mape, prophet_mape, xgb_mape, lstm_mape]
            })

            logging.info("\nModel Performance Comparison:")
            logging.info(f"\n{results}")

            # Plot and save RMSE comparison
            logging.info("Plotting RMSE comparison...")
            plt.figure(figsize=(8, 5))
            plt.bar(results["Model"], results["RMSE"], color='skyblue')
            plt.title(f"Forecasting Model Comparison — RMSE (Store {store_id})")
            plt.xlabel("Model")
            plt.ylabel("RMSE")
            plt.tight_layout()
            plt.savefig(f"{dir_figures}store_{store_id}_model_comparison.png", dpi=300)
            plt.close()
            logging.info("RMSE comparison plot saved!")

            # Find the best model
            best_model = results.loc[results["RMSE"].idxmin()]
            logging.info("\nBest Model Based on RMSE:")
            logging.info(f"\n{best_model}")

            # Save comparison results
            logging.info("\nWriting comparison results to disk...")
            path_comparison_result = f"{dir_data_output}store_{store_id}_model_comparison.csv"
            results.to_csv(path_comparison_result, index=False)
            logging.info(f"Model comparison results written to: {path_comparison_result}")

            logging.info("All models evaluated, compared, and results persisted for Store {store_id}!\n")

        except Exception as e:
            logging.error(f"Failed to evaluate models for Store {store_id}: {e}", exc_info=True)
            raise
