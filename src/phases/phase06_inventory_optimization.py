# Import libraries
import logging
import pandas as pd

def run(forecasts, dir_data_output):
    """
    Optimizes inventory levels for each store based on 30-day demand forecasts.

    Args:
        forecasts (dict): Dictionary of forecasts for each store.
        dir_data_output (str): Path to save inventory recommendations.

    Returns:
        None
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    for store_id, store_forecast in forecasts.items():
        logging.info(f"\nStarting inventory optimization for Store {store_id}...")

        try:
            # Extract and rename forecasted demand for the next 30 days
            logging.info("Extracting 30-day demand forecast...")
            forecast_30 = store_forecast['forecast_30']
            future_demand = forecast_30.tail(30)[["ds", "yhat"]].rename(columns={
                "ds": "Date",
                "yhat": "Forecast_Demand"
            })
            logging.info("\nNext 30 Days Demand Forecast:")
            logging.info(f"\n{future_demand.head()}")

            # Calculate average daily demand
            avg_daily_demand = future_demand["Forecast_Demand"].mean()
            logging.info(f"\nAverage Daily Demand: {avg_daily_demand:.2f}")

            # Reorder point calculation
            lead_time_days = 7
            reorder_point = avg_daily_demand * lead_time_days
            logging.info(f"Reorder Point: {reorder_point:.2f}")

            # Recommended stock level
            safety_stock = avg_daily_demand * 3
            recommended_stock = future_demand["Forecast_Demand"].sum() + safety_stock
            logging.info(f"Recommended Stock Level: {recommended_stock:.2f}")

            # Save results
            inventory_output = pd.DataFrame({
                "Product": [f"Store_{store_id}_Product"],
                "Forecast Demand (30 Days)": [future_demand["Forecast_Demand"].sum()],
                "Recommended Stock": [recommended_stock]
            })
            logging.info("Writing inventory recommendation to disk...")
            path_inventory = f"{dir_data_output}store_{store_id}_inventory_recommendation.csv"
            inventory_output.to_csv(path_inventory, index=False)
            logging.info(f"Inventory recommendation written to: {path_inventory}")

            logging.info("\nInventory Recommendation:")
            logging.info(f"\n{inventory_output}")

        except Exception as e:
            logging.error(f"Failed to optimize inventory for Store {store_id}: {e}", exc_info=True)
            raise
