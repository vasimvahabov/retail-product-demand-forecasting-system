"""
Generates and persists inventory recommendations per store using 30-day demand forecasts.

Computes reorder point, safety stock, and recommended stock, then persists results.
"""


import os
import logging
import pandas as pd

logger = logging.getLogger(__name__)

def run(dir_data_output, stores_to_process):
    """
    Creates and persists inventory recommendations for each store based on 30-day demand forecasts.

    Args:
        dir_data_output (str): Path to save inventory recommendations.
        stores_to_process (list[int]): List of store IDs to process.

    Returns:
        None
    """

    for store_id in stores_to_process:
        logger.info("Starting inventory optimization for Store %s...", store_id)

        try:
            # Load 30-day forecast
            logger.info("Extracting 30-day demand forecast...")

            path_forecast_30 = os.path.join(
                dir_data_output,
                f"store_{store_id}_forecast_30.csv"
            )

            if not os.path.exists(path_forecast_30):
                logger.warning("Store %s missing forecast_30 CSV!", store_id)
                continue

            future_demand = pd.read_csv(path_forecast_30)

            future_demand = future_demand.rename(columns={
                "Forecast": "Forecast_Demand"
            })
            # Calculate average daily demand
            future_demand["Forecast_Demand"] = pd.to_numeric(
                future_demand["Forecast_Demand"],
                errors="coerce"
            )
            avg_daily_demand = future_demand["Forecast_Demand"].mean()
            if pd.isna(avg_daily_demand):
                logger.warning("Store %s has invalid forecast values!", store_id)
                continue
            logger.info("Average Daily Demand: %.2f", avg_daily_demand)

            # Reorder point calculation
            lead_time_days = 7
            reorder_point = avg_daily_demand * lead_time_days
            logger.info("Reorder Point: %.2f", reorder_point)

            # Recommended stock level
            safety_stock = avg_daily_demand * 3
            recommended_stock = future_demand["Forecast_Demand"].sum() + safety_stock
            logger.info("Recommended Stock Level: %.2f", recommended_stock)

            # Save results
            inventory_output = pd.DataFrame({
                "store_id": [store_id],
                "forecast_30": [future_demand["Forecast_Demand"].sum()],
                "recommended_stock": [recommended_stock],
                "reorder_point": [reorder_point],
                "safety_stock": [safety_stock]
            })
            logger.info("Writing inventory recommendation to disk...")
            path_inventory = os.path.join(
                dir_data_output,
                f"store_{store_id}_inventory.csv"
            )
            inventory_output.to_csv(path_inventory, index=False)
            logger.info("Inventory recommendation written to %s", path_inventory)

            logger.info("Inventory Recommendation:\n%s", inventory_output)

        except Exception:
            logger.exception("Failed to optimize inventory for Store %s", store_id)
            continue