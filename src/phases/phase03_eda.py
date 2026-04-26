"""
Processes cleaned dataset per store, engineers features, and generates visualizations.

Returns store-level DataFrames and saves processed outputs.
"""

import os
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def run(df_cleaned, dir_data_output, dir_figures, stores_to_process):
    """
    Processes cleaned data for selected stores, generates features, and visualizes trends.

    Args:
        df_cleaned (pd.DataFrame): Cleaned dataset.
        dir_data_output (str): Path to save cleaned store datasets.
        dir_figures (str): Path to save visualizations.
        stores_to_process (list[int]): List of store IDs to analyze.

    Returns:
        dict: Dictionary of cleaned DataFrames for each store.
    """

    df_cleaned_stores = {}

    logger.info("Selecting stores to analyze: %s...", stores_to_process)
    try:
        df_multi = df_cleaned[df_cleaned["Store"].isin(stores_to_process)]
        logger.info("Stores selected: %s!", df_multi["Store"].unique().tolist())
    except Exception:
        logger.exception("Failed to select stores!")

    for store_id in df_multi["Store"].unique():
        logger.info("Processing Store %s...", store_id)
        try:
            df_store = df_multi[df_multi["Store"] == store_id].copy()
            logger.info("Dataset shape for Store %s: %s!", store_id, df_store.shape)

            logger.info("Sorting %s rows by Date...", df_store.shape[0])
            df_store.sort_values(by="Date", ascending=True, inplace=True)
            logger.info("%s rows sorted!", df_store.shape[0])

            logger.info("Cleaned dataset preview for Store %s:%s!", store_id, df_store.head())

            path_store_output = os.path.join(dir_data_output, f"store_{store_id}_cleaned.csv")
            df_store.to_csv(path_store_output, index=False)
            df_cleaned_stores[store_id] = df_store
            logger.info("Cleaned dataset written to: %s!", path_store_output)

            # Create time-based features
            logger.info("Creating time-based features...")
            df_store["Year"] = df_store["Date"].dt.year
            df_store["Month"] = df_store["Date"].dt.month
            df_store["Weekday"] = df_store["Date"].dt.day_name()
            logger.info(
                "Preview of new date features:%s",
                df_store[["Date", "Year", "Month", "Weekday"]].head()
            )

            df_store["IsPromo"] = df_store["Promo"].astype(int)
            df_store["IsStateHoliday"] = (df_store["StateHoliday"] != "0").astype(int)
            df_store["IsSchoolHoliday"] = df_store["SchoolHoliday"].astype(int)

            # Create competition features
            df_store["CompetitionOpen"] = (df_store["CompetitionOpenSinceYear"] > 0).astype(int)
            df_store["CompetitionDistance"] = df_store["CompetitionDistance"].fillna(0)

            # Create additional time-based features
            df_store["DayOfYear"] = df_store["Date"].dt.dayofyear
            df_store["IsWeekend"] = (df_store["DayOfWeek"] >= 6).astype(int)

            # Visualize daily sales
            logger.info("Plotting daily sales...")
            plt.figure(figsize=(12, 6))
            plt.plot(df_store["Date"], df_store["Sales"])
            plt.title(f"Store {store_id}: Daily Sales Over Time")
            plt.xlabel("Date")
            plt.ylabel("Sales")
            plt.tight_layout()
            plot_path = os.path.join(dir_figures, f"store_{store_id}_daily_sales.png")
            plt.savefig(plot_path, dpi=300)
            plt.close()
            logger.info("Daily sales plot saved!")

            # Visualize monthly sales trend
            logger.info("Plotting monthly sales trend...")
            monthly_sales = df_store.groupby("Month")["Sales"].mean()
            plt.figure(figsize=(10, 5))
            plt.plot(monthly_sales.index, monthly_sales.values, marker="o")
            plt.title(f"Store {store_id}: Average Monthly Sales")
            plt.xlabel("Month")
            plt.ylabel("Average Sales")
            plt.tight_layout()

            plot_path = os.path.join(dir_figures, f"store_{store_id}_monthly_sales.png")
            plt.savefig(plot_path, dpi=300)
            plt.close()
            logger.info("Monthly sales trend plot saved!")

            # Visualize weekly seasonality
            logger.info("Plotting weekly seasonality...")
            weekday_sales = df_store.groupby("Weekday")["Sales"].mean()
            weekday_order = [
                "Monday", "Tuesday", "Wednesday",
                "Thursday", "Friday", "Saturday", "Sunday"
            ]
            weekday_sales = weekday_sales.reindex(weekday_order)
            plt.figure(figsize=(10, 5))
            plt.bar(weekday_sales.index, weekday_sales.values)
            plt.title(f"Store {store_id}: Average Sales by Weekday")
            plt.xlabel("Day of Week")
            plt.ylabel("Average Sales")
            plt.xticks(rotation=45)
            plt.tight_layout()

            plot_path = os.path.join(dir_figures, f"store_{store_id}_weekday_pattern.png")
            plt.savefig(plot_path, dpi=300)
            plt.close()
            logger.info("Weekly seasonality plot saved!")

            # Visualize sales peaks
            logger.info("Detecting and plotting sales peaks...")
            threshold = df_store["Sales"].quantile(0.95)
            peaks = df_store[df_store["Sales"] >= threshold]
            plt.figure(figsize=(12, 6))
            plt.plot(df_store["Date"], df_store["Sales"], label="Sales")
            plt.scatter(peaks["Date"], peaks["Sales"], color="red", label="Peaks")
            plt.title(f"Store {store_id}: Sales Peaks Detection")
            plt.xlabel("Date")
            plt.ylabel("Sales")
            plt.legend()
            plt.tight_layout()
            plot_path = os.path.join(dir_figures, f"store_{store_id}_sales_peaks.png")
            plt.savefig(plot_path, dpi=300)
            plt.close()
            logger.info("Sales peaks plot saved!")

        except Exception:
            logger.exception("Failed to process Store %s!", store_id)
            continue

    return df_cleaned_stores
