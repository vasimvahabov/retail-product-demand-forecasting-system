# Import libraries

import logging
import matplotlib.pyplot as plt


def run(df_cleaned, dir_data_output, dir_figures, stores_to_use):
    """
    Processes cleaned data for selected stores, generates features, and visualizes trends.

    Args:
        df_cleaned (pd.DataFrame): Cleaned dataset.
        dir_data_output (str): Path to save cleaned store datasets.
        dir_figures (str): Path to save visualizations.
        stores_to_use (list): List of store IDs to analyze.

    Returns:
        dict: Dictionary of cleaned DataFrames for each store.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    df_cleaned_stores = {}

    # Select stores to analyze
    logging.info(f"Selecting stores to analyze: {stores_to_use}")
    try:
        df_multi = df_cleaned[df_cleaned["Store"].isin(stores_to_use)]
        logging.info(f"Stores selected: {df_multi["Store"].unique().tolist()}")
    except Exception as e:
        logging.error(f"Failed to select stores: {e}", exc_info=True)
        raise

    for store_id in df_multi["Store"].unique():
        logging.info(f"\nProcessing Store {store_id}...")
        try:
            df_store = df_multi[df_multi["Store"] == store_id].copy()
            logging.info(f"Dataset shape for Store {store_id}: {df_store.shape}")

            # Sort dataset by date
            logging.info(f"Sorting {df_store.shape[0]} rows by Date...")
            df_store.sort_values(by="Date", ascending=True, inplace=True)
            logging.info("Rows sorted!")

            # Log cleaned dataset preview
            logging.info(f"\nCleaned dataset preview for Store {store_id}:\n{df_store.head()}")

            # Save cleaned dataset
            path_store_output = f"{dir_data_output}store_{store_id}_cleaned.csv"
            df_store.to_csv(path_store_output, index=False)
            df_cleaned_stores[store_id] = df_store
            logging.info(f"Cleaned dataset written to: {path_store_output}")

            # Create time-based features
            logging.info("Creating time-based features...")
            df_store["Year"] = df_store["Date"].dt.year
            df_store["Month"] = df_store["Date"].dt.month
            df_store["Weekday"] = df_store["Date"].dt.day_name()
            logging.info("Preview of new date features:\n"
                         f"{df_store[["Date", "Year", "Month", "Weekday"]].head()}")

            # Create binary features
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
            logging.info("Plotting daily sales...")
            plt.figure(figsize=(12, 6))
            plt.plot(df_store["Date"], df_store["Sales"])
            plt.title(f"Store {store_id}: Daily Sales Over Time")
            plt.xlabel("Date")
            plt.ylabel("Sales")
            plt.tight_layout()
            plt.savefig(f"{dir_figures}store_{store_id}_daily_sales.png", dpi=300)
            plt.close()
            logging.info("Daily sales plot saved!")

            # Visualize monthly sales trend
            logging.info("Plotting monthly sales trend...")
            monthly_sales = df_store.groupby("Month")["Sales"].mean()
            plt.figure(figsize=(10, 5))
            plt.plot(monthly_sales.index, monthly_sales.values, marker="o")
            plt.title(f"Store {store_id}: Average Monthly Sales")
            plt.xlabel("Month")
            plt.ylabel("Average Sales")
            plt.tight_layout()
            plt.savefig(f"{dir_figures}store_{store_id}_monthly_sales.png", dpi=300)
            plt.close()
            logging.info("Monthly sales trend plot saved!")

            # Visualize weekly seasonality
            logging.info("Plotting weekly seasonality...")
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
            plt.savefig(f"{dir_figures}store_{store_id}_weekday_pattern.png", dpi=300)
            plt.close()
            logging.info("Weekly seasonality plot saved!")

            # Visualize sales peaks
            logging.info("Detecting and plotting sales peaks...")
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
            plt.savefig(f"{dir_figures}store_{store_id}_sales_peaks.png", dpi=300)
            plt.close()
            logging.info("Sales peaks plot saved!")

        except Exception as e:
            logging.error(f"Failed to process Store {store_id}: {e}", exc_info=True)
            raise

    return df_cleaned_stores
