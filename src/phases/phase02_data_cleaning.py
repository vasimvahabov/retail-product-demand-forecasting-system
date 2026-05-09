"""
Loads, merges, cleans, and persists retail sales and store data for forecasting.

Performs basic preprocessing, removes invalid records, handles missing values, and generates initial exploratory plots.
"""

import os
import logging
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def run(path_train_input, path_store_input, dir_data_output, dir_figures):
    """
    Loads, cleans, analyzes, and persists the dataset.

    Args:
        path_train_input (str): Path to the training dataset.
        path_store_input (str): Path to the store dataset.
        dir_data_output (str): Path to save cleaned store datasets.
        dir_figures (str): Path to save figures.

    Returns:
        None
    """

    if not os.path.exists(path_train_input):
        logger.error("Training Dataset input file not found at %s", path_train_input)
        return None

    if not os.path.exists(path_store_input):
        logger.error("Store Dataset input file not found at %s", path_store_input)
        return None

    logger.info("Loading dataset...")
    try:
        train = pd.read_csv(path_train_input, low_memory=False)
        store = pd.read_csv(path_store_input, low_memory=False)
        df = pd.merge(train, store, on="Store", how="left")
        logger.info("Dataset successfully loaded!")
    except Exception:
        logger.exception("Exception occurred on dataset loading!")
        return None

    logger.info("Dataset Shape: %s rows, %s columns!", df.shape[0], df.shape[1])
    logger.info("Column Names: %s!", df.columns.tolist())

    logger.info("Dataset info:%s!", df.dtypes)
    logger.info("First 5 rows:%s!", df.head())
    desc = df.describe().round(2)
    logger.info("Descriptive Statistics (summary):%s!", desc.to_string())

    if "Date" in df.columns:
        logger.info("Converting 'Date' column to datetime...")
        try:
            df["Date"] = pd.to_datetime(df["Date"])
            logger.info("'Date' column converted to datetime!")
        except Exception:
            logger.exception("Failed to convert 'Date' column!")

    duplicates = df.duplicated().sum()
    if duplicates > 0:
        logger.info("Removing %d duplicate rows...", duplicates)
        df = df.drop_duplicates()
        logger.info("%d duplicate rows removed...", duplicates)

    logger.info("Converting 'Date' column to datetime...")
    try:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        logger.info("'Date' column converted to datetime!")
    except Exception:
        logger.exception("Failed to convert 'Date' column!")
        return None

    logger.info("Creating quick sales distribution plot...")
    try:
        plt.figure(figsize=(8, 5))
        plt.hist(df["Sales"], bins=50)
        plt.title("Sales Distribution")
        plt.xlabel("Sales")
        plt.ylabel("Frequency")

        path_plot = os.path.join(dir_figures, "sales_distribution.png")
        os.makedirs(dir_figures, exist_ok=True)
        plt.savefig(path_plot, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Quick sales distribution plot created!")
    except Exception:
        logger.exception("Failed to create sales distribution plot!")
        return None

    try:
        logger.info("Removing rows where store is closed...")

        closed_stores = df[df["Open"] == 0].shape[0]
        open_stores = df[df["Open"] == 1].shape[0]

        logger.info("Closed stores: %s!", closed_stores)
        logger.info("Open stores: %s!", open_stores)

        df = df[df["Open"] == 1]
        logger.info("Rows where the store was closed removed!")
        logger.info("Remaining rows after removing closed stores: %s", df.shape[0])

    except Exception:
        logger.exception("Failed to remove closed stores!")
        return None

    missing = df.isnull().sum()
    logger.info("Missing values per column: %s", missing)

    logger.info("Filling missing values using forward fill...")
    try:
        missing_before = df.isnull().sum().sum()
        df = df.ffill()
        missing_after = df.isnull().sum().sum()
        logger.info("Missing values filled from %d to %d remaining!", missing_before, missing_after)
    except Exception:
        logger.exception("Failed to fill missing values!")
        return None

    logger.info("Cleaned DataFrame shape: %s!", df.shape)

    path_cleaned_df = os.path.join(dir_data_output, "sales_data_cleaned.csv")
    os.makedirs(dir_data_output, exist_ok=True)
    df.to_csv(path_cleaned_df, index=False)
    logger.info("Cleaned dataset written to %s", path_cleaned_df)

    return None
