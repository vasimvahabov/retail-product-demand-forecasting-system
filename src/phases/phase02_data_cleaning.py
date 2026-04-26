"""
Loads, merges, and cleans retail sales and store data for forecasting.

Performs basic preprocessing, removes invalid records, handles missing values, and generates initial exploratory plots.
"""

import os
import logging
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def run(dir_figures, path_train_input, path_store_input):
    """
    Loads, cleans, and analyzes the dataset.

    Args:
        dir_figures (str): Path to save figures.
        path_train_input (str): Path to the training dataset.
        path_store_input (str): Path to the store dataset.

    Returns:
        pd.DataFrame: Cleaned and processed DataFrame.
    """

    df = None
    logger.info("Loading dataset...")
    try:
        train = pd.read_csv(path_train_input, low_memory=False)
        store = pd.read_csv(path_store_input, low_memory=False)
        df = pd.merge(train, store, on="Store", how="left")
        logger.info("Dataset successfully loaded!")
    except Exception:
        logger.exception("Exception occurred on dataset loading!")

    logger.info("Dataset Shape: %s rows, %s columns!", df.shape[0], df.shape[1])
    logger.info("Column Names: %s!", df.columns.tolist())

    logger.info("Dataset info:%s!", df.dtypes)
    logger.info("First 5 rows:%s!", df.head())
    desc = df.describe().round(2)
    logger.info("Descriptive Statistics (summary):%s!", desc.to_string())
    logger.info("Duplicate row count: %s!", df.duplicated().sum())

    if "Date" in df.columns:
        logger.info("Converting 'Date' column to datetime...")
        try:
            df["Date"] = pd.to_datetime(df["Date"])
            logger.info("'Date' column converted to datetime!")
        except Exception:
            logger.exception("Failed to convert 'Date' column!")

    logger.info("Creating quick sales distribution plot...")
    if "Sales" in df.columns:
        try:
            plt.figure(figsize=(8, 5))
            plt.hist(df["Sales"], bins=50)
            plt.title("Sales Distribution")
            plt.xlabel("Sales")
            plt.ylabel("Frequency")

            path_plot = os.path.join(dir_figures, "sales_distribution.png")
            plt.savefig(path_plot, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Quick sales distribution plot created!")
        except Exception:
            logger.exception("Failed to create sales distribution plot!")

    closed_stores = df[df["Open"] == 0].shape[0]
    open_stores = df[df["Open"] == 1].shape[0]
    logger.info("Closed stores: %s!", closed_stores)
    logger.info("Open stores: %s!", open_stores)

    logger.info("Removing rows where store is closed...")
    try:
        df = df[df["Open"] == 1]
        logger.info("Rows where the store was closed removed!")
        logger.info("Remaining rows after removing closed stores: %s", df.shape[0])
    except Exception:
        logger.exception("Failed to remove closed stores!")

    missing = df.isnull().sum()
    logger.info("Missing values per column: %s", missing)

    logger.info("Filling missing values using forward fill...")
    try:
        df = df.ffill()
        logger.info("Missing values filled!")
    except Exception:
        logger.exception("Failed to fill missing values!")

    return df
