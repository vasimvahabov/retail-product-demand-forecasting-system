# Import libraries
import logging
import pandas as pd
import matplotlib.pyplot as plt


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

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Load dataset
    logging.info("Loading dataset...")
    try:
        train = pd.read_csv(path_train_input, low_memory=False)
        store = pd.read_csv(path_store_input, low_memory=False)
        df = pd.merge(train, store, on="Store", how="left")
        logging.info("Dataset successfully loaded!")
    except Exception as e:
        logging.error(f"Exception occurred on dataset loading: {e}", exc_info=True)
        raise

    # Log dataset overview
    logging.info(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    logging.info(f"Column Names: {df.columns.tolist()}")

    # Log dataset info
    logging.info("\nDataset Info:")
    logging.info(f"\n{df.info()}")

    # Log first 5 rows
    logging.info("\nFirst 5 Rows:")
    logging.info(f"\n{df.head()}")

    # Log descriptive statistics
    logging.info("\nDescriptive Statistics:")
    logging.info(f"\n{df.describe()}")

    # Log unique values per column
    logging.info("\nUnique Values Per Column:")
    for col in df.columns:
        logging.info(f"{col}: {df[col].nunique()}")

    # Log duplicate rows
    duplicates = df.duplicated().sum()
    logging.info(f"\nDuplicate row count: {duplicates}")

    # Convert Date column if exists
    if "Date" in df.columns:
        logging.info("Converting 'Date' column to datetime...")
        try:
            df["Date"] = pd.to_datetime(df["Date"])
            logging.info("'Date' column converted to datetime!")
        except Exception as e:
            logging.error(f"Failed to convert 'Date' column: {e}", exc_info=True)
            raise

    # Visualize sales distribution
    logging.info("\nCreating quick sales distribution plot...")
    try:
        plt.figure(figsize=(8, 5))
        plt.hist(df["Sales"], bins=50)
        plt.title("Sales Distribution")
        plt.xlabel("Sales")
        plt.ylabel("Frequency")
        plt.savefig(f"{dir_figures}sales_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        logging.info("Quick sales distribution plot created!")
    except Exception as e:
        logging.error(f"Failed to create sales distribution plot: {e}", exc_info=True)
        raise

    # Log closed and open stores
    closed_stores = df[df["Open"] == 0].shape[0]
    open_stores = df[df["Open"] == 1].shape[0]
    logging.info(f"\nClosed stores: {closed_stores}")
    logging.info(f"Open stores: {open_stores}")

    # Remove closed stores
    logging.info("Removing rows where store is closed...")
    try:
        df = df[df["Open"] == 1]
        logging.info("Rows where the store was closed removed!")
        logging.info(f"Remaining rows after removing closed stores: {df.shape[0]}")
    except Exception as e:
        logging.error(f"Failed to remove closed stores: {e}", exc_info=True)
        raise

    # Log missing values per column
    missing = df.isnull().sum()
    logging.info(f"\nMissing values per column:\n{missing}")

    # Fill missing values with forward fill
    logging.info("Filling missing values using forward fill...")
    try:
        df = df.ffill()
        logging.info("Missing values filled!")
    except Exception as e:
        logging.error(f"Failed to fill missing values: {e}", exc_info=True)
        raise

    return df
