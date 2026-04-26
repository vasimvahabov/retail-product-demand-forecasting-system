"""
Retail Product Demand Forecasting Sales Forecasting Dashboard (Streamlit Application)

Runs ML pipeline, loads outputs, and displays store-level analytics.
"""

import os
import streamlit as st
import pandas as pd
import logging
import pipeline
from datetime import datetime

workdir = os.getcwd()

dir_log = os.path.join(workdir, "log")
dir_input = os.path.join(workdir, "data")
dir_output = os.path.join(workdir, "out")
dir_figures = os.path.join(dir_output, "figures")
dir_data_output = os.path.join(dir_output, "data")
path_train_input = os.path.join(dir_input, "train.csv")
path_store_input = os.path.join(dir_input, "store.csv")

os.makedirs(dir_log, exist_ok=True)
run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
path_log = os.path.join(dir_log, f"{run_id}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(path_log),
        logging.StreamHandler()
    ],
    force=True
)
logger = logging.getLogger(__name__)

stores_to_process = [
    1,
    # 3,
    # 8,
    # 15,
    # 25
]


@st.cache_data(show_spinner=True)
def run_pipeline():
    """
    Runs ML pipeline.

    Returns:
        bool: True if successful, False otherwise.
    """

    try:
        logger.info("Starting pipeline execution...")
        pipeline.run(
            path_train_input,
            path_store_input,
            dir_output,
            dir_data_output,
            dir_figures,
            stores_to_process
        )
        logger.info("Pipeline execution completed!")
        return True

    except FileNotFoundError:
        logger.exception("Required file missing on pipeline execution!")
        st.error("Required file missing on pipeline execution!")
        return False

    except Exception:
        logger.exception("Pipeline execution failed!")
        st.error("Pipeline execution failed!")
        return False


if not run_pipeline():
    st.stop()

runtime_env = os.getenv("RUNTIME_ENV", "LOCAL")
logger.info("Runtime environment: %s", runtime_env)
if runtime_env != "LOCAL":
    logger.warning("Stopping app: runtime environment is not local (%s)", runtime_env)
    st.stop()

st.title("Retail Sales Forecasting & Inventory Optimization Dashboard")
st.write("Explore sales trends, forecasting results, model comparison, and inventory recommendations.")

store_options = []
for store_id in stores_to_process:
    store_options.append(f"Store {store_id}")
store_selected = st.sidebar.selectbox("Select Store", store_options)
store_id = 1 if store_selected == "Store 1" else 3

sales_file = os.path.join(dir_data_output, f"store_{store_id}_cleaned.csv")
model_file = os.path.join(dir_data_output, f"store_{store_id}_model_comparison.csv")
inventory_file = os.path.join(dir_data_output, f"store_{store_id}_inventory_recommendation.csv")

figures = {
    "daily_sales": os.path.join(dir_figures, f"store_{store_id}_daily_sales.png"),
    "lstm_forecast": os.path.join(dir_figures, f"store_{store_id}_lstm_forecast.png"),
    "monthly_sales": os.path.join(dir_figures, f"store_{store_id}_monthly_sales.png"),
    "prophet_forecast": os.path.join(dir_figures, f"store_{store_id}_prophet_forecast.png"),
    "sales_peaks": os.path.join(dir_figures, f"store_{store_id}_sales_peaks.png"),
    "weekday_pattern": os.path.join(dir_figures, f"store_{store_id}_weekday_pattern.png"),
    "model_comparison": os.path.join(dir_figures, f"store_{store_id}_model_comparison.png"),
}

sales_distribution_fig = os.path.join(dir_figures, "sales_distribution.png")


def load_csv(path, name):
    """
    Loads a CSV file safely with logging.

    Args:
        path (str): file path
        name (str): dataset label for logs

    Returns:
        pd.DataFrame or None if loading fails
    """

    logger.info("Loading %s from %s...", name, path)
    try:
        df = pd.read_csv(path)
        logger.info("%s loaded | rows=%s!", name, len(df))
        return df

    except FileNotFoundError:
        logger.exception("%s file not found at %s!", name, path)
        return None

    except pd.errors.EmptyDataError:
        logger.exception("%s file is empty!", name)
        return None

    except Exception:
        logger.exception("Failed to load %s from %s!", name, path)
        return None


sales_data = load_csv(sales_file, "Sales Data")
if sales_data is None:
    st.error("Failed to load Sales Data!")
    st.stop()

if "Date" in sales_data.columns:
    sales_data["Date"] = pd.to_datetime(sales_data["Date"])
else:
    st.error("Date column missing in Sales Data!")
    st.stop()

model_results = load_csv(model_file, "Model Results")
if model_results is None or model_results.empty:
    st.error("Model results missing or empty!")
    st.stop()

inventory = load_csv(inventory_file, "Inventory Data")
if inventory is None or inventory.empty:
    st.error("Inventory data missing or empty!")
    st.stop()

if sales_data.empty:
    logger.warning("Sales data is empty!")
    st.warning("No sales data available!")
    st.stop()

required_sales_cols = ["Date", "Sales"]
for col in required_sales_cols:
    if col not in sales_data.columns:
        logger.exception("Missing column: %s!", col)
        st.error(f"Missing column: {col}!")
        st.stop()

section = st.sidebar.radio(
    "Select Section",
    ["Sales Overview", "Forecast", "Model Comparison", "Inventory Recommendation"]
)


def show_figure(path, name):
    """
    Displays an image in Streamlit if the file exists.

    Args:
        path (str): image file path
        name (str): label for logging and UI
    """

    if os.path.exists(path):
        logger.info("Displaying figure %s...", name)
        st.image(path)
    else:
        logger.warning("%s not found at %s!", name, path)
        st.warning(f"{name} not found at {path}")


if section == "Sales Overview":
    st.header("Sales History")
    st.subheader("Daily Sales")
    show_figure(figures["daily_sales"], "Daily Sales")
    st.subheader("Monthly Sales Trend")

    show_figure(figures["monthly_sales"], "Monthly Sales")

    st.subheader("Sales Peaks & Seasonality")
    show_figure(figures["sales_peaks"], "Sales Peaks")
    show_figure(figures["weekday_pattern"], "Weekday Pattern")

    st.subheader("Overall Sales Distribution")
    show_figure(sales_distribution_fig, "Sales Distribution")

elif section == "Forecast":
    st.header("Forecasting Results")
    st.subheader("Prophet Forecast")
    show_figure(figures["prophet_forecast"], "Prophet Forecast")

    st.subheader("LSTM Forecast")
    show_figure(figures["lstm_forecast"], "LSTM Forecast")

elif section == "Model Comparison":
    st.header("Model Performance Comparison")
    st.dataframe(model_results)

    st.subheader("Visual Comparison")
    show_figure(figures["model_comparison"], "Model Comparison")

elif section == "Inventory Recommendation":
    st.header("Inventory Optimization")
    st.dataframe(inventory)

st.sidebar.info("Retail Forecasting Project Dashboard")
