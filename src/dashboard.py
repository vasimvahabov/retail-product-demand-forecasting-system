import os
import streamlit as st
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def stop_application():
    logger.info("Stopping Streamlit...")
    st.stop()

@st.cache_data
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
        df = pd.read_csv(
            path,
            low_memory=False
        )
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


def launch(stores_to_process, dir_data_output, dir_figures):
    st.title("Retail Sales Forecasting & Inventory Optimization Dashboard")

    st.write("Explore sales trends, forecasting results, model comparison, and inventory recommendations.")

    store_options = [f"Store {i}" for i in stores_to_process]
    store_selected = st.sidebar.selectbox("Select Store", store_options)
    store_id = int(store_selected.replace("Store ", ""))

    sales_file = os.path.join(
        dir_data_output,
        f"sales_data_cleaned.csv"
    )
    model_file = os.path.join(
        dir_data_output,
        f"store_{store_id}_evaluation.csv"
    )
    forecast_file = os.path.join(
        dir_data_output,
        f"store_{store_id}_forecasts.csv"
    )
    inventory_file = os.path.join(
        dir_data_output,
        f"store_{store_id}_inventory.csv"
    )

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

    sales_data = load_csv(sales_file, "Sales Data")
    if sales_data is None:
        st.error("Failed to load Sales Data!")
        stop_application()

    sales_data = sales_data[sales_data["Store"] == store_id].copy()

    if "Date" in sales_data.columns:
        sales_data["Date"] = pd.to_datetime(sales_data["Date"])
    else:
        st.error("Date column missing in Sales Data!")
        stop_application()

    model_results = load_csv(model_file, "Model Results")
    if model_results is None or model_results.empty:
        st.error("Model results missing or empty!")
        stop_application()

    forecast_data = load_csv(forecast_file, "Forecast Data")
    if forecast_data is None or forecast_data.empty:
        st.error("Forecast Data missing or empty!")
        stop_application()

    inventory = load_csv(inventory_file, "Inventory Data")
    if inventory is None or inventory.empty:
        st.error("Inventory data missing or empty!")
        stop_application()

    if sales_data.empty:
        logger.warning("Sales data is empty!")
        st.warning("No sales data available!")
        stop_application()

    required_sales_cols = ["Date", "Sales"]
    for col in required_sales_cols:
        if col not in sales_data.columns:
            logger.error("Missing column: %s!", col)
            st.error(f"Missing column: {col}!")
            stop_application()

    section = st.sidebar.radio(
        "Select Section",
        ["Sales Overview", "Forecast", "Model Comparison", "Inventory Recommendation"]
    )

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

        st.subheader("Forecast Data")
        st.dataframe(forecast_data)

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
