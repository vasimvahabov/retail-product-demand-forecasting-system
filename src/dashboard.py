import os
import streamlit as st
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def stop_application():
    logger.info("Stopping Streamlit application...")
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


def launch(stores_to_process, dir_data_output, dir_figures, dashboard_artifacts):
    st.title("Retail Sales Forecasting & Inventory Optimization Dashboard")

    st.write("Explore sales trends, forecasting results, Model Evaluation, and inventory recommendations.")

    store_options = [f"Store {i}" for i in stores_to_process]
    store_selected = st.sidebar.selectbox("Select Store", store_options)
    store_id = int(store_selected.replace("Store ", ""))

    csvs= {}
    for csv_file in dashboard_artifacts.get("data", []):
        path = os.path.join(dir_data_output, csv_file.format(store=store_id))
        df = load_csv(path, csv_file)
        if df is not None:
            key = os.path.splitext(os.path.basename(csv_file))[0]
            if key.startswith("store_") and key.count("_") >= 2:
                key = "_".join(key.split("_")[2:])  # keep everything after store id
            csvs[key] = df

    figures = {}
    for fig_file in dashboard_artifacts.get("figures", []):
        path = os.path.join(dir_figures, fig_file.format(store=store_id))
        key = os.path.splitext(os.path.basename(fig_file))[0]
        if key.startswith("store_") and key.count("_") >= 2:
            key = "_".join(key.split("_")[2:])  # remove store id
        figures[key] = path

    section = st.sidebar.radio(
        "Select Section",
        ["Sales Overview", "Forecast", "Model Evaluation", "Inventory Recommendation"]
    )


    match section:
        case "Sales Overview":
            st.header("Sales History")
            st.subheader("Daily Sales")
            show_figure(figures.get("daily_sales"), "Daily Sales")

            st.subheader("Monthly Sales Trend")
            show_figure(figures.get("monthly_sales"), "Monthly Sales")

            st.subheader("Sales Peaks")
            show_figure(figures.get("sales_peaks"), "Sales Peaks")

            st.subheader("Weekday Pattern")
            show_figure(figures.get("weekday_pattern"), "Weekday Pattern")

            st.subheader("Overall Sales Distribution")
            show_figure(figures.get("sales_distribution"), "Sales Distribution")

        case "Forecast":
            st.header("Forecasting Results")

            st.subheader("Forecast Data")
            st.dataframe(csvs.get("forecasts"))

            st.subheader("Prophet Forecast")
            show_figure(figures.get("prophet_forecast"), "Prophet Forecast")

            st.subheader("LSTM Forecast")
            show_figure(figures.get("lstm_forecast"), "LSTM Forecast")

            with st.expander("Show ARIMA Forecast"):
                show_figure(figures.get("arima_forecast"), "ARIMA Forecast")

            with st.expander("Show 30-Day Forecast Data"):
                st.dataframe(csvs.get("forecast_30"))

            with st.expander("Show Forecast Metrics"):
                st.table(csvs.get("forecast_metrics"))

            with st.expander("Show EDA Data"):
                st.dataframe(csvs.get("eda"))

        case "Model Evaluation":
            st.header("Model Evaluation")
            st.dataframe(csvs.get("evaluation"))

            st.subheader("Visual Comparison")
            show_figure(figures.get("model_comparison"), "Model Evaluation")

        case "Inventory Recommendation":
            st.header("Inventory Optimization")
            st.dataframe(csvs.get("inventory"))

    st.sidebar.info("Retail Forecasting Project Dashboard")
