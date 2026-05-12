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
    st.title("Retail Product Demand Forecasting Dashboard")

    st.write("Explore sales trends, forecasts, model evaluation, and inventory")

    store_options = [f"Store {i}" for i in stores_to_process]
    store_selected = st.sidebar.selectbox("Select Store", store_options)
    store_id = int(store_selected.replace("Store ", ""))

    csvs = {}
    for csv_file in dashboard_artifacts.get("data", []):
        path = os.path.join(dir_data_output, csv_file.format(store=store_id))
        df = load_csv(path, csv_file)
        if df is not None:
            key = os.path.splitext(os.path.basename(csv_file))[0]
            if key.startswith("store_") and key.count("_") >= 2:
                key = "_".join(key.split("_")[2:])
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
            st.header("Sales Overview")

            col_prophet, col_lstm = st.columns(2)

            with col_prophet:
                st.subheader("Daily Sales")
                show_figure(figures.get("daily_sales"), "Daily Sales")

            with col_lstm:
                st.subheader("Monthly Sales Trend")
                show_figure(figures.get("monthly_sales"), "Monthly Sales")

            col_arima, col_xgboost = st.columns(2)

            with col_arima:
                st.subheader("Sales Peaks")
                show_figure(figures.get("sales_peaks"), "Sales Peaks")

            with col_xgboost:
                st.subheader("Weekday Pattern")
                show_figure(figures.get("weekday_pattern"), "Weekday Pattern")

            st.subheader("Sales Distribution")
            show_figure(figures.get("sales_distribution"), "Sales Distribution")

        case "Forecast":
            st.header("Forecasting Results")

            tab_charts, tab_forecast_data = st.tabs([
                "Forecast Charts",
                "Forecast Data",
            ])

            # Forecast Charts Tab
            with tab_charts:

                col_prophet, col_lstm = st.columns(2)

                with col_prophet:
                    subheader_prophet = "Prophet Forecast"
                    st.subheader(subheader_prophet)
                    show_figure(figures.get("prophet_forecast"), subheader_prophet)

                with col_lstm:
                    subheader_lstm = "LSTM Forecast"
                    st.subheader(subheader_lstm)
                    show_figure(figures.get("lstm_forecast"), subheader_lstm)


                col_arima, col_xgboost = st.columns(2)
                with col_arima:
                    subheader_arima = "ARIMA Forecast"
                    st.subheader(subheader_arima)
                    show_figure(figures.get("arima_forecast"), subheader_arima)

                with col_xgboost:
                    subheader_xgboost = "XGBoost Forecast"
                    st.subheader(subheader_xgboost)
                    show_figure(figures.get("xgboost_forecast"), subheader_xgboost)

            # Forecast Data Tab
            with tab_forecast_data:

                st.subheader("Forecast Dataset")
                st.dataframe(csvs.get("forecasts"), width='stretch')

                st.subheader("30-Day Forecast")
                st.dataframe(csvs.get("forecast_30"), width='stretch')

                st.subheader("EDA Dataset")
                st.dataframe(csvs.get("eda"), width='stretch')

        case "Model Evaluation":

            st.header("Model Evaluation")

            evaluation_df = csvs.get("evaluation")

            if evaluation_df is not None and not evaluation_df.empty:

                best_model = evaluation_df.loc[
                    evaluation_df["RMSE"].idxmin()
                ]

                col_prophet, col_lstm, col_arima = st.columns(3)

                with col_prophet:
                    st.metric(
                        "Best Model",
                        best_model["Model"]
                    )

                with col_lstm:
                    st.metric(
                        "Lowest RMSE",
                        f"{best_model['RMSE']:,.2f}"
                    )

                with col_arima:
                    st.metric(
                        "Lowest MAPE",
                        f"{best_model['MAPE']:.2f}%"
                    )

                st.dataframe(
                    evaluation_df,
                    width='stretch'
                )

                col_prophet, col_lstm = st.columns(2)

                with col_prophet:
                    subheader_mape = "MAPE Comparison"
                    st.subheader(subheader_mape)

                    show_figure(
                        figures.get("model_comparison_mape"),
                        subheader_mape
                    )

                with col_lstm:
                    subheader_rmse = "RMSE Comparison"
                    st.subheader(subheader_rmse)

                    show_figure(
                        figures.get("model_comparison_rmse"),
                        subheader_rmse
                    )

            else:
                st.warning("Evaluation data not available.")

        case "Inventory Recommendation":

            st.header("Inventory Recommendation")

            inventory_df = csvs.get("inventory")

            if inventory_df is not None and not inventory_df.empty:

                row = inventory_df.iloc[0]

                col_prophet, col_lstm = st.columns(2)

                with col_prophet:
                    st.metric(
                        "30-Day Forecast",
                        f"{row['forecast_30']:,.0f}"
                    )

                    st.metric(
                        "Recommended Stock",
                        f"{row['recommended_stock']:,.0f}"
                    )

                with col_lstm:
                    st.metric(
                        "Reorder Point",
                        f"{row['reorder_point']:,.0f}"
                    )

                    st.metric(
                        "Safety Stock",
                        f"{row['safety_stock']:,.0f}"
                    )

                st.dataframe(
                    inventory_df,
                    width='stretch'
                )

            else:
                st.warning("Inventory data not available.")


    st.sidebar.info("Retail Forecasting Project Dashboard")
