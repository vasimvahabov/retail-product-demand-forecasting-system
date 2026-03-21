import os
import streamlit as st
import pandas as pd
import logging
import pipeline

# Initialize variables
workdir = os.getcwd()
dir_input = f"{workdir}/data/"
dir_output = f"{workdir}/out/"
dir_figures = f"{dir_output}figures/"
dir_data_output = f"{dir_output}data/"
path_train_input = f"{dir_input}train.csv"
path_store_input = f"{dir_input}store.csv"

stores_to_use = [
    1,
    # 3,
    # 8,
    # 15,
    # 25
]

os.makedirs(dir_output, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@st.cache_data(show_spinner=True)
def run_pipeline():
    pipeline.run(path_train_input, path_store_input, dir_output, dir_data_output, dir_figures, stores_to_use)
    return True

logging.info("Running pipeline...")
run_pipeline()
logging.info("Pipeline running completed!")

runtime_env = os.getenv("RUNTIME_ENV", "LOCAL")
logging.info(f"Runtime environment: {runtime_env}")
if runtime_env != "LOCAL":
    exit(0)

st.title("Retail Sales Forecasting & Inventory Optimization Dashboard")
st.write("Explore sales trends, forecasting results, model comparison, and inventory recommendations.")

# Select store to analyse
store_options = []
for store_id in stores_to_use:
    store_options.append(f"Store {store_id}")
store_selected = st.sidebar.selectbox("Select Store", store_options)
store_id = 1 if store_selected == "Store 1" else 3

# File paths based on store selection
sales_file = os.path.join(dir_data_output, f"store_{store_id}_cleaned.csv")
model_file = os.path.join(dir_data_output, f"store_{store_id}_model_comparison.csv")
inventory_file = os.path.join(dir_data_output, f"store_{store_id}_inventory_recommendation.csv")

# Figures
figures = {
    "daily_sales": os.path.join(dir_figures, f"store_{store_id}_daily_sales.png"),
    "lstm_forecast": os.path.join(dir_figures, f"store_{store_id}_lstm_forecast.png"),
    "monthly_sales": os.path.join(dir_figures, f"store_{store_id}_monthly_sales.png"),
    "prophet_forecast": os.path.join(dir_figures, f"store_{store_id}_prophet_forecast.png"),
    "sales_peaks": os.path.join(dir_figures, f"store_{store_id}_sales_peaks.png"),
    "weekday_pattern": os.path.join(dir_figures, f"store_{store_id}_weekday_pattern.png"),
    "model_comparison": os.path.join(dir_figures, f"store_{store_id}_model_comparison.png"),
}
# Global figure
sales_distribution_fig = os.path.join(dir_figures, "sales_distribution.png")

# Load data
try:
    logging.info(f"Loading sales data from: {sales_file}")
    sales_data = pd.read_csv(sales_file)
    sales_data["Date"] = pd.to_datetime(sales_data["Date"])
    logging.info("Sales data loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load sales data: {e}", exc_info=True)
    st.error("Failed to load sales data. Check logs for details.")
    raise

try:
    logging.info(f"Loading model results from: {model_file}")
    model_results = pd.read_csv(model_file)
    logging.info("Model results loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load model results: {e}", exc_info=True)
    st.error("Failed to load model results. Check logs for details.")
    raise

try:
    logging.info(f"Loading inventory data from: {inventory_file}")
    inventory = pd.read_csv(inventory_file)
    logging.info("Inventory data loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load inventory data: {e}", exc_info=True)
    st.error("Failed to load inventory data. Check logs for details.")
    raise

# Sidebar Navigation
section = st.sidebar.radio(
    "Select Section",
    ["Sales Overview", "Forecast", "Model Comparison", "Inventory Recommendation"]
)

# Sales Overview
if section == "Sales Overview":
    st.header("Sales History")
    st.subheader("Daily Sales")
    try:
        if os.path.exists(figures["daily_sales"]):
            st.image(figures["daily_sales"])
        else:
            st.warning("Daily sales figure not found.")
    except Exception as e:
        logging.error(f"Error displaying daily sales figure: {e}", exc_info=True)
        st.error("Error displaying daily sales figure. Check logs for details.")
        raise

    st.subheader("Monthly Sales Trend")
    try:
        if os.path.exists(figures["monthly_sales"]):
            st.image(figures["monthly_sales"])
        else:
            st.warning("Monthly sales figure not found.")
    except Exception as e:
        logging.error(f"Error displaying monthly sales figure: {e}", exc_info=True)
        st.error("Error displaying monthly sales figure. Check logs for details.")
        raise

    st.subheader("Sales Peaks & Seasonality")
    try:
        if os.path.exists(figures["sales_peaks"]):
            st.image(figures["sales_peaks"])
        else:
            st.warning("Sales peaks figure not found.")
        if os.path.exists(figures["weekday_pattern"]):
            st.image(figures["weekday_pattern"])
        else:
            st.warning("Weekday pattern figure not found.")
    except Exception as e:
        logging.error(f"Error displaying sales peaks/seasonality figures: {e}", exc_info=True)
        st.error("Error displaying sales peaks/seasonality figures. Check logs for details.")
        raise

    st.subheader("Overall Sales Distribution")
    try:
        if os.path.exists(sales_distribution_fig):
            st.image(sales_distribution_fig)
        else:
            st.warning("Sales distribution figure not found.")
    except Exception as e:
        logging.error(f"Error displaying sales distribution figure: {e}", exc_info=True)
        st.error("Error displaying sales distribution figure. Check logs for details.")
        raise

# Forecast
elif section == "Forecast":
    st.header("Forecasting Results")
    st.subheader("Prophet Forecast")
    try:
        if os.path.exists(figures["prophet_forecast"]):
            st.image(figures["prophet_forecast"])
        else:
            st.warning("Prophet forecast figure not found.")
    except Exception as e:
        logging.error(f"Error displaying Prophet forecast figure: {e}", exc_info=True)
        st.error("Error displaying Prophet forecast figure. Check logs for details.")
        raise

    st.subheader("LSTM Forecast")
    try:
        if os.path.exists(figures["lstm_forecast"]):
            st.image(figures["lstm_forecast"])
        else:
            st.warning("LSTM forecast figure not found.")
    except Exception as e:
        logging.error(f"Error displaying LSTM forecast figure: {e}", exc_info=True)
        st.error("Error displaying LSTM forecast figure. Check logs for details.")
        raise

# Model Comparison
elif section == "Model Comparison":
    st.header("Model Performance Comparison")
    try:
        st.dataframe(model_results)
    except Exception as e:
        logging.error(f"Error displaying model results: {e}", exc_info=True)
        st.error("Error displaying model results. Check logs for details.")
        raise

    st.subheader("Visual Comparison")
    try:
        if os.path.exists(figures["model_comparison"]):
            st.image(figures["model_comparison"])
        else:
            st.warning("Model comparison figure not found.")
    except Exception as e:
        logging.error(f"Error displaying model comparison figure: {e}", exc_info=True)
        st.error("Error displaying model comparison figure. Check logs for details.")
        raise

# Inventory Recommendation
elif section == "Inventory Recommendation":
    st.header("Inventory Optimization")
    try:
        st.dataframe(inventory)
    except Exception as e:
        logging.error(f"Error displaying inventory data: {e}", exc_info=True)
        st.error("Error displaying inventory data. Check logs for details.")
        raise
    st.success("Using demand forecasting helps reduce stockouts and overstocking.")

# Footer
st.sidebar.info("Retail Forecasting Project Dashboard")
