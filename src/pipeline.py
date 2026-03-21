# Import libraries
import time
import logging
import phases.phase01_project_setup as project_setup
import phases.phase02_data_cleaning as data_cleaning
import phases.phase03_eda as eda
import phases.phase04_forecasting_models as forecasting_models
import phases.phase05_model_evaluation as model_evaluation
import phases.phase06_inventory_optimization as inventory_optimization

def run(path_train_input, path_store_input, dir_output, dir_data_output, dir_figures, stores_to_use):
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{dir_output}pipeline.log"),
            logging.StreamHandler()
        ]
    )

    # Phase 1: Project Setup
    start_time = time.time()
    logging.info("Phase 1: Project Setup - Started")
    try:
        project_setup.run(dir_output, dir_data_output, dir_figures)
        logging.info(f"Phase 1: Project Setup - Completed in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        logging.error(f"Phase 1: Project Setup - Failed: {e}", exc_info=True)
        raise

    # Phase 2: Data Cleaning
    start_time = time.time()
    logging.info("Phase 2: Data Cleaning - Started")
    try:
        df_cleaned = data_cleaning.run(dir_figures, path_train_input, path_store_input)
        logging.info(f"Cleaned DataFrame shape: {df_cleaned.shape}")
        logging.info(f"Phase 2: Data Cleaning - Completed in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        logging.error(f"Phase 2: Data Cleaning - Failed: {e}", exc_info=True)
        raise

    # Phase 3: EDA
    start_time = time.time()
    logging.info("Phase 3: EDA - Started")
    try:
        df_cleaned_stores = eda.run(df_cleaned, dir_data_output, dir_figures, stores_to_use)
        logging.info(f"Phase 3: EDA - Completed in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        logging.error(f"Phase 3: EDA - Failed: {e}", exc_info=True)
        raise

    # Phase 4: Forecasting Models
    start_time = time.time()
    logging.info("Phase 4: Forecasting Models - Started")
    try:
        forecasts = forecasting_models.run(df_cleaned_stores, dir_figures)
        logging.info(f"Phase 4: Forecasting Models - Completed in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        logging.error(f"Phase 4: Forecasting Models - Failed: {e}", exc_info=True)
        raise

    # Phase 5: Model Evaluation
    start_time = time.time()
    logging.info("Phase 5: Model Evaluation - Started")
    try:
        model_evaluation.run(forecasts, dir_data_output, dir_figures)
        logging.info(f"Phase 5: Model Evaluation - Completed in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        logging.error(f"Phase 5: Model Evaluation - Failed: {e}", exc_info=True)
        raise

    # Phase 6: Inventory Optimization
    start_time = time.time()
    logging.info("Phase 6: Inventory Optimization - Started")
    try:
        inventory_optimization.run(forecasts, dir_data_output)
        logging.info(f"Phase 6: Inventory Optimization - Completed in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        logging.error(f"Phase 6: Inventory Optimization - Failed: {e}", exc_info=True)
        raise
