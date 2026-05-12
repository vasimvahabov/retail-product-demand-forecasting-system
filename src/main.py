"""
Retail Product Demand Forecasting Sales Forecasting Dashboard (Streamlit Application)

Runs ML pipeline, loads outputs, and displays store-level analytics.
"""

import os
import sys
import pandas as pd
import argparse
import logging
from datetime import datetime

# Paths/Global Configs
workdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dir_log = os.path.join(workdir, "log")
dir_input = os.path.join(workdir, "data")
dir_output = os.path.join(workdir, "out")
dir_figures = os.path.join(dir_output, "figures")
dir_data_output = os.path.join(dir_output, "data")
path_train_input = os.path.join(dir_input, "train.csv")
path_store_input = os.path.join(dir_input, "store.csv")

# Logging Config

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
logger.info("Execute `python src/main.py --help` to see details.")

# Phase Config
phase_help = {
    1: "Project Setup",
    2: "Data Cleaning",
    3: "EDA",
    4: "Forecasting",
    5: "Evaluation",
    6: "Inventory Optimization"
}
valid_phases = set(phase_help.keys())

pipeline_artifacts = {
    2: {
        "data": ["sales_data_cleaned.csv"],
        "figures": ["sales_distribution.png"],
    },
    3: {
        "data": ["store_{store}_eda.csv"],
        "figures": [
            "store_{store}_daily_sales.png",
            "store_{store}_monthly_sales.png",
            "store_{store}_sales_peaks.png",
            "store_{store}_weekday_pattern.png"
        ],
    },
    4: {
        "data": [
            "store_{store}_forecasts.csv",
            "store_{store}_forecast_30.csv"
        ],
        "figures": [
            "store_{store}_lstm_forecast.png",
            "store_{store}_arima_forecast.png",
            "store_{store}_prophet_forecast.png",
            "store_{store}_xgboost_forecast.png"
        ],
    },
    5: {
        "data": ["store_{store}_evaluation.csv"],
        "figures": [
            "store_{store}_model_comparison_rmse.png",
            "store_{store}_model_comparison_mape.png",
        ],
    },
    6: {
        "data": ["store_{store}_inventory.csv"],
    }
}


# Helper Functions
def missing_pipeline_artifacts(stores):
    missing_artifacts = []

    for store in stores:
        for phase in valid_phases:
            if phase not in pipeline_artifacts:
                continue

            artifacts = pipeline_artifacts[phase]

            for file in artifacts.get("data", []):
                path = os.path.join(dir_data_output, file.format(store=store))
                if not os.path.exists(path):
                    missing_artifacts.append({
                        "store": store,
                        "phase": phase,
                        "path": path,
                        "type": "data"
                    })

            for file in artifacts.get("figures", []):
                path = os.path.join(dir_figures, file.format(store=store))
                if not os.path.exists(path):
                    missing_artifacts.append({
                        "store": store,
                        "phase": phase,
                        "path": path,
                        "type": "figure"
                    })

    return missing_artifacts


def stop_application(exit_code=1):
    logger.info("Stopping application...")
    sys.exit(exit_code)


def main():
    # Commanline Argument Parser
    parser = argparse.ArgumentParser(
        description="""
    Retail Product Demand Forecasting System
    
    Stores:
      - Refer to `data/store.csv` for valid Store IDs (column 'Store').
    
    Phases:
      - Available phases:
          1: Project Setup
          2: Data Cleaning
          3: EDA
          4: Forecasting
          5: Evaluation
          6: Inventory Optimization
    
    Full Example:
      python src/main.py --stores 1 2 --phases 2 3
    """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--stores",
        type=int,
        nargs="+",
        default=[],
        help="List of Store IDs to process (e.g., --stores 1 2 3)"
    )

    parser.add_argument(
        "--phases",
        type=int,
        nargs="+",
        default=[],
        help="List of Phase IDs to run (e.g., --phases 1 2 3)"
    )

    parser.add_argument(
        "--all-phases",
        action="store_true",
        help="Execute all pipeline phases (same as --phases 1 2 3 4 5 6)"
    )
    cli_arguments, _ = parser.parse_known_args()

    stores_to_process = cli_arguments.stores
    if not stores_to_process:
        logger.error("No stores passed via commandline arguments!")
        logger.info("Please provide at least one store ID via `--stores` flag (e.g., `--stores 1 2 3`)!")
        stop_application()
    else:
        valid_stores = pd.read_csv(path_store_input)['Store'].tolist()
        invalid_stores = [s for s in stores_to_process if s not in valid_stores]
        if invalid_stores:
            logger.error("Invalid store IDs detected %s!", invalid_stores)
            stop_application()
        logger.info("Selected stores %s!", stores_to_process)

    # Runtime Mode detection (streamlit/python command)
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        STREAMLIT_AVAILABLE = True
    except ImportError:
        STREAMLIT_AVAILABLE = False

    runtime_mode = (
        "streamlit"
        if STREAMLIT_AVAILABLE and get_script_run_ctx() is not None
        else "python"
    )

    # Python Execution Mode
    if runtime_mode == "python":
        logger.info("Python execution detected!")

        if cli_arguments.all_phases:
            phases_to_run = sorted(phase_help.keys())
        else:
            phases_to_run = cli_arguments.phases

            invalid_phases = [phase for phase in phases_to_run if phase not in valid_phases]
            phases_to_run = [phase for phase in phases_to_run if phase in valid_phases]

            if invalid_phases:
                logger.warning("Ignoring invalid phases %s!", sorted(invalid_phases))
                logger.info("Available phases: %s", list(valid_phases))

            if not phases_to_run:
                logger.error("No valid phases selected!")
                logger.info("Available phases: %s", list(valid_phases))
                stop_application()

            missing_artifacts = missing_pipeline_artifacts(stores_to_process)

            from collections import defaultdict

            missing_artifacts_by_store = defaultdict(set)

            max_phase_to_run = max(phases_to_run)
            for artifact in missing_artifacts:
                phase = artifact["phase"]
                if phase not in phases_to_run and phase < max_phase_to_run:
                    missing_artifacts_by_store[artifact["store"]].add(artifact["phase"])

            if missing_artifacts_by_store:
                logger.error("Missing required previous phase artifacts for pipeline execution!")
                for store, phases in missing_artifacts_by_store.items():
                    phases = sorted(phases)
                    logger.info(
                        "Run `python src/main.py --stores %s --phases %s`",
                        store,
                        " ".join(map(str, phases))
                    )
                stop_application()

        logger.info("Selected phases %s!", phases_to_run)

        try:
            import pipeline

            logger.info("Starting pipeline execution...")
            pipeline.run(
                phases_to_run,
                path_train_input,
                path_store_input,
                dir_output,
                dir_data_output,
                dir_figures,
                stores_to_process
            )
            logger.info("Pipeline execution completed!")

        except FileNotFoundError:
            logger.exception("Required file missing on pipeline execution!")
            stop_application()

        except Exception:
            logger.exception("Pipeline execution failed!")
            stop_application()

        logger.info(
            "Run `streamlit run src/main.py -- --stores %s` to launch Streamlit dashboard!",
            " ".join(map(str, stores_to_process))
        )


    # Streamlit Execution Mode
    else:
        logger.info("Streamlit execution detected!")
        missing_artifacts = missing_pipeline_artifacts(
            stores_to_process
        )

        if not missing_artifacts:
            import dashboard
            dashboard_artifacts = {
                "data": [],
                "figures": []
            }

            for phase in pipeline_artifacts.values():
                dashboard_artifacts["data"].extend(phase.get("data", []))
                dashboard_artifacts["figures"].extend(phase.get("figures", []))

            dashboard.launch(stores_to_process, dir_data_output, dir_figures, dashboard_artifacts)
        else:
            from collections import defaultdict

            missing_artifacts_by_store = defaultdict(set)

            for artifact in missing_artifacts:
                missing_artifacts_by_store[artifact["store"]].add(artifact["phase"])

            logger.error("Missing required pipeline outputs for dashboard!")
            for store, phases in missing_artifacts_by_store.items():
                phases = sorted(phases)

                logger.info(
                    "Run `python src/main.py --stores %s --phases %s`",
                    store,
                    " ".join(map(str, phases))
                )

            stop_application()


if __name__ == "__main__":
    main()
