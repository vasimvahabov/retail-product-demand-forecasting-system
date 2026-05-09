"""
Retail Product Demand Forecasting Sales Forecasting Dashboard (Streamlit Application)

Runs ML pipeline, loads outputs, and displays store-level analytics.
"""

import os
import sys
import argparse
import logging
from datetime import datetime

workdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

def stop_application(exit_code=1):
    logger.info("Stopping application...")
    sys.exit(exit_code)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--stores",
    type=int,
    nargs="+",
    default=[1],
    help="List of Store IDs to process (e.g., --stores 1 2 3)"
)

parser.add_argument(
    "--phases",
    type=int,
    nargs="+",
    default=[1],
    help="List of Phase IDs to run (e.g., --phases 1 2 3)"
)
cli_arguments, _ = parser.parse_known_args()

stores_to_process = cli_arguments.stores
logger.info("Stores to process %s!", stores_to_process)

if len(stores_to_process) == 0:
    logger.error("Empty stores_to_process passed via commandline arguments!")
    stop_application()

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


if runtime_mode == "python":
    logger.info("Python execution detected!")

    phases_to_run = cli_arguments.phases
    logger.info("Phases to run %s!", phases_to_run)

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

    logger.info("Run `streamlit run src/main.py` to launch Streamlit dashboard!")

else:
    import dashboard
    logger.info("Streamlit execution detected!")
    dashboard.launch(stores_to_process, dir_data_output, dir_figures)
