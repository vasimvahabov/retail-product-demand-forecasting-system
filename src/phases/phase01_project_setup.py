import os
import shutil
import logging

def run(dir_output, dir_data_output, dir_figures):

    """
    Cleans up previous output and creates necessary folders for the pipeline.

    Args:
        dir_output (str): Path to the main output directory.
        dir_data_output (str): Path to the data output directory.
        dir_figures (str): Path to the figures output directory.
    """

    # Configure logging for this function
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Clean-up previous output
    logging.info("Cleaning previous output...")
    try:
        if os.path.exists(dir_output):
            shutil.rmtree(dir_output)
            logging.info(f"Successfully removed previous output directory: {dir_output}")
        else:
            logging.info(f"No previous output directory found at: {dir_output}")
    except Exception as e:
        logging.error(f"Failed to clean previous output: {e}", exc_info=True)
        raise

    # Create necessary folders
    logging.info("Creating necessary folders...")
    try:
        os.makedirs(dir_figures, exist_ok=True)
        os.makedirs(dir_data_output, exist_ok=True)
        logging.info(f"Successfully created directories: {dir_output}, {dir_figures}, {dir_data_output}")
    except Exception as e:
        logging.error(f"Failed to create directories: {e}", exc_info=True)
        raise
