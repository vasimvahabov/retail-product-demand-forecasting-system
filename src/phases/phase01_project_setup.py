"""
Setups output directories for the full pipeline, cleaning-up any previous builds.

Processes cleaned dataset per store, engineers features, and generates visualizations.
"""

import os
import shutil
import logging

logger = logging.getLogger(__name__)

def run(dir_output, dir_data_output, dir_figures):
    """
    Prepares output directories for the pipeline.

    Deletes previous outputs and recreates required folders.

    Args:
        dir_output (str): Root output directory.
        dir_data_output (str): Data output directory.
        dir_figures (str): Figures output directory.

    Returns:
        None
    """

    logger.info("Cleaning previous output...")
    try:
        if os.path.exists(dir_output):
            shutil.rmtree(dir_output)
            logger.info("Cleaned up previous output directory %s!", dir_output)
        else:
            logger.info("No previous output directory found at %s!", dir_output)
    except Exception:
        logger.exception("Failed to clean previous output!")

    logger.info("Creating necessary folders...")
    try:
        for path in [dir_output, dir_data_output, dir_figures]:
            os.makedirs(path, exist_ok=True)
        logger.info("Created output directories under %s!", dir_output)
        logger.info("Project setup phase completed successfully!")
    except Exception:
        logger.exception("Failed to create directories under %s!", dir_output)