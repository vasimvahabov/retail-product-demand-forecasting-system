"""
Pipeline Executor

Runs all ML pipeline phases in sequence:
1. Project setup
2. Data cleaning
3. EDA
4. Forecasting
5. Evaluation
6. Inventory optimization
"""

import time
import logging
import phases.phase01_project_setup as project_setup
import phases.phase02_data_cleaning as data_cleaning
import phases.phase03_eda as eda
import phases.phase04_forecasting_models as forecasting_models
import phases.phase05_model_evaluation as model_evaluation
import phases.phase06_inventory_optimization as inventory_optimization

logger = logging.getLogger(__name__)


def run_phase(name, func, *args, **kwargs):
    """
    Executes a pipeline phase with timing and logging.

    Args:
        name (str): Phase name for logs
        func (callable): Function to execute
        *args: Positional arguments for function
        **kwargs: Keyword arguments for function

    Returns:
        Any: Output of executed function

    Raises:
        Exception: Propagates any error from phase execution
    """

    start_time = time.time()
    logger.info("Starting %s execution...", name)
    try:
        result = func(*args, **kwargs)
        logger.info("%s execution completed in %.2f seconds!", name, time.time() - start_time)
        return result

    except Exception:
        logger.exception("%s execution failed!", name)
        raise


def run(phases_to_run, path_train_input, path_store_input, dir_output, dir_data_output, dir_figures, stores_to_process):
    """
    Executes full ML pipeline end-to-end.

    Args:
        phases_to_run (list[int]): Phase IDs to run
        path_train_input (str): Path to training dataset
        path_store_input (str): Path to store data
        dir_output (str): Root output directory
        dir_data_output (str): Processed data output directory
        dir_figures (str): Figures output directory
        stores_to_process (list[int]): Store IDs to process

    Returns:
        None
    """

    for phase in phases_to_run:
        match phase:
            case 1:
                # Phase 1: Project Setup
                run_phase(
                    "Phase 1: Project Setup",
                    project_setup.run,
                    dir_output,
                    dir_data_output,
                    dir_figures
                )

            case 2:
                # Phase 2: Data Cleaning
                run_phase(
                    "Phase 2: Data Cleaning",
                    data_cleaning.run,
                    path_train_input,
                    path_store_input,
                    dir_data_output,
                    dir_figures
                )

            case 3:
                # Phase 3: EDA
                run_phase(
                    "Phase 3: EDA",
                    eda.run,
                    dir_data_output,
                    dir_figures,
                    stores_to_process
                )

            case 4:
                # Phase 4: Forecasting Models
                run_phase(
                    "Phase 4: Forecasting Models",
                    forecasting_models.run,
                    dir_data_output,
                    dir_figures,
                    stores_to_process
                )

            case 5:
                # Phase 5: Model Evaluation
                run_phase(
                    "Phase 5: Model Evaluation",
                    model_evaluation.run,
                    dir_data_output,
                    dir_figures,
                    stores_to_process
                )

            case 6:
                # Phase 6: Inventory Optimization
                run_phase(
                    "Phase 6: Inventory Optimization",
                    inventory_optimization.run,
                    dir_data_output,
                    stores_to_process
                )

            case _:
                logger.warning("Unknown phase %s!", phase)
