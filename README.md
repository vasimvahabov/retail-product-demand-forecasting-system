# retail-product-demand-forecasting-system

## Dataset
This project uses the [**Rossmann Store Sales dataset**](https://www.kaggle.com/competitions/rossmann-store-sales/data).

## Quick Start

1. Clone the repository:

    ```sh
    git clone https://github.com/vasimvahabov/retail-product-demand-forecasting-system.git

    # move to workdir
    cd retail-product-demand-forecasting-system/
    ```

2. Create a virtual environment:

    ```sh
    python3 -m venv venv
    ```

3. Activate the virtual environment:

    - On macOS/Linux:
        ```sh
        source venv/bin/activate
        ```
    - On Windows:
        ```sh
        venv\Scripts\activate
        ```

4. Install the required dependencies:

    ```sh
    pip install -r requirements.txt
    ```

5. Run the application:

    ```sh
    streamlit run src/main.py
    ```

   > **Note:** This command will open a new tab in your browser at `http://localhost:8501/`. Please allow 30–60 seconds for the data to be processed. You can monitor the logs in the terminal.

6. Stop the application (optional):

   To stop the app, press `Ctrl + C` in the terminal.

## CI-CD
Processed data, generated plots, and logs are available in the `ci/output` branch.
You can also check the logs directly from the GitHub Actions job outputs if you want to see what happened during the workflow.
