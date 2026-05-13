# retail-product-demand-forecasting-system

## Dataset

This project uses the [**Rossmann Store Sales dataset**](https://www.kaggle.com/competitions/rossmann-store-sales/data).

## Quick Start

Create output and log directories in the workspace:

```bash
mkdir -p out log
```

### Building from Source Code

#### Clone the repository:

```bash
git clone https://github.com/vasimvahabov/retail-product-demand-forecasting-system.git
  
# move to the workdir
cd retail-product-demand-forecasting-system/
```

#### Option 1: Run with Python

##### Create a virtual environment:

```bash
python3 -m venv venv
```

##### Activate the virtual environment:

```bash
# On macOS/Linux
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

##### Install the required dependencies:

```bash 
pip install -r requirements.txt
```

##### Run the pipeline:

```bash
python src/main.py --stores 1 --all-phases 
```

> **Note:** Run `python src/main.py --help` for more details.

##### Run the Streamlit application:

```bash
streamlit src/main.py -- --stores 1 --all-phases  
```

> **Note:**
>- Pipeline artifacts should be present to run the Streamlit application for specified store/stores.
>- The app will open in your browser at `http://localhost:8501/`.
>- Allow time for visualizing, depending on the specified store/stores.

##### Option 2: Run with Docker

###### Build Docker Image:

```bash
docker build -t retail-product-demand-forecasting-system:$(date +%s) .
```

###### Run the pipeline mounting log and output directories into Docker Image:

```bash
docker run -e EXECUTION_CMD="python src/main.py --stores 1 --all-phases"
-v $(pwd)/out:/app/out -v $(pwd)/log:/app/log retail-product-demand-forecasting-system:$(date +%s)
```

> **Note:** Pass EXECUTION_CMD as `python src/main.py --help` for more details.

###### Run the Streamlit application mounting log and output directories into Docker Image:

```bash
docker run -e EXECUTION_CMD="streamlit run src/main.py -- --stores 1"
-v $(pwd)/out:/app/out -v $(pwd)/log:/app/log retail-product-demand-forecasting-system:$(date +%s)
```

> **Note:**
>- Pipeline artifacts should be present to run the Streamlit application for specified store/stores.
>- The app will print URLs (Local, Network, External) in the terminal. Use the Local Network to access the app in your
   browser.
>- Allow time for visualizing, depending on the specified store/stores.

### Docker Image

##### Pull Latest Built Docker Image from GitHub Container Registry:

```bash
docker pull ghcr.io/vasimvahabov/retail-product-demand-forecasting-system:latest
```

###### Run the pipeline mounting log and output directories into the Pulled Docker Image:

```bash
docker run -e EXECUTION_CMD="python src/main.py --stores 1 --all-phases"
-v $(pwd)/out:/app/out -v $(pwd)/log:/app/log ghcr.io/vasimvahabov/retail-product-demand-forecasting-system:latest
```

###### Run the Streamlit application mounting log and output directories into Docker Image:

```bash
docker run -e EXECUTION_CMD="streamlit run src/main.py -- --stores 1"
-v $(pwd)/out:/app/out -v $(pwd)/log:/app/log ghcr.io/vasimvahabov/retail-product-demand-forecasting-system:latest
```

> **Note:**
>- Pipeline artifacts should be present to run the Streamlit application for specified store/stores.
>- The app will print URLs (Local, Network, External) in the terminal. Use the Local Network to access the app in your
   browser.
>- Allow time for visualizing, depending on the specified store/stores.

### Stop Application (optional)

Press `Ctrl + C` in the terminal

## CI-CD

Processed data, generated plots, and logs are available in the ci/output branch:

- `out/data` - Processed datasets
- `out/figures` - generated plots
- `log` - Logs

> **Note:** GitHub Actions job output can be inspected for troubleshooting on workflow execution.

## Logging

Logs are written to file in the `log` directory and will also be printed in the terminal.
