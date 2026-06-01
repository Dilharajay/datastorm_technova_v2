# OCTAVE – John Keells Group: Latent Demand & Budget Optimization Engine

This repository contains the end-to-end solution for the Final Round of the Data Storm v7.0 competition, powered by John Keells Group. The project is designed to shift a leading beverage manufacturer's strategy from historical-based allocation to a forward-looking, potential-based model.

The core objective is to estimate the **Maximum Monthly Purchase Potential** for 20,000 traditional trade outlets and use this insight to drive strategic decisions, including the optimized allocation of a promotional budget.

## Advanced Problem Statement

This solution directly addresses the three core challenges of the final round:

1.  **Spatial Distance-Decay Modeling:** Implements a non-linear distance-decay model (Exponential Decay) to weigh the influence of nearby Points of Interest (POIs) more heavily, providing a nuanced understanding of an outlet's location.
2.  **Competitive Catchment Density:** Calculates the density of competing outlets within a 500-meter radius to adjust sales potential based on market saturation and competitive intensity.
3.  **Marketing Spend Optimization:** Utilizes a linear programming model to allocate a **LKR 5 Million** promotional budget across outlets in the Western Province, aiming to maximize additional sales volume for January 2026.

## Key Features

-   **Medallion Lakehouse Architecture:** A robust, idempotent pipeline that processes data through Bronze (raw), Silver (cleaned & enriched), and Gold (aggregated for modeling) layers.
-   **Advanced Feature Engineering:**
    -   **Competitive Catchment Density:** Uses spatial joins (EPSG:5235) to count competitors within a 500m radius.
    -   **POI Distance-Decay Scoring:** Applies exponential decay functions to score the influence of schools, hospitals, bus stops, and tourist attractions.
-   **Two-Stage Latent Demand Model:**
    1.  **Censoring Detection:** Employs proxy rules to identify and score observations where sales were likely constrained by supply issues.
    2.  **Potential Estimation:** Uses a Tobit model to handle censored data, followed by an XGBoost model to predict the de-censored, latent demand (MAE 4.87, R² 0.985).
-   **Budget Optimization:** A PuLP-based linear programming solver that allocates the LKR 5M budget to maximize incremental volume.
-   **Functional Explainable AI (XAI):** Integrates a local LLM (Ollama) to generate dynamic, business-friendly explanations for each outlet's predicted potential, detailing the key drivers and environmental factors.
-   **Interactive Intelligence Dashboard:** A Streamlit web application that allows users to browse predictions, filter by province and distributor, and drill down into a detailed view for any outlet, including its XAI-generated narrative.

## Repository Structure

```text
technova_datastrom_v3/
├── run_pipeline.py              # Main entry point to run the full ETL and modeling pipeline
├── app.py                       # The Streamlit web application
├── src/
│   ├── configs/                 # Paths and pipeline configuration
│   ├── ingest/                  # Bronze-layer: Raw data ingestion
│   ├── cleaning/                # Silver-layer: Cleaning, normalization, and feature engineering
│   ├── optimization/            # PuLP-based budget optimizer
│   ├── prediction/              # Latent demand estimation model
│   └── xai/                     # Outlet explanation generator (Ollama)
├── data/
│   ├── raw/                     # Source CSVs and external data
│   ├── bronze/                  # Bronze Parquet outputs
│   ├── silver/                  # Silver Parquet outputs
│   ├── gold/                    # Gold fact table, predictions, and model artifacts
│   └── rejects/                 # Rejected records store for data quality assurance
├── notebooks/                   # EDA and modeling notebooks
└── scripts/
    ├── test_latent_demand_model.py   # Demo script to check dependencies and component functionality
    └── train_latent_demand_model.py  # Standalone script to train the model from Gold-layer data
```

## Quick Start

### 1. Install Dependencies

The project uses `uv` for dependency management.

```bash
uv sync
```

### 2. Run the Full Pipeline

This command executes the entire data pipeline from raw data ingestion to model training, prediction, and budget optimization.

```bash
uv run python run_pipeline.py
```

You can run specific parts of the pipeline using flags:
-   `--skip-etl`: Skips the Bronze, Silver, and Gold stages.
-   `--skip-model`: Skips model training, prediction, and optimization.

### 3. Launch the Outlet Intelligence Web App

To start the interactive dashboard:

```bash
uv run streamlit run app.py
```

## Standalone Scripts

### Functionality Check

To quickly verify dependencies and see a demonstration of the model's components with dummy data, run:

```powershell
python scripts/test_latent_demand_model.py
```

### Standalone Model Training

To run only the model training process using the fact table from the `data/gold/` directory:

```powershell
python scripts/train_latent_demand_model.py
```

This will train the model, save the artifacts, and generate predictions.

## Final Deliverables

This repository generates the following key outputs as required:

| File | Description |
|------------------------------------------------|----------------------------------------------------------------|
| `data/gold/teamname_predictions.csv` | **Latent Potential Output:** Outlet_ID and predicted Maximum_Monthly_Liters for Jan 2026. |
| `data/gold/teamname_budget_allocations.csv` | **Marketing Spend Allocation:** Outlet_ID and the allocated Trade Spend (LKR) for Western Province. |
| `app.py` | **Outlet Intelligence Web App:** The interactive Streamlit dashboard. |

The full codebase, methodology paper, and executive pitch deck are provided as part of the final submission package.
