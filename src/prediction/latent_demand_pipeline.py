"""
Integration point for Latent Demand Model in the main pipeline.

This module provides functions to integrate the latent demand model training
into the existing data pipeline architecture.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

from src.configs.config import config
from src.utils.io import read_parquet, write_parquet
from src.prediction.latent_demand_model import LatentDemandModel

log = logging.getLogger("pipeline.latent_demand")


def train_latent_demand_model(
    model_output_dir: Optional[Path] = None,
    train_years: list = None,
    val_years: list = None,
    prediction_year: int = 2026,
    prediction_month: int = 1,
) -> Dict:
    """
    Train latent demand model and generate predictions.

    Args:
        model_output_dir: Directory to save trained model
        train_years: Years for training (default: [2023, 2024])
        val_years: Years for validation (default: [2025])
        prediction_year: Year to predict for
        prediction_month: Month to predict for

    Returns:
        Dictionary with results including metrics and predictions file path
    """
    if model_output_dir is None:
        model_output_dir = config.MODEL_PATH / "latent_demand_model"

    if train_years is None:
        train_years = [2023, 2024]
    if val_years is None:
        val_years = [2025]

    log.info("Training Latent Demand Model")
    log.info("=" * 60)

    # Load fact table
    df = read_parquet(config.GOLD_PATH, "fact_table")
    log.info(f"Loaded fact table: {df.shape[0]:,} rows")

    # Strip audit columns
    meta_cols = [c for c in df.columns if c.startswith("_")]
    if meta_cols:
        df.drop(columns=meta_cols, inplace=True)

    # Prepare disaggregated and aggregated copies
    df_wide = df.sort_values(["Outlet_ID", "Year", "Month"]).reset_index(drop=True)

    df_agg = df_wide.groupby(
        ["Outlet_ID", "Year", "Month"],
        as_index=False,
        sort=False,
    ).agg(
        Volume_Liters=("Volume_Liters", "sum"),
        Cooler_Count=("Cooler_Count", "first"),
        Outlet_Size=("Outlet_Size", "first"),
        Outlet_Type=("Outlet_Type", "first"),
        Outlet_Size_Imputed=("Outlet_Size_Imputed", "first"),
        Seasonality_Index=("Seasonality_Index", "first"),
        holiday_count=("holiday_count", "first"),
        school_score=("school_score", "first"),
        hospital_score=("hospital_score", "first"),
        bus_stop_score=("bus_stop_score", "first"),
        tourist_score=("tourist_score", "first"),
        Latitude=("Latitude", "first"),
        Longitude=("Longitude", "first"),
    ).sort_values(["Outlet_ID", "Year", "Month"]).reset_index(drop=True)

    # Train model
    model = LatentDemandModel()

    log.info(f"Training on years: {train_years}")
    log.info(f"Validation on years: {val_years}")

    metrics = model.fit(df_wide, df_agg, train_years=train_years, val_years=val_years)

    # Save model
    model.save(model_output_dir)
    log.info(f"Model saved to {model_output_dir}")

    # Generate predictions
    log.info(f"Generating predictions for {prediction_year}-{prediction_month:02d}")

    import numpy as np

    # Calculate lags on full dataset before getting latest
    df_wide_sorted = df_wide.sort_values(["Outlet_ID", "Distributor_ID", "SKU_ID", "Year", "Month"])
    df_wide_sorted["vol_lag_1"] = df_wide_sorted.groupby(["Outlet_ID", "Distributor_ID", "SKU_ID"])["Volume_Liters"].shift(1)
    df_wide_sorted["vol_lag_12"] = df_wide_sorted.groupby(["Outlet_ID", "Distributor_ID", "SKU_ID"])["Volume_Liters"].shift(12)
    df_wide_sorted["rolling3_avg"] = (
        df_wide_sorted.groupby(["Outlet_ID", "Distributor_ID", "SKU_ID"])["Volume_Liters"]
        .transform(lambda x: x.rolling(3, min_periods=2).mean().shift(1))
    )

    # Get latest observations
    latest_wide = (
        df_wide_sorted
        .groupby(["Outlet_ID", "Distributor_ID", "SKU_ID"])
        .last()
        .reset_index()
    )

    # Create prediction dataset
    pred_data = latest_wide.copy()
    pred_data["Year"] = prediction_year
    pred_data["Month"] = prediction_month
    pred_data["month_sin"] = np.sin(2 * np.pi * prediction_month / 12)
    pred_data["month_cos"] = np.cos(2 * np.pi * prediction_month / 12)
    
    # Calculate correct lags for prediction target
    # vol_lag_1 is the last actual volume (from latest_wide["Volume_Liters"])
    pred_data["vol_lag_1"] = pred_data.get("Volume_Liters", pred_data["vol_lag_1"])
    # Note: We keep the Notebook's logic for rolling3_avg where it takes the previous month's rolling avg if available.
    pred_data["rolling3_avg"] = pred_data.get("rolling3_avg", pred_data["Volume_Liters"])

    # Add outlet features from latest aggregated data
    latest_agg = df_agg.sort_values(["Outlet_ID", "Year", "Month"]).groupby("Outlet_ID").last().reset_index()

    outlet_cols = [
        "Outlet_Size", "Outlet_Type", "Cooler_Count", "Outlet_Size_Imputed",
        "Seasonality_Index", "holiday_count", "school_score", "hospital_score",
        "bus_stop_score", "tourist_score", "Latitude", "Longitude",
    ]
    for col in outlet_cols:
        if col in latest_agg.columns and col not in pred_data.columns:
            pred_data[col] = pred_data["Outlet_ID"].map(latest_agg.set_index("Outlet_ID")[col])

    # Add outlet mean volumes
    if "outlet_vol_mean" not in pred_data.columns and model.outlet_means:
        pred_data["outlet_vol_mean"] = pred_data["Outlet_ID"].map(model.outlet_means)

    # Generate predictions
    predictions = model.predict(pred_data)

    # Aggregate to outlet level
    pred_data["predicted_volume"] = predictions
    outlet_predictions = pred_data.groupby("Outlet_ID").agg(
        predicted_volume=("predicted_volume", "sum"),
        Latitude=("Latitude", "first"),
        Longitude=("Longitude", "first"),
    ).reset_index()

    # Save predictions
    pred_output = config.PREDICTION_PATH / f"latent_demand_predictions_{prediction_year}_{prediction_month:02d}.parquet"
    rows = write_parquet(outlet_predictions, config.PREDICTION_PATH, f"latent_demand_predictions_{prediction_year}_{prediction_month:02d}", "gold")
    log.info(f"Predictions saved to {pred_output} of {rows} rows")

    # Compile results
    results = {
        "model_saved": True,
        "model_dir": str(model_output_dir),
        "predictions_file": str(pred_output),
        "n_outlets": len(outlet_predictions),
        "total_predicted_volume": outlet_predictions["predicted_volume"].sum(),
        "mean_per_outlet": outlet_predictions["predicted_volume"].mean(),
        "metrics": metrics,
    }

    log.info("=" * 60)
    log.info("Latent Demand Model Training Complete")
    log.info(f"  Outlets: {results['n_outlets']:,}")
    log.info(f"  Total volume: {results['total_predicted_volume']:,.0f} L")
    log.info(f"  Mean/outlet: {results['mean_per_outlet']:.1f} L")
    for metric_name, metric_value in metrics.items():
        log.info(f"  {metric_name}: {metric_value:.4f}")
    log.info("=" * 60)

    return results


if __name__ == "__main__":
    import logging
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  [%(levelname)-8s]  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    train_latent_demand_model()
