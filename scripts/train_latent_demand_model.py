"""
Training script for Latent Demand Estimation Model.

This script trains the two-stage latent demand model:
1. Detects censored observations using proxy rules
2. Recovers latent demand with Tobit model
3. Trains XGBoost on de-censored series
4. Generates predictions for specified months
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

import pandas as pd
import numpy as np

from src.configs.config import config
from src.utils.io import read_parquet
from src.prediction.latent_demand_model import LatentDemandModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-8s]  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("latent_demand.trainer")


class LatentDemandTrainer:
    """Train and manage latent demand estimation model."""

    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = model_dir or config.GOLD_PATH / "latent_demand_model"
        self.model = LatentDemandModel()
        self.train_metrics = None

    def load_and_prepare_datasets(self) -> tuple:
        """
        Load datasets and prepare for modeling.

        Returns:
            (df_wide, df_agg) where:
            - df_wide: Transaction-level data (disaggregated)
            - df_agg: Aggregated outlet-month level data
        """
        log.info("Loading datasets...")

        # Load fact table
        df = read_parquet(config.GOLD_PATH, "fact_table")
        log.info(f"Loaded fact table: {df.shape[0]:,} rows x {df.shape[1]} cols")

        # Strip audit columns
        meta_cols = [c for c in df.columns if c.startswith("_")]
        if meta_cols:
            df.drop(columns=meta_cols, inplace=True)
            log.info(f"Dropped audit columns: {meta_cols}")

        # Disaggregated copy
        df_wide = df.copy()
        df_wide = df_wide.sort_values(["Outlet_ID", "Year", "Month"]).reset_index(drop=True)

        # Aggregated copy (outlet-month level)
        df_agg = df_wide.groupby(
            ["Outlet_ID", "Year", "Month"],
            as_index=False,
            sort=False,
        ).agg(
            Volume_Liters=("Volume_Liters", "sum"),
            n_distributors=("Distributor_ID", "nunique"),
            n_sku=("SKU_ID", "nunique"),
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

        log.info(f"Aggregated data: {df_agg.shape[0]:,} rows")
        log.info(f"Unique outlets: {df_agg['Outlet_ID'].nunique():,}")
        log.info(f"Years: {sorted(df_agg['Year'].unique())}")

        return df_wide, df_agg

    def train(
        self,
        df_wide: pd.DataFrame,
        df_agg: pd.DataFrame,
        train_years: list = None,
        val_years: list = None,
    ) -> dict:
        """
        Train the latent demand model.

        Args:
            df_wide: Disaggregated transaction data
            df_agg: Aggregated outlet-month data
            train_years: Years to use for training
            val_years: Years to use for validation

        Returns:
            Dictionary with training metrics
        """
        log.info("=" * 70)
        log.info("TRAINING LATENT DEMAND MODEL")
        log.info("=" * 70)

        if train_years is None:
            train_years = [2023, 2024]
        if val_years is None:
            val_years = [2025]

        log.info(f"Training years: {train_years}")
        log.info(f"Validation years: {val_years}")

        # Train model
        self.train_metrics = self.model.fit(
            df_wide, df_agg,
            train_years=train_years,
            val_years=val_years,
        )

        log.info("\n" + "=" * 70)
        log.info("TRAINING METRICS")
        log.info("=" * 70)
        for metric_name, metric_value in self.train_metrics.items():
            log.info(f"  {metric_name.upper()}: {metric_value:.4f}")

        return self.train_metrics

    def predict_for_period(
        self,
        df_wide: pd.DataFrame,
        df_agg: pd.DataFrame,
        target_year: int = 2026,
        target_month: int = 1,
    ) -> pd.DataFrame:
        """
        Generate predictions for a specific period.

        Args:
            df_wide: Latest disaggregated data
            df_agg: Latest aggregated data
            target_year: Year to predict
            target_month: Month to predict

        Returns:
            DataFrame with outlet-level predictions and confidence intervals
        """
        log.info("=" * 70)
        log.info(f"GENERATING PREDICTIONS FOR {target_year}-{target_month:02d}")
        log.info("=" * 70)

        # Get latest observations per outlet x distributor x SKU
        latest_wide = (
            df_wide.sort_values(["Outlet_ID", "Distributor_ID", "SKU_ID", "Year", "Month"])
            .groupby(["Outlet_ID", "Distributor_ID", "SKU_ID"])
            .last()
            .reset_index()
        )

        # Create prediction data for target period
        pred_data = latest_wide.copy()
        pred_data["Year"] = target_year
        pred_data["Month"] = target_month
        pred_data["month_sin"] = np.sin(2 * np.pi * target_month / 12)
        pred_data["month_cos"] = np.cos(2 * np.pi * target_month / 12)

        # Get latest outlet features
        latest_agg = df_agg.sort_values(["Outlet_ID", "Year", "Month"]).groupby("Outlet_ID").last().reset_index()

        outlet_cols = [
            "Outlet_Size", "Outlet_Type", "Cooler_Count", "Outlet_Size_Imputed",
            "Seasonality_Index", "holiday_count", "school_score", "hospital_score",
            "bus_stop_score", "tourist_score", "Latitude", "Longitude",
        ]
        for col in outlet_cols:
            if col in latest_agg.columns and col not in pred_data.columns:
                pred_data[col] = pred_data["Outlet_ID"].map(latest_agg.set_index("Outlet_ID")[col])

        # Ensure outlet_vol_mean is available
        if "outlet_vol_mean" not in pred_data.columns:
            outlet_means = self.model.outlet_means or latest_agg.groupby("Outlet_ID")["Volume_Liters"].mean().to_dict()
            pred_data["outlet_vol_mean"] = pred_data["Outlet_ID"].map(outlet_means)

        log.info(f"Prediction data shape: {pred_data.shape}")
        log.info(f"Unique outlets: {pred_data['Outlet_ID'].nunique():,}")

        # Generate predictions
        predictions = self.model.predict(pred_data)
        pred_data["predicted_volume"] = predictions

        # Aggregate to outlet level
        outlet_preds = pred_data.groupby("Outlet_ID").agg(
            predicted_volume=("predicted_volume", "sum"),
            outlet_vol_mean=("outlet_vol_mean", "first"),
            Latitude=("Latitude", "first"),
            Longitude=("Longitude", "first"),
        ).reset_index()

        log.info(f"Aggregated predictions for {len(outlet_preds):,} outlets")
        log.info(f"Total predicted volume: {outlet_preds['predicted_volume'].sum():,.0f} L")
        log.info(f"Mean per outlet: {outlet_preds['predicted_volume'].mean():.1f} L")
        log.info(f"Median per outlet: {outlet_preds['predicted_volume'].median():.1f} L")

        return outlet_preds

    def save_model(self) -> None:
        """Save trained model to disk."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model.save(self.model_dir)
        log.info(f"Model saved to {self.model_dir}")

    def load_model(self) -> None:
        """Load trained model from disk."""
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        self.model.load(self.model_dir)
        log.info(f"Model loaded from {self.model_dir}")


def main():
    """Main execution."""
    start_time = datetime.now()

    try:
        # Initialize trainer
        trainer = LatentDemandTrainer()

        # Load datasets
        df_wide, df_agg = trainer.load_and_prepare_datasets()

        # Train model
        metrics = trainer.train(df_wide, df_agg)

        # Save model
        trainer.save_model()

        # Generate predictions for January 2026
        predictions = trainer.predict_for_period(
            df_wide, df_agg,
            target_year=2026,
            target_month=1,
        )

        # Save predictions
        output_path = config.GOLD_PATH / "latent_demand_predictions_jan2026.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_parquet(output_path, index=False)
        log.info(f"Predictions saved to {output_path}")

        # Summary
        log.info("\n" + "=" * 70)
        log.info("LATENT DEMAND MODEL - TRAINING COMPLETE")
        log.info("=" * 70)
        elapsed = (datetime.now() - start_time).total_seconds()
        log.info(f"Total time: {elapsed:.1f} seconds")
        log.info(f"Model directory: {trainer.model_dir}")
        log.info(f"Predictions output: {output_path}")
        log.info("=" * 70)

    except Exception as e:
        log.exception("Training pipeline failed: %s", e)
        raise


if __name__ == "__main__":
    main()

