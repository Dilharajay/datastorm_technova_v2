"""
Advanced model training script with censoring handling.

Trains Hurdle, XGBoost, LightGBM, Tobit, and Ensemble models on transaction data,
then generates predictions for potential estimation.
"""
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.configs.config import config
from src.utils.io import read_parquet, write_parquet
from src.prediction.advanced_models import (
    EnsembleModel,
    HurdleModel,
    XGBoostCensoringModel,
    LightGBMCensoringModel,
    TobitLikeModel,
    FeatureEngineering,
    CensoringHandler,
    ModelConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-8s]  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("advanced_prediction.trainer")


class AdvancedModelTrainer:
    """Trains and manages advanced ML models for volume prediction."""

    # Features to use for modeling
    FEATURE_COLS = [
        "Year", "Month", "Cooler_Count", "holiday_count",
        "school_score", "hospital_score", "bus_stop_score", "tourist_score"
    ]

    CATEGORICAL_COLS = ["Outlet_Size", "Outlet_Type", "Seasonality_Index"]

    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = data_path or config.GOLD_PATH
        self.models = {}
        self.feature_names = None
        self.categorical_encoders = {}

    def load_and_prepare_data(
        self,
        use_bronze: bool = False,
        train_split: float = 0.8,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Load data and prepare for modeling.

        Args:
            use_bronze: If True, use raw bronze data to capture censoring; else use cleaned gold
            train_split: Train/test split ratio

        Returns:
            (X_train, X_test, y_train, y_test)
        """
        if use_bronze:
            log.info("Loading raw bronze transactions (with censoring)")
            # Load raw data to see zero/negative volumes
            tx = read_parquet(config.BRONZE_PATH, "transactions_history_final")

            # Join with outlet features
            out = read_parquet(config.SILVER_PATH, "outlet_master")
            dist = read_parquet(config.SILVER_PATH, "distributor_seasonality_details")
            hol = read_parquet(config.SILVER_PATH, "holiday_list")

            df = tx.merge(out[["Outlet_ID", "Outlet_Size", "Outlet_Type", "Cooler_Count"]],
                         on="Outlet_ID", how="left")
            df = df.merge(dist[["Distributor_ID", "Year", "Month", "Seasonality_Index"]],
                         on=["Distributor_ID", "Year", "Month"], how="left")

            hol_agg = hol.groupby(["Year", "Month"]).size().reset_index(name="holiday_count")
            df = df.merge(hol_agg, on=["Year", "Month"], how="left")
            df["holiday_count"] = df["holiday_count"].fillna(0)

            # Try to load POI scores
            try:
                poi = read_parquet(config.SILVER_PATH, "outlet_poi_scores")
                df = df.merge(poi, on="Outlet_ID", how="left")
            except:
                log.warning("POI scores not available")
                df["school_score"] = 0.0
                df["hospital_score"] = 0.0
                df["bus_stop_score"] = 0.0
                df["tourist_score"] = 0.0
        else:
            log.info("Loading cleaned gold fact table")
            df = read_parquet(self.data_path, "fact_table")

        # Fill NaN values in numeric columns
        numeric_cols = [c for c in self.FEATURE_COLS if c in df.columns]
        for col in numeric_cols:
            if df[col].dtype in [np.float64, np.int64]:
                df[col] = df[col].fillna(df[col].median())

        # Fill categorical NaN with 'missing'
        for col in self.CATEGORICAL_COLS:
            if col in df.columns:
                df[col] = df[col].fillna("missing")

        log.info(f"Loaded {len(df)} records with shape {df.shape}")
        log.info(f"Volume stats: min={df['Volume_Liters'].min():.2f}, "
                f"max={df['Volume_Liters'].max():.2f}, "
                f"mean={df['Volume_Liters'].mean():.2f}")

        # Select features
        available_features = [c for c in self.FEATURE_COLS + self.CATEGORICAL_COLS
                             if c in df.columns]
        self.feature_names = available_features

        X = df[available_features].copy()
        y = df["Volume_Liters"].copy()

        # Encode categorical variables ONCE
        from sklearn.preprocessing import LabelEncoder
        self.label_encoders = {}
        for col in self.CATEGORICAL_COLS:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str).fillna("missing"))
                self.label_encoders[col] = le

        # Identify censoring
        log.info(f"Censoring info: {(y <= 0).sum()} censored, {(y > 0).sum()} observed")

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=1-train_split, random_state=42
        )

        log.info(f"Train set: {len(X_train)} samples")
        log.info(f"Test set: {len(X_test)} samples")

        return X_train, X_test, y_train, y_test

    def train_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        models_to_train: Optional[list[str]] = None,
    ) -> dict:
        """
        Train all or selected models.

        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data (for validation)
            models_to_train: List of model names to train. If None, trains all.

        Returns:
            Dictionary of trained models
        """
        if models_to_train is None:
            models_to_train = ["hurdle", "xgboost", "lightgbm", "tobit", "ensemble"]

        config_obj = ModelConfig(random_state=42, verbose=True)
        results = {}

        # Train Hurdle Model
        if "hurdle" in models_to_train:
            log.info("\n" + "="*60)
            log.info("Training Hurdle Model")
            log.info("="*60)
            try:
                hurdle = HurdleModel(config_obj)
                # For hurdle model, use raw categorical cols since it handles encoding
                hurdle.fit(
                    X_train, y_train,
                    categorical_cols=self.CATEGORICAL_COLS,
                    numeric_cols=[c for c in self.FEATURE_COLS if c not in self.CATEGORICAL_COLS],
                    validation_data=(X_test, y_test) if len(X_test) > 0 else None,
                )
                self.models["hurdle"] = hurdle
                results["hurdle"] = "✓ Trained"

                # Evaluate
                test_pred = hurdle.predict(X_test)
                test_mape = self._compute_mape(y_test, test_pred)
                log.info(f"Hurdle Model MAPE on test set: {test_mape:.4f}")
                results["hurdle_mape"] = test_mape
            except Exception as e:
                log.error(f"Hurdle model training failed: {e}")
                results["hurdle"] = f"✗ Failed: {e}"

        # Train XGBoost Model
        if "xgboost" in models_to_train:
            log.info("\n" + "="*60)
            log.info("Training XGBoost Censoring Model")
            log.info("="*60)
            try:
                xgb_model = XGBoostCensoringModel(config_obj)
                xgb_model.fit(X_train, y_train)  # X_train already has encoded categoricals
                self.models["xgboost"] = xgb_model
                results["xgboost"] = "✓ Trained"

                # Evaluate
                test_pred = xgb_model.predict(X_test)
                test_mape = self._compute_mape(y_test, test_pred)
                log.info(f"XGBoost MAPE on test set: {test_mape:.4f}")
                results["xgboost_mape"] = test_mape
            except Exception as e:
                log.error(f"XGBoost model training failed: {e}")
                results["xgboost"] = f"✗ Failed: {e}"

        # Train LightGBM Model
        if "lightgbm" in models_to_train:
            log.info("\n" + "="*60)
            log.info("Training LightGBM Censoring Model")
            log.info("="*60)
            try:
                lgb_model = LightGBMCensoringModel(config_obj)
                lgb_model.fit(X_train, y_train)  # X_train already has encoded categoricals
                self.models["lightgbm"] = lgb_model
                results["lightgbm"] = "✓ Trained"

                # Evaluate
                test_pred = lgb_model.predict(X_test)
                test_mape = self._compute_mape(y_test, test_pred)
                log.info(f"LightGBM MAPE on test set: {test_mape:.4f}")
                results["lightgbm_mape"] = test_mape
            except Exception as e:
                log.error(f"LightGBM model training failed: {e}")
                results["lightgbm"] = f"✗ Failed: {e}"

        # Train Tobit Model
        if "tobit" in models_to_train:
            log.info("\n" + "="*60)
            log.info("Training Tobit-like Model")
            log.info("="*60)
            try:
                tobit = TobitLikeModel(config_obj)
                tobit.fit(X_train, y_train)  # X_train already has encoded categoricals
                self.models["tobit"] = tobit
                results["tobit"] = "✓ Trained"

                # Evaluate
                test_pred = tobit.predict(X_test)
                test_mape = self._compute_mape(y_test, test_pred)
                log.info(f"Tobit MAPE on test set: {test_mape:.4f}")
                results["tobit_mape"] = test_mape
            except Exception as e:
                log.error(f"Tobit model training failed: {e}")
                results["tobit"] = f"✗ Failed: {e}"

        # Train Ensemble
        if "ensemble" in models_to_train and len(self.models) > 1:
            log.info("\n" + "="*60)
            log.info("Training Ensemble Model")
            log.info("="*60)
            try:
                ensemble = EnsembleModel(config_obj)
                ensemble.fit(
                    X_train, y_train,
                    categorical_cols=self.CATEGORICAL_COLS,
                    numeric_cols=[c for c in self.FEATURE_COLS if c not in self.CATEGORICAL_COLS],
                    val_split=0.2,
                )
                self.models["ensemble"] = ensemble
                results["ensemble"] = "✓ Trained"

                # Evaluate
                test_pred = ensemble.predict(X_test)
                test_mape = self._compute_mape(y_test, test_pred)
                log.info(f"Ensemble MAPE on test set: {test_mape:.4f}")
                results["ensemble_mape"] = test_mape
            except Exception as e:
                log.error(f"Ensemble model training failed: {e}")
                results["ensemble"] = f"✗ Failed: {e}"

        log.info("\n" + "="*60)
        log.info("Training Summary")
        log.info("="*60)
        for key, value in results.items():
            log.info(f"  {key}: {value}")

        return results

    def generate_predictions(
        self,
        model_name: str = "ensemble",
        aggregate_by: str = "Outlet_ID",
        use_fact_table: bool = True,
    ) -> pd.DataFrame:
        """
        Generate predictions using trained model.

        Args:
            model_name: Name of model to use
            aggregate_by: Column to aggregate predictions (e.g., Outlet_ID for outlet potential)
            use_fact_table: If True, predict on full fact table

        Returns:
            Predictions DataFrame
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not trained. Available: {list(self.models.keys())}")

        model = self.models[model_name]

        if use_fact_table:
            log.info(f"Loading fact table for prediction")
            fact = read_parquet(config.GOLD_PATH, "fact_table")

            # Prepare features
            available_features = [c for c in self.FEATURE_COLS + self.CATEGORICAL_COLS
                                 if c in fact.columns]
            X = fact[available_features].copy()

            # Handle missing values
            for col in available_features:
                if X[col].dtype in [np.float64, np.int64]:
                    X[col] = X[col].fillna(X[col].median())
                else:
                    X[col] = X[col].fillna("missing")

            # Encode categorical variables using the same encoders
            for col in self.CATEGORICAL_COLS:
                if col in X.columns and col in self.label_encoders:
                    X[col] = self.label_encoders[col].transform(X[col].astype(str))

            # Generate predictions
            log.info(f"Generating predictions using {model_name} model")
            predictions = model.predict(X)

            # Attach to fact table
            fact["predicted_volume"] = predictions

            # Aggregate by specified column
            if aggregate_by:
                log.info(f"Aggregating predictions by {aggregate_by}")
                agg_preds = fact.groupby(aggregate_by).agg({
                    "predicted_volume": ["mean", "max", "sum"],
                }).reset_index()
                agg_preds.columns = [aggregate_by, f"{aggregate_by}_avg_potential",
                                     f"{aggregate_by}_max_potential", f"{aggregate_by}_total_potential"]

                return agg_preds
            else:
                return fact[["Outlet_ID", "predicted_volume"]]

        return None

    @staticmethod
    def _compute_mape(y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Compute Mean Absolute Percentage Error."""
        mask = y_true > 0
        if mask.sum() == 0:
            return np.nan
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def main():
    """Main training script."""
    log.info("Advanced Model Training Pipeline")
    log.info("="*60)

    # Initialize trainer
    trainer = AdvancedModelTrainer()

    # Load and prepare data (using raw data to capture censoring)
    X_train, X_test, y_train, y_test = trainer.load_and_prepare_data(
        use_bronze=True,  # Use raw data with zero volumes
        train_split=0.8,
    )

    # Train models
    training_results = trainer.train_models(
        X_train, y_train, X_test, y_test,
        models_to_train=["hurdle", "xgboost", "lightgbm", "tobit", "ensemble"]
    )

    # Generate predictions using ensemble model
    log.info("\n" + "="*60)
    log.info("Generating Outlet Potential Predictions")
    log.info("="*60)

    predictions = trainer.generate_predictions(
        model_name="ensemble",
        aggregate_by="Outlet_ID",
        use_fact_table=True,
    )

    # Save predictions
    output_path = config.GOLD_PATH / "advanced_predictions.csv"
    predictions.to_csv(output_path, index=False)
    log.info(f"Predictions saved to {output_path}")

    log.info("\n" + "="*60)
    log.info("Training Pipeline Completed")
    log.info("="*60)

    return trainer, predictions


if __name__ == "__main__":
    trainer, predictions = main()

