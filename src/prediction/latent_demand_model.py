"""
Latent Demand Estimation Model with Tobit-based censoring correction and XGBoost.

This module implements a two-stage pipeline:
1. Detect and quantify censoring via proxy rules
2. Recover latent demand using Tobit model
3. Train XGBoost on de-censored series for final predictions

Key features:
- Censoring score computed from 6 proxy rules (sudden_drop, ramp_up, cooler_ceiling, etc.)
- Confidence labels (high, medium, low) based on censoring score
- Torch-based Tobit implementation with GPU acceleration
- XGBoost regressor with sample weighting on de-censored data
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from joblib import dump, load
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm

log = logging.getLogger("prediction.latent_demand_model")


class CensoringScoreCalculator:
    """Compute censoring scores using 6 proxy rules."""

    @staticmethod
    def compute_censoring_scores(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute composite censoring score on aggregated outlet-month data.

        Args:
            df: DataFrame with outlet-month aggregation (Outlet_ID, Year, Month, Volume_Liters, etc.)

        Returns:
            DataFrame with added censoring score columns
        """
        d = df.sort_values(["Outlet_ID", "Year", "Month"]).copy()

        # Compute rolling statistics
        d["rolling3_avg"] = d.groupby("Outlet_ID")["Volume_Liters"].transform(
            lambda x: x.rolling(3, min_periods=2).mean().shift(1)
        )
        d["rolling3_std"] = d.groupby("Outlet_ID")["Volume_Liters"].transform(
            lambda x: x.rolling(3, min_periods=2).std().shift(1)
        )

        # Rule 1: Sudden Drop (weight 0.35)
        d["drop_ratio"] = np.where(
            d["rolling3_avg"] > 0,
            d["Volume_Liters"] / d["rolling3_avg"],
            1.0,
        )
        d["sudden_drop_score"] = np.clip(1.0 - d["drop_ratio"] / 0.3, 0, 1)
        d.loc[d["drop_ratio"] >= 0.7, "sudden_drop_score"] = 0.0

        # Rule 2: Ramp-up (weight 0.15)
        d["ramp_up_score"] = np.exp(-d["outlet_age"] / 3)
        d.loc[d["outlet_age"] > 5, "ramp_up_score"] = 0.0

        # Rule 3: Cooler Ceiling (weight 0.15)
        d["cooler_efficiency"] = d["Volume_Liters"] / (d["Cooler_Count"] + 1)
        cooler_pct = d.groupby("Cooler_Count")["cooler_efficiency"].transform(
            lambda x: x.rank(pct=True)
        )
        d["cooler_ceiling_score"] = np.where(cooler_pct >= 0.90, 1.0, 0.0)
        d.loc[d["Cooler_Count"] >= 3, "cooler_ceiling_score"] = 0.0

        # Rule 4: Seasonality Mismatch (weight 0.15)
        d["seasonality_mismatch_score"] = 0.0
        mask = (d["Seasonality_Index"] != "un-favorable") & (d["drop_ratio"] < 0.5)
        d.loc[mask, "seasonality_mismatch_score"] = np.clip(
            1.0 - d.loc[mask, "drop_ratio"] / 0.5, 0, 1
        )

        # Rule 5: High Volatility (weight 0.10)
        d["cv"] = np.where(
            d["rolling3_avg"] > 0,
            d["rolling3_std"] / d["rolling3_avg"],
            0,
        )
        cv_threshold = d.loc[d["rolling3_avg"] > 0, "cv"].quantile(0.90)
        d["volatility_score"] = np.where(
            (d["cv"] >= cv_threshold) & (d["cv"].notna()),
            np.clip((d["cv"] - cv_threshold) / (d["cv"].max() - cv_threshold + 1e-8), 0, 1),
            0.0,
        )

        # Rule 6: Imputed Size (weight 0.10)
        d["imputed_size_score"] = np.where(d["Outlet_Size_Imputed"], 0.3, 0.0)

        # Composite score (weighted sum)
        weights = {
            "sudden_drop_score": 0.35,
            "ramp_up_score": 0.15,
            "cooler_ceiling_score": 0.15,
            "seasonality_mismatch_score": 0.15,
            "volatility_score": 0.10,
            "imputed_size_score": 0.10,
        }
        d["censoring_score"] = sum(d[k] * w for k, w in weights.items())

        # Confidence labels
        d["confidence_label"] = pd.cut(
            d["censoring_score"],
            bins=[-0.01, 0.2, 0.6, 1.01],
            labels=["high", "medium", "low"],
        )

        return d


class TobitModel:
    """Torch-based Tobit model for latent demand recovery."""

    def __init__(self, device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.beta = None
        self.sigma = None
        self.is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        thresholds: np.ndarray,
        learning_rate: float = 1.0,
        max_iter: int = 100,
    ) -> None:
        """
        Fit Tobit model using PyTorch LBFGS optimizer.

        Args:
            X: Feature matrix (already preprocessed)
            y: Observed values (clipped at thresholds)
            thresholds: Censoring thresholds per observation
            learning_rate: LBFGS learning rate
            max_iter: Max iterations
        """
        # Add constant for intercept
        X_sm = sm.add_constant(X, has_constant="add")

        X_tensor = torch.tensor(X_sm, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device)
        censor_tensor = torch.tensor(thresholds, dtype=torch.float32, device=self.device)

        n_features = X_tensor.shape[1]
        beta = torch.randn(n_features, requires_grad=True, device=self.device, dtype=torch.float32)
        log_sigma = torch.zeros(1, requires_grad=True, device=self.device, dtype=torch.float32)

        def tobit_nll_loss(X, y, thresholds, beta, log_sigma):
            sigma = torch.exp(log_sigma)
            mu = X @ beta
            residual = (y - mu) / sigma

            is_censored = y >= (thresholds - 1e-4)
            is_uncensored = ~is_censored

            log_pdf = -log_sigma - 0.5 * np.log(2 * np.pi) - 0.5 * (residual ** 2)
            ll_uncensored = log_pdf[is_uncensored].sum()

            normalized_threshold = (thresholds[is_censored] - mu[is_censored]) / sigma
            survival_prob = 0.5 * torch.special.erfc(normalized_threshold / np.sqrt(2))
            ll_censored = torch.log(torch.clamp(survival_prob, min=1e-12)).sum()

            return -(ll_uncensored + ll_censored)

        optimizer = torch.optim.LBFGS([beta, log_sigma], lr=learning_rate, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            loss = tobit_nll_loss(X_tensor, y_tensor, censor_tensor, beta, log_sigma)
            loss.backward()
            return loss

        optimizer.step(closure)

        self.beta = beta.detach().cpu().numpy()
        self.sigma = torch.exp(log_sigma).detach().cpu().item()
        self.is_fitted = True
        log.info(f"Tobit model fitted. Sigma: {self.sigma:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict latent demand."""
        if self.beta is None:
            raise RuntimeError("Model must be fitted before prediction")

        X_sm = sm.add_constant(X, has_constant="add")
        return X_sm @ self.beta


class LatentDemandModel:
    """Two-stage latent demand estimation model."""

    def __init__(self, n_region_clusters: int = 20, random_state: int = 42):
        self.n_region_clusters = n_region_clusters
        self.random_state = random_state
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.preprocessor = None
        self.tobit_model = None
        self.xgb_model = None
        self.feature_cols = None
        self.num_features = None
        self.cat_features = None
        self.lag_features = None
        self.id_features = None
        self.censoring_stats = {}
        self.outlet_means = None
        self.censor_thresholds = {}
        self.is_fitted = False

    def prepare_data(
        self,
        df_wide: pd.DataFrame,
        df_agg: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare and aggregate data for modeling.

        Args:
            df_wide: Disaggregated transaction data (Outlet_ID, Distributor_ID, SKU_ID, Year, Month, Volume_Liters)
            df_agg: Pre-aggregated outlet-month data (Outlet_ID, Year, Month, Volume_Liters)

        Returns:
            (df_agg_processed, df_wide_processed)
        """
        # Sort and add outlet age
        df_agg = df_agg.sort_values(["Outlet_ID", "Year", "Month"]).reset_index(drop=True)
        df_agg["outlet_age"] = df_agg.groupby("Outlet_ID").cumcount() + 1

        # Compute outlet mean volumes
        outlet_means = df_agg.groupby("Outlet_ID")["Volume_Liters"].agg(["sum", "count"])
        outlet_means["mean"] = outlet_means["sum"] / outlet_means["count"]
        self.outlet_means = outlet_means["mean"].to_dict()

        df_agg["outlet_vol_mean"] = df_agg["Outlet_ID"].map(self.outlet_means)

        # Add cyclic features
        df_wide = df_wide.sort_values(["Outlet_ID", "Distributor_ID", "SKU_ID", "Year", "Month"]).reset_index(drop=True)
        df_wide["outlet_month_idx"] = df_wide.groupby("Outlet_ID").cumcount() + 1
        df_wide["vol_lag_1"] = df_wide.groupby(["Outlet_ID", "Distributor_ID", "SKU_ID"])["Volume_Liters"].shift(1)
        df_wide["vol_lag_12"] = df_wide.groupby(["Outlet_ID", "Distributor_ID", "SKU_ID"])["Volume_Liters"].shift(12)
        df_wide["rolling3_avg"] = (
            df_wide.groupby(["Outlet_ID", "Distributor_ID", "SKU_ID"])["Volume_Liters"]
            .transform(lambda x: x.rolling(3, min_periods=2).mean().shift(1))
        )
        df_wide = df_wide.dropna(subset=["vol_lag_1"]).copy()

        outlet_mean_map = df_agg.groupby("Outlet_ID")["Volume_Liters"].mean().to_dict()
        df_wide["outlet_vol_mean"] = df_wide["Outlet_ID"].map(outlet_mean_map)

        # Compute censoring thresholds
        for outlet_id in df_wide["Outlet_ID"].unique():
            mask = df_wide["Outlet_ID"] == outlet_id
            threshold = df_wide.loc[mask, "Volume_Liters"].quantile(0.95)
            self.censor_thresholds[outlet_id] = max(threshold, 1.0)

        return df_agg, df_wide

    def compute_censoring_scores(self, df_agg: pd.DataFrame) -> pd.DataFrame:
        """Compute censoring scores and confidence labels."""
        df_agg = CensoringScoreCalculator.compute_censoring_scores(df_agg)

        # Store statistics
        self.censoring_stats = {
            "score_dist": df_agg["censoring_score"].describe().to_dict(),
            "label_counts": df_agg["confidence_label"].value_counts().to_dict(),
        }

        return df_agg

    def fit_preprocessor(self, X_train: pd.DataFrame) -> None:
        """Build sklearn preprocessor pipeline."""
        self.num_features = [
            "Cooler_Count", "holiday_count",
            "school_score", "hospital_score", "bus_stop_score", "tourist_score",
            "Latitude", "Longitude",
        ]
        self.cat_features = [
            "Outlet_Size", "Outlet_Type", "Seasonality_Index",
        ]
        self.lag_features = ["vol_lag_1", "vol_lag_12", "rolling3_avg"]
        self.id_features = ["Distributor_ID", "SKU_ID"]

        self.preprocessor = ColumnTransformer([
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), self.num_features),
            ("cat", Pipeline([
                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), self.cat_features),
            ("lag", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), self.lag_features),
            ("id", Pipeline([
                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), self.id_features),
            ("cyclic", "passthrough", ["month_sin", "month_cos", "Year"]),
        ])

        self.feature_cols = (
            self.num_features + self.cat_features + self.lag_features + self.id_features
            + ["month_sin", "month_cos", "Year", "outlet_vol_mean"]
        )

        self.preprocessor.fit(X_train[self.feature_cols])

    def fit(
        self,
        df_wide: pd.DataFrame,
        df_agg: pd.DataFrame,
        train_years: list = None,
        val_years: list = None,
    ) -> Dict[str, Any]:
        """
        Train the two-stage model.

        Args:
            df_wide: Disaggregated data
            df_agg: Aggregated data
            train_years: Years for training (default: [2023, 2024])
            val_years: Years for validation (default: [2025])

        Returns:
            Dictionary with training metrics
        """
        if train_years is None:
            train_years = [2023, 2024]
        if val_years is None:
            val_years = [2025]

        # Prepare data
        df_agg, df_wide = self.prepare_data(df_wide, df_agg)
        df_agg = self.compute_censoring_scores(df_agg)

        # Add cyclic features
        for df in [df_wide, df_agg]:
            df["month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
            df["month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

        # Split
        train_agg = df_agg[df_agg["Year"].isin(train_years)].copy()
        val_agg = df_agg[df_agg["Year"].isin(val_years)].copy()
        train_wide = df_wide[df_wide["Year"].isin(train_years)].copy()
        val_wide = df_wide[df_wide["Year"].isin(val_years)].copy()

        # Merge censoring scores to wide data
        score_cols = ["censoring_score", "confidence_label"]
        train_wide = train_wide.merge(
            train_agg[["Outlet_ID", "Year", "Month"] + score_cols],
            on=["Outlet_ID", "Year", "Month"],
            how="left",
        )
        val_wide = val_wide.merge(
            val_agg[["Outlet_ID", "Year", "Month"] + score_cols],
            on=["Outlet_ID", "Year", "Month"],
            how="left",
        )

        log.info(f"Train: {len(train_wide):,} rows | Val: {len(val_wide):,} rows")

        # Fit preprocessor
        self.fit_preprocessor(train_wide)

        # Prepare features
        X_train = train_wide[self.feature_cols]
        y_train = train_wide["Volume_Liters"].values
        X_val = val_wide[self.feature_cols]
        y_val = val_wide["Volume_Liters"].values

        X_train_processed = self.preprocessor.transform(X_train)
        X_val_processed = self.preprocessor.transform(X_val)

        # Stage 1: Train Tobit on training data
        log.info("=== Stage 1: Tobit Model ===")
        censor_thresholds_train = train_wide["Outlet_ID"].map(self.censor_thresholds).values
        y_train_clipped = np.minimum(y_train, censor_thresholds_train)

        self.tobit_model = TobitModel(device=self.device)
        self.tobit_model.fit(X_train_processed, y_train_clipped, censor_thresholds_train)

        # De-censor training data
        y_latent_train = self.tobit_model.predict(X_train_processed)
        censor_mask_train = train_wide["censoring_score"] > 0.3
        y_train_decensored = y_train.copy()
        y_train_decensored[censor_mask_train] = y_latent_train[censor_mask_train]

        log.info(f"De-censored: {censor_mask_train.sum():,} / {len(y_train):,} rows")
        log.info(f"Mean volume: {y_train.mean():.2f} -> {y_train_decensored.mean():.2f}")

        # Stage 2: Train XGBoost
        log.info("=== Stage 2: XGBoost on De-censored Data ===")
        sample_weights = np.clip(1.0 - train_wide["censoring_score"].values, 0.1, 1.0)

        self.xgb_model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=self.random_state,
            tree_method="hist",
        )
        self.xgb_model.fit(X_train_processed, y_train_decensored, sample_weight=sample_weights)

        # Evaluate on validation
        y_pred_val = self.xgb_model.predict(X_val_processed)

        metrics = {
            "mae": mean_absolute_error(y_val, y_pred_val),
            "rmse": np.sqrt(mean_squared_error(y_val, y_pred_val)),
            "bias": np.mean(y_pred_val - y_val),
            "r2": r2_score(y_val, y_pred_val),
        }

        log.info(f"Validation MAE:  {metrics['mae']:.2f}")
        log.info(f"Validation RMSE: {metrics['rmse']:.2f}")
        log.info(f"Validation Bias: {metrics['bias']:.2f}")
        log.info(f"Validation R2:   {metrics['r2']:.4f}")

        self.is_fitted = True
        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        X_processed = self.preprocessor.transform(X[self.feature_cols])
        return self.xgb_model.predict(X_processed)

    def save(self, path: Path) -> None:
        """Save model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save XGBoost model
        self.xgb_model.save_model(str(path / "xgb_model.json"))

        # Save Tobit model
        tobit_data = {
            "beta": self.tobit_model.beta,
            "sigma": self.tobit_model.sigma,
        }
        np.savez(path / "tobit_model.npz", **tobit_data)

        # Save preprocessor and metadata
        dump({
            "preprocessor": self.preprocessor,
            "feature_cols": self.feature_cols,
            "num_features": self.num_features,
            "cat_features": self.cat_features,
            "lag_features": self.lag_features,
            "id_features": self.id_features,
            "outlet_means": self.outlet_means,
            "censor_thresholds": self.censor_thresholds,
            "censoring_stats": self.censoring_stats,
        }, path / "metadata.pkl")

        log.info(f"Model saved to {path}")

    def load(self, path: Path) -> None:
        """Load model from disk."""
        path = Path(path)

        # Load XGBoost model
        self.xgb_model = xgb.XGBRegressor()
        self.xgb_model.load_model(str(path / "xgb_model.json"))

        # Load Tobit model
        tobit_data = np.load(path / "tobit_model.npz")
        self.tobit_model = TobitModel(device=self.device)
        self.tobit_model.beta = tobit_data["beta"]
        self.tobit_model.sigma = tobit_data["sigma"].item()
        self.tobit_model.is_fitted = True

        # Load metadata
        metadata = load(path / "metadata.pkl")
        self.preprocessor = metadata["preprocessor"]
        self.feature_cols = metadata["feature_cols"]
        self.num_features = metadata["num_features"]
        self.cat_features = metadata["cat_features"]
        self.lag_features = metadata["lag_features"]
        self.id_features = metadata["id_features"]
        self.outlet_means = metadata["outlet_means"]
        self.censor_thresholds = metadata["censor_thresholds"]
        self.censoring_stats = metadata["censoring_stats"]

        self.is_fitted = True
        log.info(f"Model loaded from {path}")

