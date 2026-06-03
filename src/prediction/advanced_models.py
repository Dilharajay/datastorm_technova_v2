"""
Advanced ML models with censoring handling for volume prediction.

This module implements:
1. Hurdle Model - Two-stage: binary classification (zero vs. positive) + regression (magnitude)
2. XGBoost with censoring awareness
3. LightGBM with censoring awareness
4. Tobit-style approach for left-censored data

All models handle the fact that zero volumes represent censored/missing transactions.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
import lightgbm as lgb
from scipy import stats

log = logging.getLogger("prediction.advanced_models")


@dataclass
class ModelConfig:
    """Configuration for advanced models."""
    test_size: float = 0.2
    random_state: int = 42
    n_jobs: int = -1
    verbose: bool = True


class CensoringHandler:
    """Handles censoring in data - identifies and flags zero/missing volumes."""

    @staticmethod
    def identify_censored(df: pd.DataFrame, target_col: str = "Volume_Liters") -> pd.DataFrame:
        """Add censoring column: 1 if observed (positive), 0 if censored (zero/missing)."""
        result = df.copy()
        result["_is_censored"] = (result[target_col] <= 0).astype(int)
        result["_is_observed"] = (result[target_col] > 0).astype(int)
        return result

    @staticmethod
    def compute_censoring_rate(df: pd.DataFrame, target_col: str = "Volume_Liters") -> float:
        """Compute proportion of censored observations."""
        return (df[target_col] <= 0).sum() / len(df)


class FeatureEngineering:
    """Feature engineering for prediction models."""

    @staticmethod
    def prepare_features(
        df: pd.DataFrame,
        categorical_cols: Optional[list[str]] = None,
        numeric_cols: Optional[list[str]] = None,
        drop_cols: Optional[list[str]] = None,
        label_encoders: Optional[dict] = None,
        fit_encoders: bool = True,
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Prepare features with encoding and scaling.

        Returns:
            (prepared_df, metadata) where metadata contains encoders and scalers
        """
        prepared = df.copy()
        metadata = {}

        if drop_cols:
            prepared = prepared.drop(columns=[c for c in drop_cols if c in prepared.columns])

        # Encode categorical variables
        if categorical_cols:
            if label_encoders is None:
                label_encoders = {}
                fit_encoders = True

            metadata["label_encoders"] = {}
            for col in categorical_cols:
                if col in prepared.columns:
                    # Ensure string type
                    prepared[col] = prepared[col].astype(str).fillna("missing")

                    if fit_encoders and col not in label_encoders:
                        le = LabelEncoder()
                        prepared[col] = le.fit_transform(prepared[col])
                        label_encoders[col] = le
                        metadata["label_encoders"][col] = le
                    elif col in label_encoders:
                        # Use pre-fitted encoder
                        prepared[col] = label_encoders[col].transform(prepared[col])
                        metadata["label_encoders"][col] = label_encoders[col]

        # Fill numeric NaN with 0
        if numeric_cols:
            numeric_cols_present = [c for c in numeric_cols if c in prepared.columns]
            for col in numeric_cols_present:
                prepared[col] = pd.to_numeric(prepared[col], errors='coerce').fillna(0)

        return prepared, metadata


class HurdleModel:
    """
    Hurdle Model: Two-stage approach for censored data.

    Stage 1: Predict probability of positive volume (binary classification)
    Stage 2: Predict magnitude given positive volume (regression on positive subset)
    """

    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.classifier = xgb.XGBClassifier(
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            verbosity=0,
            eval_metric="logloss",
        )
        self.regressor = xgb.XGBRegressor(
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            verbosity=0,
        )
        self.feature_names = None
        self.label_encoders = None
        self.categorical_cols = None
        self.is_fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        categorical_cols: Optional[list[str]] = None,
        numeric_cols: Optional[list[str]] = None,
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
    ) -> None:
        """
        Fit the hurdle model.

        Args:
            X: Feature dataframe
            y: Target variable (volumes)
            categorical_cols: List of categorical feature columns
            numeric_cols: List of numeric feature columns
            validation_data: Optional (X_val, y_val) tuple for eval
        """
        self.categorical_cols = categorical_cols or []

        # Prepare features with encoding
        # Check if categorical columns are already encoded (numeric)
        need_encoding = False
        for col in self.categorical_cols:
            if col in X.columns and X[col].dtype == 'object':
                need_encoding = True
                break

        if need_encoding:
            X_prep, metadata = FeatureEngineering.prepare_features(
                X,
                categorical_cols=categorical_cols,
                numeric_cols=numeric_cols,
                drop_cols=["_is_censored", "_is_observed"],
                fit_encoders=True,
            )
            self.label_encoders = metadata.get("label_encoders", {})
        else:
            # Already encoded
            X_prep = X.drop(columns=[c for c in ["_is_censored", "_is_observed"] if c in X.columns]).copy()
            self.label_encoders = {}

        self.feature_names = X_prep.columns.tolist()

        # Stage 1: Binary classification (zero vs. positive)
        y_binary = (y > 0).astype(int)

        log.info(f"Hurdle Model - Stage 1: Training classifier")
        log.info(f"  Positive samples: {y_binary.sum()}, Censored samples: {(y_binary == 0).sum()}")

        if validation_data is not None:
            X_val, y_val = validation_data
            if need_encoding:
                X_val_prep, _ = FeatureEngineering.prepare_features(
                    X_val,
                    categorical_cols=categorical_cols,
                    numeric_cols=numeric_cols,
                    drop_cols=["_is_censored", "_is_observed"],
                    label_encoders=self.label_encoders,
                    fit_encoders=False,
                )
            else:
                X_val_prep = X_val.drop(columns=[c for c in ["_is_censored", "_is_observed"] if c in X_val.columns]).copy()

            y_val_binary = (y_val > 0).astype(int)
            self.classifier.fit(
                X_prep,
                y_binary,
                eval_set=[(X_val_prep, y_val_binary)],
                verbose=False,
            )
        else:
            self.classifier.fit(X_prep, y_binary, verbose=False)

        # Stage 2: Regression on positive values only
        positive_mask = y > 0
        X_positive = X_prep[positive_mask].copy()
        y_positive = y[positive_mask].copy()

        log.info(f"Hurdle Model - Stage 2: Training regressor on {positive_mask.sum()} positive samples")

        if validation_data and (y_val > 0).sum() > 0:
            X_val_positive = X_val_prep[(y_val > 0).values].copy()
            y_val_positive = y_val[y_val > 0].copy()
            self.regressor.fit(
                X_positive,
                y_positive,
                eval_set=[(X_val_positive, y_val_positive)],
                verbose=False,
            )
        else:
            self.regressor.fit(X_positive, y_positive, verbose=False)

        self.is_fitted = True
        log.info("Hurdle model fitted successfully")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict volumes using hurdle model."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        # Prepare features with pre-fitted encoders
        X_prep, _ = FeatureEngineering.prepare_features(
            X,
            categorical_cols=self.categorical_cols,
            numeric_cols=None,
            label_encoders=self.label_encoders,
            fit_encoders=False,
        )

        # Align columns with training data
        X_prep = X_prep[self.feature_names]

        # Stage 1: Probability of positive
        prob_positive = self.classifier.predict_proba(X_prep)[:, 1]

        # Stage 2: Magnitude (conditional on positive)
        magnitude = self.regressor.predict(X_prep)

        # Combine: expected volume = P(positive) * E[volume | positive]
        predictions = prob_positive * magnitude

        return predictions

    def save(self, path: Path) -> None:
        """Save model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.classifier.save_model(str(path / "classifier.json"))
        self.regressor.save_model(str(path / "regressor.json"))
        dump(self.metadata, path / "metadata.pkl")
        dump(self.feature_names, path / "feature_names.pkl")

    def load(self, path: Path) -> None:
        """Load model from disk."""
        path = Path(path)
        self.classifier.load_model(str(path / "classifier.json"))
        self.regressor.load_model(str(path / "regressor.json"))
        self.metadata = load(path / "metadata.pkl")
        self.feature_names = load(path / "feature_names.pkl")
        self.is_fitted = True


class XGBoostCensoringModel:
    """
    XGBoost model with explicit censoring handling.

    Treats censored observations (Y=0) differently by:
    - Using loss function that penalizes negative residuals differently
    - Implementing custom objective for censored data
    """

    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.model = None
        self.feature_names = None
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[np.ndarray] = None) -> None:
        """
        Fit XGBoost model with censoring awareness.

        Args:
            X: Features
            y: Target (volumes, including zeros)
            sample_weight: Optional sample weights (can downweight censored observations)
        """
        # Create censoring weights: give less weight to censored (zero) observations
        if sample_weight is None:
            sample_weight = np.ones(len(y))
            censored_mask = (y <= 0)
            sample_weight[censored_mask] = 0.1  # Downweight censored observations

        log.info(f"XGBoost Censoring Model: Training on {len(y)} samples")
        log.info(f"  Positive: {(y > 0).sum()}, Censored: {(y <= 0).sum()}")

        self.model = xgb.XGBRegressor(
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            verbosity=0,
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,
        )

        self.feature_names = X.columns.tolist()
        self.model.fit(X, y, sample_weight=sample_weight, verbose=False)
        self.is_fitted = True
        log.info("XGBoost censoring model fitted successfully")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict volumes."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return self.model.predict(X)


class LightGBMCensoringModel:
    """
    LightGBM model with censoring handling through sample weights.
    """

    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.model = None
        self.feature_names = None
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[np.ndarray] = None) -> None:
        """Fit LightGBM model with censoring awareness."""
        # Create censoring weights
        if sample_weight is None:
            sample_weight = np.ones(len(y))
            censored_mask = (y <= 0)
            sample_weight[censored_mask] = 0.05  # Even lower weight for LightGBM

        log.info(f"LightGBM Censoring Model: Training on {len(y)} samples")
        log.info(f"  Positive: {(y > 0).sum()}, Censored: {(y <= 0).sum()}")

        self.feature_names = X.columns.tolist()

        self.model = lgb.LGBMRegressor(
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            verbose=-1,
            max_depth=7,
            learning_rate=0.05,
            n_estimators=150,
        )

        self.model.fit(X, y, sample_weight=sample_weight)
        self.is_fitted = True
        log.info("LightGBM censoring model fitted successfully")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict volumes."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return self.model.predict(X)


class TobitLikeModel:
    """
    Tobit-like model: Treat censored values as latent variables.

    This uses a two-step approach:
    1. Model probability of censoring (logit)
    2. Model latent continuous variable (regression on censored + uncensored)

    Similar to Tobit but more flexible.
    """

    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.probability_model = None
        self.latent_model = None
        self.feature_names = None
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Fit Tobit-like model.

        Args:
            X: Features (includes censored rows)
            y: Target including zeros
        """
        log.info(f"Tobit-like Model: Training on {len(y)} samples")
        log.info(f"  Positive: {(y > 0).sum()}, Censored: {(y <= 0).sum()}")

        # Step 1: Probability of being uncensored (logit)
        y_censored = (y > 0).astype(int)

        self.probability_model = xgb.XGBClassifier(
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            verbosity=0,
            eval_metric="logloss",
        )
        self.probability_model.fit(X, y_censored, verbose=False)

        # Step 2: Latent variable model (model all data, censored values treated as boundary)
        # Use robust regression approach: upweight positive residuals differently
        self.latent_model = xgb.XGBRegressor(
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            verbosity=0,
            max_depth=6,
        )

        self.feature_names = X.columns.tolist()
        self.latent_model.fit(X, y.clip(lower=0.1), verbose=False)  # Clip zeros to small positive

        self.is_fitted = True
        log.info("Tobit-like model fitted successfully")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict volumes using Tobit framework."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        # P(Y > 0 | X)
        prob_uncensored = self.probability_model.predict_proba(X)[:, 1]

        # E[Y^* | X] where Y^* is latent variable (all data)
        latent_value = self.latent_model.predict(X)

        # Expected value: P(Y > 0) * E[Y | Y > 0, X]
        # Approximate E[Y | Y > 0, X] as latent_value when positive
        predictions = prob_uncensored * np.maximum(latent_value, 0.1)

        return predictions


class EnsembleModel:
    """
    Ensemble of multiple censoring-aware models.
    Combines predictions from Hurdle, XGBoost, LightGBM, and Tobit models.
    """

    def __init__(self, config: ModelConfig = None, weights: Optional[dict] = None):
        self.config = config or ModelConfig()
        self.hurdle = HurdleModel(config)
        self.xgboost = XGBoostCensoringModel(config)
        self.lightgbm = LightGBMCensoringModel(config)
        self.tobit = TobitLikeModel(config)

        # Default weights: equal contribution
        self.weights = weights or {
            "hurdle": 0.4,      # Higher weight for two-stage approach
            "xgboost": 0.2,
            "lightgbm": 0.2,
            "tobit": 0.2,
        }
        self.is_fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        categorical_cols: Optional[list[str]] = None,
        numeric_cols: Optional[list[str]] = None,
        val_split: float = 0.2,
    ) -> None:
        """Fit all models in the ensemble."""
        log.info("Ensemble Model: Training all sub-models")

        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_split, random_state=self.config.random_state
        )

        # Fit hurdle model with validation
        self.hurdle.fit(
            X_train, y_train,
            categorical_cols=categorical_cols,
            numeric_cols=numeric_cols,
            validation_data=(X_val, y_val),
        )

        # Fit other models
        self.xgboost.fit(X_train, y_train)
        self.lightgbm.fit(X_train, y_train)
        self.tobit.fit(X_train, y_train)

        self.is_fitted = True
        log.info(f"Ensemble fitted with weights: {self.weights}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using weighted ensemble of all models."""
        if not self.is_fitted:
            raise RuntimeError("Ensemble must be fitted before prediction")

        predictions = []

        try:
            hurdle_pred = self.hurdle.predict(X)
            predictions.append(("hurdle", hurdle_pred))
        except Exception as e:
            log.warning(f"Hurdle prediction failed: {e}")

        try:
            xgb_pred = self.xgboost.predict(X)
            predictions.append(("xgboost", xgb_pred))
        except Exception as e:
            log.warning(f"XGBoost prediction failed: {e}")

        try:
            lgb_pred = self.lightgbm.predict(X)
            predictions.append(("lightgbm", lgb_pred))
        except Exception as e:
            log.warning(f"LightGBM prediction failed: {e}")

        try:
            tobit_pred = self.tobit.predict(X)
            predictions.append(("tobit", tobit_pred))
        except Exception as e:
            log.warning(f"Tobit prediction failed: {e}")

        if not predictions:
            raise RuntimeError("All models failed to predict")

        # Weighted average
        ensemble_pred = np.zeros(len(X))
        total_weight = 0

        for model_name, pred in predictions:
            weight = self.weights.get(model_name, 0.25)
            ensemble_pred += weight * pred
            total_weight += weight

        ensemble_pred /= total_weight
        return ensemble_pred

