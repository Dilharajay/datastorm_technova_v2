"""
Quick demo and testing script for Latent Demand Model.

This script demonstrates how to use the model and can be used to verify
installation and basic functionality.
"""

import logging
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-8s]  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("latent_demand_demo")


def check_dependencies():
    """Verify all required packages are installed."""
    log.info("Checking dependencies...")

    required_packages = {
        "torch": "PyTorch",
        "xgboost": "XGBoost",
        "pandas": "pandas",
        "numpy": "NumPy",
        "sklearn": "scikit-learn",
        "statsmodels": "statsmodels",
    }

    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            log.info(f"  ✓ {name}")
        except ImportError:
            log.error(f"  ✗ {name} NOT FOUND")
            missing.append(name)

    if missing:
        log.error(f"\nMissing packages: {', '.join(missing)}")
        log.error("Install with: pip install torch xgboost")
        return False

    log.info("All dependencies OK!\n")
    return True


def demo_model_creation():
    """Demonstrate model creation."""
    log.info("Creating model instance...")

    from src.prediction.latent_demand_model import LatentDemandModel

    model = LatentDemandModel(n_region_clusters=20, random_state=42)
    log.info(f"  ✓ Model created: {model.__class__.__name__}")
    log.info(f"  Device: {model.device}")

    return model


def demo_censoring_scoring():
    """Demonstrate censoring score calculation."""
    log.info("Demonstrating censoring score calculation...")

    import pandas as pd
    import numpy as np
    from src.prediction.latent_demand_model import CensoringScoreCalculator

    # Create dummy data
    np.random.seed(42)
    n_rows = 100

    dummy_data = pd.DataFrame({
        "Outlet_ID": np.repeat(range(5), n_rows // 5),
        "Year": 2025,
        "Month": np.tile(range(1, 13), n_rows // 12 + 1)[:n_rows],
        "Volume_Liters": np.random.gamma(100, 5, n_rows),
        "Cooler_Count": np.random.randint(0, 4, n_rows),
        "Outlet_Size_Imputed": np.random.choice([True, False], n_rows),
        "Seasonality_Index": np.random.choice(["favorable", "neutral", "un-favorable"], n_rows),
    })

    # Add outlet_age
    dummy_data = dummy_data.sort_values(["Outlet_ID", "Year", "Month"]).reset_index(drop=True)
    dummy_data["outlet_age"] = dummy_data.groupby("Outlet_ID").cumcount() + 1

    # Compute scores
    scored_data = CensoringScoreCalculator.compute_censoring_scores(dummy_data)

    log.info(f"  Input rows: {len(dummy_data)}")
    log.info(f"  Output rows: {len(scored_data)}")
    log.info(f"\n  Censoring score distribution:")
    for label in ["high", "medium", "low"]:
        count = (scored_data["confidence_label"] == label).sum()
        pct = count / len(scored_data) * 100
        log.info(f"    {label.capitalize()}: {count:>3} ({pct:>5.1f}%)")

    return scored_data


def demo_tobit_model():
    """Demonstrate Tobit model."""
    log.info("Demonstrating Tobit model...")

    import numpy as np
    from src.prediction.latent_demand_model import TobitModel

    # Create dummy data
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 5)  # 5 features
    true_latent = X @ np.array([1.0, 2.0, 0.5, -0.5, 1.5])
    thresholds = np.percentile(true_latent, 70) * np.ones(n_samples)
    y = np.minimum(true_latent, thresholds)  # Censored

    # Fit model
    model = TobitModel(device="cpu")
    log.info("  Fitting Tobit model...")
    model.fit(X, y, thresholds, learning_rate=1.0, max_iter=50)
    log.info(f"  ✓ Tobit model fitted")
    log.info(f"    Sigma: {model.sigma:.4f}")

    # Make predictions
    predictions = model.predict(X[:10])
    log.info(f"  ✓ Generated {len(predictions)} predictions")
    log.info(f"    Prediction range: [{predictions.min():.2f}, {predictions.max():.2f}]")

    return model


def main():
    """Run all demos."""
    log.info("=" * 70)
    log.info("LATENT DEMAND MODEL - FUNCTIONALITY CHECK")
    log.info("=" * 70 + "\n")

    try:
        # Check dependencies
        if not check_dependencies():
            return False

        # Demo model creation
        model = demo_model_creation()
        log.info("")

        # Demo censoring scoring
        scored_data = demo_censoring_scoring()
        log.info("")

        # Demo Tobit model
        tobit_model = demo_tobit_model()
        log.info("")

        # Summary
        log.info("=" * 70)
        log.info("✓ ALL CHECKS PASSED")
        log.info("=" * 70)

        return True

    except Exception as e:
        log.exception("Demo failed: %s", e)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)



