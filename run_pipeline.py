"""
Project Entry point. Run from the project root:
"""

import logging
import sys
from datetime import datetime, timezone

from src.configs.config import config
from src.utils.io import ensure_dirs
from src.ingest.ingester import BronzeIngester
from src.cleaning.cleaner import GoldCleaner, SilverCleaner
from src.prediction.latent_demand_pipeline import train_latent_demand_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-8s]  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("pipeline.runner")


def main() -> None:
    start = datetime.now(timezone.utc)
    log.info("Pipeline starting  —  %s", start.isoformat())

    ensure_dirs(*config.managed_dirs())

    try:
        bronze_results = BronzeIngester().run()
        silver_results = SilverCleaner().run()
        gold_results = GoldCleaner().run()

        # Train Latent Demand Model
        log.info("\n" + "=" * 55)
        log.info("Training Latent Demand Estimation Model")
        log.info("=" * 55)

        try:
            latent_results = train_latent_demand_model(
                model_output_dir=config.GOLD_PATH / "latent_demand_model",
                train_years=[2023, 2024],
                val_years=[2025],
                prediction_year=2026,
                prediction_month=1,
            )

            log.info("\n" + "=" * 55)
            log.info("Latent Demand Model - Training Summary")
            log.info("=" * 55)
            log.info("  Model saved: %s", latent_results.get("model_dir"))
            log.info("  Predictions: %s", latent_results.get("predictions_file"))
            log.info("  Outlets: %s", latent_results.get("n_outlets"))
            log.info("  Total volume: %.0f L", latent_results.get("total_predicted_volume", 0))

            metrics = latent_results.get("metrics", {})
            if metrics:
                log.info("  Model Metrics:")
                log.info("    MAE:  %.2f", metrics.get("mae", 0))
                log.info("    RMSE: %.2f", metrics.get("rmse", 0))
                log.info("    Bias: %.2f", metrics.get("bias", 0))
                log.info("    R²:   %.4f", metrics.get("r2", 0))

        except Exception as latent_exc:
            log.warning("Latent Demand Model training failed (pipeline continues): %s", latent_exc)
            latent_results = {"status": "failed", "error": str(latent_exc)}

        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        log.info("=" * 55)
        log.info("Pipeline finished in %.1f seconds", elapsed)
        log.info("  Bronze : %s", list(bronze_results.keys()))
        log.info("  Silver : %s", list(silver_results.keys()))
        log.info("  Gold   : %s", list(gold_results.keys()))
        if latent_results.get("status") != "failed":
            log.info("  Latent Demand Model: ✓ Trained")
        log.info("=" * 55)

    except Exception as exc:
        log.exception("Pipeline failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
