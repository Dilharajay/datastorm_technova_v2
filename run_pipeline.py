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

        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        log.info("=" * 55)
        log.info("Pipeline finished in %.1f seconds", elapsed)
        log.info("  Bronze : %s", list(bronze_results.keys()))
        log.info("  Silver : %s", list(silver_results.keys()))
        log.info("  Gold   : %s", list(gold_results.keys()))
        log.info("=" * 55)

    except Exception as exc:
        log.exception("Pipeline failed: %s", exc)
        raise


if __name__ == "__main__":
    main()