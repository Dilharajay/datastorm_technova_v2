# outlet statistics builder

import pandas as pd
import numpy as np
import logging

from src.configs.config import config
from src.utils.io import read_parquet, write_parquet
from src.utils.eda_utils import compute_outlet_stats

log = logging.getLogger("pipeline.stats_builder")

class StatsBuilder:

    LAYER = "silver"

    def run(self) -> dict[str, int]:

        log.info("Starting stats builder for layer %s", self.LAYER.upper())
        results: dict[str, int] = {}

        try:
            tx = read_parquet(config.SILVER_PATH, "transactions_history_final")
            outlet_stats = compute_outlet_stats(tx)
            rows = write_parquet(outlet_stats, config.SILVER_PATH, "outlet_stats", self.LAYER)
            results["outlet_stats"] = rows
            log.info("Outlet stats computed and written with %d rows", rows)

        except Exception as e:
            log.error("Error computing outlet stats: %s", e)
            results["outlet_stats"] = 0

        return results

