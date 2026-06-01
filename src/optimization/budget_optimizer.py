# Budget Optimizer — allocate promotional budget across Western Province outlets

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, value

from src.configs.config import config

log = logging.getLogger("pipeline.budget_optimizer")

WESTERN_DISTRIBUTORS = ["DIST_W_01", "DIST_W_02", "DIST_W_03"]
DEFAULT_BUDGET_LKR = 5_000_000
DEFAULT_COST_PER_LITER = 50.0


class BudgetOptimizer:
    def __init__(
        self,
        budget_lkr: float = DEFAULT_BUDGET_LKR,
        cost_per_liter: float = DEFAULT_COST_PER_LITER,
    ):
        self.budget_lkr = budget_lkr
        self.cost_per_liter = cost_per_liter

    def run(
        self,
        predictions_path: Path | str = Path("notebooks/predictions_jan2026.parquet"),
    ) -> pd.DataFrame:
        log.info(
            "[BUDGET] Starting optimization (budget=LKR %.0f, cost_per_liter=LKR %.0f)",
            self.budget_lkr,
            self.cost_per_liter,
        )

        tx = pd.read_parquet(config.GOLD_PATH / "fact_table" / "data.parquet")
        preds = pd.read_parquet(predictions_path)

        western = tx[tx["Distributor_ID"].isin(WESTERN_DISTRIBUTORS)]
        hist_mean = (
            western.groupby("Outlet_ID")["Volume_Liters"]
            .mean()
            .reset_index(name="historical_mean")
        )

        outlets = preds.merge(hist_mean, on="Outlet_ID", how="inner")
        outlets["potential"] = (
            outlets["predicted_volume"] - outlets["historical_mean"]
        ).clip(lower=0)

        log.info(
            "[BUDGET] %d Western Province outlets with positive potential",
            (outlets["potential"] > 0).sum(),
        )

        prob = LpProblem("TradeSpendOptimization", LpMaximize)

        spend_vars = {
            row["Outlet_ID"]: LpVariable(
                f"spend_{row['Outlet_ID']}", lowBound=0, cat="Continuous"
            )
            for _, row in outlets.iterrows()
        }

        incr_vol_expr = []
        for _, row in outlets.iterrows():
            oid = row["Outlet_ID"]
            max_vol = row["potential"]
            spend = spend_vars[oid]
            incr = spend / self.cost_per_liter
            incr_vol_expr.append(incr)
            prob += incr <= max_vol, f"max_vol_{oid}"

        prob += lpSum(spend_vars.values()) <= self.budget_lkr, "budget_constraint"
        prob += lpSum(incr_vol_expr), "total_incremental_volume"

        prob.solve()

        results = []
        total_spend = 0.0
        total_incr = 0.0
        for _, row in outlets.iterrows():
            oid = row["Outlet_ID"]
            spend_val = value(spend_vars[oid])
            incr_val = min(spend_val / self.cost_per_liter, row["potential"])
            total_spend += spend_val
            total_incr += incr_val
            results.append(
                {
                    "Outlet_ID": oid,
                    "Trade_Spend_LKR": round(spend_val, 2),
                    "predicted_volume": round(row["predicted_volume"], 2),
                    "historical_mean": round(row["historical_mean"], 2),
                    "incremental_volume": round(incr_val, 2),
                }
            )

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values("Trade_Spend_LKR", ascending=False).reset_index(drop=True)

        output_path = config.REPORTS_DIR / "teamname_budget_allocations.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)

        log.info("[BUDGET] Optimization complete:")
        log.info("[BUDGET]   Total spend: LKR %.2f / %.2f", total_spend, self.budget_lkr)
        log.info("[BUDGET]   Outlets funded: %d", (results_df["Trade_Spend_LKR"] > 0).sum())
        log.info("[BUDGET]   Total incremental volume: %.0f L", total_incr)
        log.info("[BUDGET]   Output: %s", output_path)

        return results_df
