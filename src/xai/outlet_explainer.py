# XAI Module — generate human-readable explanations for outlet predictions
# Uses Google Gemini API with graceful fallback to template-based explanations

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from src.configs.config import config

log = logging.getLogger("pipeline.xai")

DEFAULT_MODEL = "llama-3.3-70b-versatile"
EXPLANATION_COLS = [
    "Outlet_ID", "predicted_volume", "confidence_label",
    "key_drivers", "local_signals", "operational_constraints",
    "narrative",
]


class OutletExplainer:
    def __init__(self, model_name: str = DEFAULT_MODEL, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")

    def _build_context(self, row: pd.Series, feature_importance: Optional[dict] = None) -> str:
        top_drivers = ""
        if feature_importance:
            sorted_feats = sorted(feature_importance.items(), key=lambda x: -x[1])[:5]
            top_drivers = "Top model drivers: " + ", ".join(
                f"{k} ({v:.3f})" for k, v in sorted_feats
            )

        context = f"""
Outlet {row.get('Outlet_ID', 'N/A')}:
- Predicted volume: {row.get('predicted_volume', 'N/A'):.1f} L/month
- Confidence: {row.get('confidence_label', 'N/A')}
- Censoring score: {row.get('censoring_score', 'N/A'):.3f}
- Outlet type: {row.get('Outlet_Type', 'N/A')}
- Outlet size: {row.get('Outlet_Size', 'N/A')}
- Cooler count: {row.get('Cooler_Count', 'N/A')}
- Competition density: {row.get('competition_density', 'N/A')} competitors within 5km
- POI scores (school/hospital/bus/tourist): {row.get('school_score', 'N/A')} / {row.get('hospital_score', 'N/A')} / {row.get('bus_stop_score', 'N/A')} / {row.get('tourist_score', 'N/A')}
- Holiday count (month): {row.get('holiday_count', 'N/A')}
- Seasonality: {row.get('Seasonality_Index', 'N/A')}
{top_drivers}
"""

        return context.strip()

    def _build_prompt(self, context: str) -> str:
        return f"""You are a business analyst explaining sales predictions to a non-technical audience.

Given the following data about a retail outlet, write a short business explanation (2-3 sentences) covering:
1. Why the model gave this specific sales prediction
2. Which factors increased or decreased the score
3. How local environment and constraints influenced the result

Keep it simple, no technical jargon.

DATA:
{context}

EXPLANATION:"""

    def _load_api_key(self) -> Optional[str]:
        if self.api_key:
            return self.api_key
        key = os.environ.get("GROQ_API_KEY")
        if key:
            return key
        try:
            from dotenv import load_dotenv
            load_dotenv()
            key = os.environ.get("GROQ_API_KEY")
            if key:
                return key
        except ImportError:
            pass
        return None

    def _call_llm(self, prompt: str) -> Optional[str]:
        key = self._load_api_key()
        if not key:
            log.warning("[XAI] No GROQ_API_KEY set; using template fallback")
            return None
        self.api_key = key
        try:
            from groq import Groq
            from tenacity import retry, stop_after_attempt, wait_exponential

            client = Groq(api_key=self.api_key)

            @retry(
                stop=stop_after_attempt(5),
                wait=wait_exponential(multiplier=2, min=4, max=30),
                reraise=True
            )
            def _do_call():
                return client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}]
                )

            resp = _do_call()
            return resp.choices[0].message.content.strip() or None
        except Exception as exc:
            err_msg = str(exc)
            if "429" in err_msg or "quota" in err_msg.lower():
                log.info("[XAI] Groq quota exceeded after retries; using template fallback")
            else:
                log.warning("[XAI] Groq API call failed: %s", err_msg)
            return None

    def _template_explanation(self, row: pd.Series) -> str:
        vol = row.get("predicted_volume", 0)
        censor = row.get("censoring_score", 0)
        comp = row.get("competition_density", 0)

        parts = [f"This outlet has a predicted monthly potential of {vol:.0f} liters."]
        if censor > 0.2:
            parts.append("Historical data shows signs of supply constraint, meaning true demand may be higher than recorded.")
        if comp > 10:
            parts.append(f"It operates in a competitive area with {comp:.0f} nearby outlets, which may cap individual volume.")
        elif comp == 0:
            parts.append("There are no competing outlets nearby, suggesting a captive market opportunity.")

        return " ".join(parts)

    def explain_outlet(
        self,
        row: pd.Series,
        feature_importance: Optional[dict] = None,
    ) -> dict:
        context = self._build_context(row, feature_importance)
        narrative = self._call_llm(self._build_prompt(context))
        if not narrative:
            narrative = self._template_explanation(row)

        return {
            "Outlet_ID": row.get("Outlet_ID"),
            "predicted_volume": row.get("predicted_volume"),
            "confidence_label": row.get("confidence_label"),
            "narrative": narrative,
        }

    def run(
        self,
        predictions_path: Path | str = Path("notebooks/predictions_jan2026.parquet"),
        feature_importance_path: Optional[Path | str] = None,
        max_outlets: Optional[int] = None,
    ) -> pd.DataFrame:
        log.info("[XAI] Generating outlet explanations")

        fact = pd.read_parquet(config.GOLD_PATH / "fact_table" / "data.parquet")
        preds = pd.read_parquet(predictions_path)

        outlet_data = fact.groupby("Outlet_ID").first().reset_index()
        merged = preds.merge(outlet_data, on="Outlet_ID", how="left", suffixes=("", "_drop"))
        merged = merged.loc[:, ~merged.columns.str.endswith("_drop")]

        feature_importance = None
        if feature_importance_path:
            try:
                fi_df = pd.read_csv(feature_importance_path)
                feature_importance = dict(zip(fi_df["feature"], fi_df["importance"]))
            except Exception as exc:
                log.warning("[XAI] Could not load feature importance: %s", exc)

        if max_outlets:
            merged = merged.head(max_outlets)

        explanations = []
        for _, row in merged.iterrows():
            explanations.append(self.explain_outlet(row, feature_importance))

        result = pd.DataFrame(explanations)

        out_dir = config.REPORTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_dir / "teamname_outlet_explanations.csv", index=False)
        log.info("[XAI] Wrote %d explanations to %s", len(result), out_dir / "teamname_outlet_explanations.csv")

        return result
