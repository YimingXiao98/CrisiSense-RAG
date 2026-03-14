"""Create structured context for model prompting."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional


def _list_to_markdown(records: List[dict], columns: List[str]) -> str:
    if not records:
        return ""
    header = " | ".join(columns)
    separator = " | ".join(["---"] * len(columns))
    rows = []
    for record in records[:5]:
        rows.append(" | ".join(str(record.get(col, "")) for col in columns))
    return "\n".join([header, separator, *rows])


@lru_cache(maxsize=1)
def _load_json(path: str) -> dict:
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text())
    return {}


def _get_rainfall_prior(zip_code: str) -> Optional[str]:
    data = _load_json("data/processed/rainfall_peak_by_zip.json")
    entry = data.get(str(zip_code))
    if not entry:
        return None
    rain = entry.get("peak_rain_in", 0)
    dist = entry.get("nearest_dist_km", "?")
    n = entry.get("n_gages_used", 1)
    return (
        f"Estimated peak cumulative rainfall (Aug 25-Sep 1): {rain:.1f} inches "
        f"(interpolated from {n} HCFCD gauge station(s), nearest at {dist} km)."
    )


def _get_nfip_prior(zip_code: str) -> Optional[str]:
    data = _load_json("data/processed/harvey_claims_by_zip.json")
    entry = data.get(str(zip_code))
    if not entry:
        return None
    n = entry.get("claim_count", 0)
    years = entry.get("years_span", 1)
    annual = entry.get("avg_annual_claims", 0)
    avg = entry.get("avg_building_payout", 0)
    depth = entry.get("median_water_depth_ft", 0)
    tier = entry.get("risk_tier", "unknown")
    return (
        f"Historical flood risk profile for ZIP {zip_code} (pre-event NFIP data, 1978–2017): "
        f"{n} flood insurance claims over {years} years (~{annual:.0f} claims/yr). "
        f"Avg historical payout: ${avg:,.0f}/claim. "
        f"Median reported water depth in past floods: {depth:.1f} ft. "
        f"Risk tier: {tier.upper()}. "
        f"Use this as a vulnerability prior only — it reflects long-term flood exposure, "
        f"NOT Harvey-specific damage. Do not use this to infer Harvey damage_severity_pct directly."
    )


def package_context(candidates: Dict[str, object]) -> Dict[str, object]:
    imagery = candidates.get("imagery", [])
    tweets: List[dict] = candidates.get("tweets", [])
    calls: List[dict] = candidates.get("calls", [])
    sensors: List[dict] = candidates.get("sensors", [])
    fema: List[dict] = candidates.get("fema", [])
    zip_code: str = str(candidates.get("zip_code", ""))

    text_snippets: List[str] = []
    text_snippets.extend((tweet.get("text") or "")[:400] for tweet in tweets)
    text_snippets.extend((call.get("description") or "")[:400] for call in calls)

    sensor_table = _list_to_markdown(
        sensors, ["sensor_id", "timestamp", "value", "unit"]) if sensors else ""
    kb_summary = _list_to_markdown(fema, ["year", "loss_mean"]) if fema else ""

    # Spatial priors: rainfall interpolation + NFIP claims
    spatial_priors: List[str] = []
    if zip_code:
        rain_str = _get_rainfall_prior(zip_code)
        if rain_str:
            spatial_priors.append(rain_str)
        nfip_str = _get_nfip_prior(zip_code)
        if nfip_str:
            spatial_priors.append(nfip_str)

    return {
        "imagery_tiles": imagery,
        "tweets": tweets,
        "calls": calls,
        "sensors": sensors,
        "fema": fema,
        "text_snippets": [snippet for snippet in text_snippets if snippet],
        "sensor_table": sensor_table,
        "kb_summary": kb_summary,
        "spatial_priors": spatial_priors,
    }
