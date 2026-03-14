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



def _get_harvey_damage_prior(zip_code: str) -> Optional[str]:
    data = _load_json("data/processed/harvey_damage_by_zip.json")
    entry = data.get(str(zip_code))
    if not entry:
        return None
    n = entry.get("harvey_claim_count", 0)
    depth = entry.get("avg_water_depth_ft", 0)
    dmg = entry.get("avg_building_damage_usd", 0)
    paid = entry.get("avg_paid_building_usd", 0)
    return (
        f"Harvey 2017 NFIP claims for ZIP {zip_code} (DIRECT DAMAGE EVIDENCE): "
        f"{n} flood insurance claims filed after Harvey. "
        f"Average reported water depth: {depth:.1f} ft inside buildings. "
        f"Average building damage: ${dmg:,.0f}. Average payout: ${paid:,.0f}. "
        f"Use claim count and water depth to inform damage_severity_pct directly — "
        f"this is Harvey-specific structural damage evidence, not a historical prior."
    )


def compute_evidence_floors(zip_code: str) -> dict:
    """Compute minimum flood/damage floors from Harvey NFIP claims and rainfall.

    These are hard minimum estimates computed from quantitative evidence —
    the LLM must not go below these values.
    """
    flood_floor = 0.0
    damage_floor = 0.0
    reasons = []

    # Harvey NFIP claims → flood + damage floors
    harvey = _load_json("data/processed/harvey_damage_by_zip.json").get(str(zip_code), {})
    n_claims = harvey.get("harvey_claim_count", 0)
    avg_depth = harvey.get("avg_water_depth_ft", 0.0)

    if n_claims > 1500:
        flood_floor = max(flood_floor, 50.0)
        reasons.append(f"{n_claims} Harvey NFIP claims → flood ≥ 50%")
    elif n_claims > 500:
        flood_floor = max(flood_floor, 35.0)
        reasons.append(f"{n_claims} Harvey NFIP claims → flood ≥ 35%")
    elif n_claims > 100:
        flood_floor = max(flood_floor, 20.0)
        reasons.append(f"{n_claims} Harvey NFIP claims → flood ≥ 20%")
    elif n_claims > 0:
        flood_floor = max(flood_floor, 10.0)
        reasons.append(f"{n_claims} Harvey NFIP claims → flood ≥ 10%")

    if avg_depth > 6:
        damage_floor = max(damage_floor, 45.0)
        reasons.append(f"avg water depth {avg_depth:.1f}ft → damage ≥ 45%")
    elif avg_depth > 3:
        damage_floor = max(damage_floor, 25.0)
        reasons.append(f"avg water depth {avg_depth:.1f}ft → damage ≥ 25%")
    elif avg_depth > 1:
        damage_floor = max(damage_floor, 15.0)
        reasons.append(f"avg water depth {avg_depth:.1f}ft → damage ≥ 15%")
    elif avg_depth > 0:
        damage_floor = max(damage_floor, 8.0)
        reasons.append(f"avg water depth {avg_depth:.1f}ft → damage ≥ 8%")

    # Rainfall → additional flood floor
    rain = _load_json("data/processed/rainfall_peak_by_zip.json").get(str(zip_code), {})
    peak_rain = rain.get("peak_rain_in", 0.0)
    if peak_rain >= 40:
        flood_floor = max(flood_floor, 45.0)
        reasons.append(f"{peak_rain:.0f}in rainfall → flood ≥ 45%")
    elif peak_rain >= 30:
        flood_floor = max(flood_floor, 30.0)
        reasons.append(f"{peak_rain:.0f}in rainfall → flood ≥ 30%")
    elif peak_rain >= 20:
        flood_floor = max(flood_floor, 15.0)
        reasons.append(f"{peak_rain:.0f}in rainfall → flood ≥ 15%")

    return {
        "flood_floor": round(flood_floor, 1),
        "damage_floor": round(damage_floor, 1),
        "reasons": reasons,
    }

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
        harvey_str = _get_harvey_damage_prior(zip_code)
        if harvey_str:
            spatial_priors.append(harvey_str)

    floors = compute_evidence_floors(zip_code) if zip_code else {"flood_floor": 0.0, "damage_floor": 0.0, "reasons": []}

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
        "evidence_floors": floors,
    }
