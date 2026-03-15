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


@lru_cache(maxsize=4)
def _load_json(path: str) -> dict:
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text())
    return {}


@lru_cache(maxsize=1)
def _load_caption_index() -> Dict[str, str]:
    """Load fixed imagery captions indexed by tile_id."""
    data = _load_json("data/processed/imagery_captions_fixed.json")
    return {c["tile_id"]: c["caption"] for c in data.get("captions", [])}


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



def compute_evidence_floors(zip_code: str) -> dict:
    """Compute minimum flood floor from rainfall only.

    Harvey 2017 NFIP claims removed — they are post-event data (filed weeks
    after Harvey) unavailable in a real-time disaster response system.
    Rainfall from HCFCD gauges is observable in near-real-time.
    """
    flood_floor = 0.0
    reasons = []

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
        "damage_floor": 0.0,
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

    # sensor_table and kb_summary removed — gauge data has poor spatial coverage
    # and post-flood timestamps; FEMA KB only covers 2010. Both signals are
    # superseded by spatial_priors (rainfall_peak_by_zip + historical NFIP profile).
    sensor_table = ""
    kb_summary = ""

    # Spatial priors: rainfall (real-time) + historical NFIP risk profile (pre-event)
    # Harvey 2017 NFIP claims excluded — post-event data, not available in real-time.
    spatial_priors: List[str] = []
    if zip_code:
        rain_str = _get_rainfall_prior(zip_code)
        if rain_str:
            spatial_priors.append(rain_str)
        nfip_str = _get_nfip_prior(zip_code)
        if nfip_str:
            spatial_priors.append(nfip_str)

    floors = compute_evidence_floors(zip_code) if zip_code else {"flood_floor": 0.0, "damage_floor": 0.0, "reasons": []}

    # Attach captions for retrieved imagery tiles
    caption_index = _load_caption_index()
    captions: List[dict] = []
    for tile in imagery:
        tile_id = tile.get("tile_id", "")
        caption_text = caption_index.get(tile_id)
        if caption_text:
            captions.append({"tile_id": tile_id, "doc_id": tile_id, "text": caption_text, "caption": caption_text})

    return {
        "imagery_tiles": imagery,
        "tweets": tweets,
        "calls": calls,
        "sensors": sensors,
        "fema": fema,
        "captions": captions,
        "text_snippets": [snippet for snippet in text_snippets if snippet],
        "sensor_table": sensor_table,
        "kb_summary": kb_summary,
        "spatial_priors": spatial_priors,
        "evidence_floors": floors,
    }
