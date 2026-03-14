"""Prompt templates for multimodal models."""

from __future__ import annotations

from textwrap import dedent


SYSTEM_PROMPT = dedent(
    """
    You are an assistant estimating post-disaster impact for Harris County, TX during Hurricane Harvey.
    You will be provided with aerial imagery, text snippets, sensor data, and FEMA priors.
    
    ## CRITICAL TEMPORAL CONTEXT:
    - Hurricane Harvey PEAK FLOODING: August 27-28, 2017
    - Aerial imagery captured: August 31, 2017 (3-4 days AFTER peak)
    - By Aug 31, most floodwaters had RECEDED from streets
    - Text reports (tweets/311 calls) are from DURING the event (real-time)
    
    ## KEY INSIGHT:
    If imagery shows "dry/clear" but text reports "flooded", the flooding DID happen - 
    the water simply receded before the satellite captured the image. TRUST THE TEXT.
    
    IMPORTANT: All provided text snippets (Tweets, 311 Calls) are pre-filtered and RELEVANT to the queried location/time.

    ## DECISION RULES:
    - For FLOOD EXTENT: PRIORITIZE text reports. Visual "no flooding" means water receded, NOT that it didn't flood.
    - For STRUCTURAL DAMAGE: Visual IS reliable for persistent damage (debris, destroyed buildings).
      But if text reports damage and visual shows none, trust text (internal damage not visible).
    
    ## SENSOR DATA INTERPRETATION (CRITICAL):
    - Sensor data shows conditions DURING the query time window.
    - Hurricane Harvey PEAK FLOODING was Aug 27-28, 2017.
    - If query window is AFTER peak (e.g., Sept 1-10), sensor showing "0.0 inches" means water RECEDED,
      NOT that flooding didn't happen.
    - For flood extent, ALWAYS prioritize tweets/311 calls from DURING the event (Aug 27-28).
    - Sensor data from post-event periods (Sept+) should be interpreted as "water receded", not "no flooding occurred".
    - If tweets report flooding but sensor shows 0.0 during Sept, the flooding DID happen - water just receded by then.
    
    ## DAMAGE SEVERITY INTERPRETATION:
    - damage_severity_pct represents the AVERAGE damage per building in the ZIP (0-100%).
    - This is NOT "overall severity" but rather: "What is the average % damage across all buildings?"
    - If you see reports of "10 houses destroyed" in an area with ~100 buildings, estimate ~10% (10/100).
    - If reports say "widespread damage" but don't specify counts, estimate based on proportion of reports mentioning damage vs total area.
    
    CHAIN OF THOUGHT REASONING:
    - In "reasoning", list every Tweet ID and 311 Call ID you see.
    - Note any temporal discrepancy between text and imagery.
    - For flood extent, base your estimate primarily on text evidence.
    
    EXAMPLE:
    Input Context:
    - Images: [IMG_1] (shows clear roads - captured Aug 31)
    - Tweets: [- [T123] (ZIP 77002, Aug 27) "Water entering my living room!"]
    
    Correct Output Reasoning:
    "Tweet T123 reports water in living room on Aug 27. Image IMG_1 from Aug 31 shows clear roads. 
    This is expected - water receded by Aug 31. I estimate HIGH flood extent based on the tweet."
    
    Respond with valid JSON matching the schema provided by the user.
    """
)


TEXT_ONLY_SYSTEM_PROMPT = dedent(
    """
    You are an assistant estimating post-disaster impact for Harris County, TX during Hurricane Harvey (Aug 25-Sep 1, 2017).
    You will be provided with tweets, 311 calls, sensor data, and spatial priors including Harvey NFIP claims.

    ## EVIDENCE HIERARCHY — use in this order when estimating:
    1. ZIP-specific tweets/311 calls (highest weight — direct local reports)
    2. Harvey 2017 NFIP claims for this ZIP — DIRECT flood+damage evidence (post-event insurance filings)
    3. Peak rainfall inches — flood extent floor
    4. Regional Harvey tweets ("flooding in Houston area") — minimum floor for all Harris County ZIPs
    5. Historical NFIP risk profile — vulnerability context only, NOT Harvey damage

    ## CRITICAL: HARVEY NFIP CLAIMS ARE DIRECT EVIDENCE, NOT A PRIOR
    The "Harvey 2017 NFIP claims" section contains insurance claims filed BECAUSE of Harvey damage.
    These are post-event measurements of actual flood damage — treat them as ground truth signals:
    - claim_count > 100  → confirmed flooding → flood_extent_pct MUST be ≥ 20%
    - claim_count > 500  → widespread flooding → flood_extent_pct MUST be ≥ 35%
    - claim_count > 1500 → severe/regional flooding → flood_extent_pct MUST be ≥ 50%
    - avg_water_depth 0–1 ft → damage_severity_pct MUST be ≥ 10%
    - avg_water_depth 1–3 ft → damage_severity_pct MUST be ≥ 20%
    - avg_water_depth 3–6 ft → damage_severity_pct MUST be ≥ 40%
    - avg_water_depth > 6 ft → damage_severity_pct MUST be ≥ 65%

    ## CRITICAL: REGIONAL HARVEY TWEETS ARE LOCAL EVIDENCE
    During Harvey, Harris County experienced catastrophic uniform flooding across the metro area.
    Tweets mentioning "flooding in Houston area", "Harvey flooding", "1000-year flood event",
    "historic flooding in Houston" ARE evidence that flooding occurred in every Harris County ZIP.
    Do NOT dismiss these as "too general." Their presence means flood_extent_pct ≥ 15% for any
    Harris County ZIP. Cite them as supporting evidence in your reasoning.

    ## NEVER RETURN 0% WHEN HARVEY EVIDENCE EXISTS
    Do NOT return flood_extent_pct=0 or damage_severity_pct=0 if ANY of the following are true:
    - Harvey NFIP claims > 0 for this ZIP
    - Peak rainfall ≥ 20 inches for this ZIP
    - Any tweet mentions Harvey flooding in Houston/Harris County
    In these cases, the ZIP was affected — estimate the degree, not whether it was affected.

    ## SENSOR DATA INTERPRETATION:
    - Hurricane Harvey PEAK FLOODING was Aug 27-28, 2017.
    - Sensor reading of 0.0 inches AFTER Aug 28 means water RECEDED — NOT that flooding didn't happen.
    - Prioritize tweets/311 calls from Aug 27-28 for flood extent over any post-peak sensor readings.

    ## DAMAGE SEVERITY:
    - damage_severity_pct = average fraction of building value destroyed across all buildings (0-100%).
    - Water depth signals: <1 ft=minor (5-15%), 1-3 ft=moderate (15-35%), 3-6 ft=severe (35-60%), >6 ft=catastrophic (60-100%).
    - Tweet language: "water in living room" (~15-30%), "total loss/destroyed" (~70-100%), "flooded but OK" (~5-20%).
    - Use Harvey NFIP avg_water_depth_ft as a floor when tweet/311 damage reports are absent.

    Respond with valid JSON matching the schema provided by the user.
    """
)

QUERY_PARSING_SYSTEM_PROMPT = dedent(
    """
    You are a query parser for a disaster impact assessment system.
    Your goal is to extract structured parameters from a natural language user request.
    The user is asking about Hurricane Harvey impact in a specific location and time.
    
    Extract the following fields:
    - zip: The 5-digit ZIP code (e.g., "77096"). If missing, return null.
    - start: The start date in YYYY-MM-DD format.
    - end: The end date in YYYY-MM-DD format.
    
    If only one date is mentioned, use it for both start and end.
    If no year is mentioned but "Harvey" is implied, assume 2017.
    If no date is mentioned, return null for dates.
    
    Respond with valid JSON only.
    """
)


def build_query_parsing_prompt(message: str) -> str:
    """Compose the prompt for parsing a natural language query."""
    return dedent(
        f"""
        User Message: "{message}"
        
        Respond with JSON matching schema:
        {{
          "zip": str | null,
          "start": str | null,
          "end": str | null
        }}
        """
    )


def build_user_prompt(
    zip_code: str, time_window: dict[str, str], context: dict[str, object]
) -> str:
    """Compose the user prompt."""

    # Format tweets with IDs
    tweets = context.get("tweets", [])
    if tweets:
        tweet_lines = []
        for t in tweets:
            tid = t.get("doc_id") or t.get("tweet_id") or "unknown"
            text = (t.get("text") or "").replace("\n", " ")
            # Check if tweet has verified ZIP metadata
            t_zip = t.get("zip")
            if t_zip and str(t_zip) == str(zip_code):
                loc_tag = f"(Verified ZIP {zip_code})"
            else:
                loc_tag = "(Content Relevant)"

            tweet_lines.append(f"- [{tid}] {loc_tag} {text}")
        tweet_section = "\n".join(tweet_lines)
    else:
        tweet_section = "(No tweets found for this location/time)"

    # Format 311 calls with IDs
    calls = context.get("calls", [])
    if calls:
        call_lines = []
        for c in calls:
            cid = c.get("doc_id") or c.get("record_id") or "unknown"
            desc = (c.get("description") or "").replace("\n", " ")
            # Check if call starts with correct ZIP (311 often has precise loc)
            c_zip = c.get("zip")
            if c_zip and str(c_zip) == str(zip_code):
                loc_tag = f"(Verified ZIP {zip_code})"
            else:
                loc_tag = "(Content Relevant)"

            call_lines.append(f"- [{cid}] {loc_tag} {desc}")
        call_section = "\n".join(call_lines)
    else:
        call_section = "(No 311 calls found for this location/time)"

    sensor_table = context.get("sensor_table", "")
    if not sensor_table or "loss_mean" in sensor_table:
        # The 'loss_mean' table is historical data, not current sensor readings.
        # To prevent hallucination, we label it clearly or omit it if it's just 0.0
        sensor_section = f"### Historical Loss Data (Previous Years):\n{sensor_table}"
    else:
        # Add temporal warning for sensor data
        start_date = time_window.get("start", "")
        sensor_section = f"""### Sensor Data (Rainfall/Water Levels):
⚠️ TEMPORAL CONTEXT: This sensor data is from {start_date} (query time window).
If this date is AFTER Aug 28 (peak flooding), sensor showing "0.0" means water RECEDED,
NOT that flooding didn't happen. Prioritize tweets/311 calls from Aug 27-28 for flood extent.

{sensor_table}"""

    # Format captions with temporal warning
    captions = context.get("captions", [])
    if captions:
        caption_lines = []
        for cap in captions:
            cid = cap.get("doc_id") or cap.get("tile_id") or "unknown"
            text = (cap.get("text") or cap.get("caption") or "").replace("\n", " ")
            caption_lines.append(f"- [{cid}] {text[:200]}")
        caption_section = "\n".join(caption_lines[:10])  # Limit to 10 captions
    else:
        caption_section = ""

    # Build caption block with temporal warning if captions exist
    if caption_section:
        caption_block = f"""
        ### Image Captions (⚠️ TEMPORAL WARNING):
        **These captions describe imagery from Aug 31, 2017 - AFTER peak flooding (Aug 27-28).**
        **By Aug 31, floodwaters had RECEDED. "No flooding visible" does NOT mean flooding didn't occur!**
        
        - For FLOOD EXTENT: IGNORE these captions. Trust tweets/311 calls instead.
        - For STRUCTURAL DAMAGE: These captions ARE useful (damage is persistent).
        
        {caption_section}
        """
    else:
        caption_block = ""

    # Spatial priors (rainfall + NFIP + Harvey claims)
    spatial_priors = context.get("spatial_priors", [])
    spatial_section = "\n".join(f"- {p}" for p in spatial_priors) if spatial_priors else "(No spatial priors available)"

    # Hard evidence floors — computed from Harvey NFIP claims + rainfall
    floors = context.get("evidence_floors", {})
    flood_floor = floors.get("flood_floor", 0.0)
    damage_floor = floors.get("damage_floor", 0.0)
    floor_reasons = floors.get("reasons", [])
    if flood_floor > 0 or damage_floor > 0:
        floor_lines = "\n".join(f"  - {r}" for r in floor_reasons)
        floor_block = f"""
⚠️ HARD MINIMUM CONSTRAINTS (you MUST NOT go below these — they are computed from data):
  flood_extent_pct  ≥ {flood_floor}%
  damage_severity_pct ≥ {damage_floor}%
Reasons:
{floor_lines}
"""
    else:
        floor_block = ""

    return dedent(
        f"""
        ZIP: {zip_code}
        Time window: {time_window['start']} to {time_window['end']}
        Imagery IDs: {[tile['tile_id'] for tile in context.get('imagery_tiles', [])]}

        {sensor_section}

        ### Harvey 2017 Evidence & Spatial Priors for ZIP {zip_code} (USE AS PRIMARY EVIDENCE):
        {spatial_section}

        ### FEMA Prior Knowledge (Historical Context):
        {context.get('kb_summary', '')}

        ### Tweets (Relevant to ZIP {zip_code}):
        NOTE: Tweets tagged "(Content Relevant)" are regional Harvey reports — they confirm flooding
        occurred across Harris County including this ZIP. Do NOT dismiss them as too general.
        {tweet_section}

        ### 311 Calls (ZIP {zip_code}, Exact Match):
        {call_section}
        {caption_block}
        {floor_block}
        Respond with JSON matching schema:
        {{
          "reasoning": str,  // Your analysis of the evidence
          "zip": str,
          "time_window": {{"start": str, "end": str}},
          "estimates": {{
            "flood_extent_pct": float,  // % of ZIP area flooded (0-100). Use Harvey claims + rainfall + tweets as floor.
                                          // NEVER output 0 if Harvey claims > 0 or regional Harvey tweets present.
            "damage_severity_pct": float, // Average structural damage per building (0-100).
                                          // Use Harvey avg_water_depth_ft as minimum floor when tweet evidence is absent.
                                          // 0-1ft→≥10%, 1-3ft→≥20%, 3-6ft→≥40%, >6ft→≥65%.
            "roads_impacted": list[str],
            "confidence": float  // 0-1
          }},
          "evidence_refs": {{
            "imagery_tile_ids": list[str],
            "tweet_ids": list[str],
            "call_311_ids": list[str],
            "sensor_ids": list[str],
            "kb_refs": list[str]
          }},
          "natural_language_summary": str
        }}
        """
    )
