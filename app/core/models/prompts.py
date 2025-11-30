"""Prompt templates for multimodal models."""

from __future__ import annotations

from textwrap import dedent


SYSTEM_PROMPT = dedent(
    """
    You are an assistant estimating post-disaster impact for Harris County, TX during Hurricane Harvey.
    You will be provided with satellite imagery, text snippets, sensor data, and FEMA priors.
    
    IMPORTANT: All provided text snippets (Tweets, 311 Calls) are pre-filtered and RELEVANT to the queried location/time. Do not discard them just because they lack an explicit ZIP code in the text.

    CRITICAL INSTRUCTION:
    - You MUST analyze the provided images to visually confirm flooding (e.g., water on roads, submerged structures).
    - You MUST ALSO analyze the provided Tweets and 311 Calls for on-the-ground reports.
    - If images show no damage but tweets report flooding, acknowledge the discrepancy and report the text evidence.
    - Cite specific Image IDs, Tweet IDs, and Call IDs in your summary.
    
    CHAIN OF THOUGHT REASONING:
    - Before generating the final estimates, you MUST populate the "reasoning" field.
    - In "reasoning", list every Tweet ID and 311 Call ID you see.
    - Explicitly state what each text snippet reports.
    - Then, combine this with visual evidence to form your conclusion.
    
    EXAMPLE:
    Input Context:
    - Images: [IMG_1] (shows clear roads)
    - Tweets: [- [T123] (ZIP 77002) "Water entering my living room!"]
    
    Correct Output Reasoning:
    "I see Tweet T123 reporting water in living room. Image IMG_1 shows clear roads. There is a discrepancy. I will report the flooding based on the tweet."
    
    Respond with valid JSON matching the schema provided by the user.
    """
)


TEXT_ONLY_SYSTEM_PROMPT = dedent(
    """
    You are an assistant estimating post-disaster impact for Harris County, TX during Hurricane Harvey.
    You will be provided with text snippets (Tweets, 311 Calls), sensor data, and FEMA priors.
    
    CRITICAL INSTRUCTION:
    - You MUST analyze the provided Tweets and 311 Calls for on-the-ground reports.
    - Cite specific Tweet IDs and Call IDs in your summary.
    - Do NOT mention imagery or visual evidence, as none is provided.
    
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
            # Force model to see ZIP relevance
            tweet_lines.append(f"- [{tid}] (ZIP {zip_code}) {text}")
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
            # Force model to see ZIP relevance
            call_lines.append(f"- [{cid}] (ZIP {zip_code}) {desc}")
        call_section = "\n".join(call_lines)
    else:
        call_section = "(No 311 calls found for this location/time)"

    sensor_table = context.get("sensor_table", "")
    if not sensor_table or "loss_mean" in sensor_table:
        # The 'loss_mean' table is historical data, not current sensor readings.
        # To prevent hallucination, we label it clearly or omit it if it's just 0.0
        sensor_section = f"### Historical Loss Data (Previous Years):\n{sensor_table}"
    else:
        sensor_section = f"### Sensor Data (Rainfall/Water Levels):\n{sensor_table}"

    return dedent(
        f"""
        ZIP: {zip_code}
        Time window: {time_window['start']} to {time_window['end']}
        Imagery IDs: {[tile['tile_id'] for tile in context.get('imagery_tiles', [])]}
        
        {sensor_section}
        
        ### FEMA Prior Knowledge (Historical Context):
        {context.get('kb_summary', '')}

        ### Tweets (Confirmed in ZIP {zip_code}):
        {tweet_section}

        ### 311 Calls (Confirmed in ZIP {zip_code}):
        {call_section}

        Respond with JSON matching schema:
        {{
          "reasoning": str,
          "zip": str,
          "time_window": {{"start": str, "end": str}},
          "estimates": {{"structural_damage_pct": float, "roads_impacted": list[str], "confidence": float}},
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
