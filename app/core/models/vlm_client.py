"""Model client adapters for multimodal inference."""
from __future__ import annotations

import json
import os
import time
import urllib.request
from typing import Dict, List

from loguru import logger

from .prompts import (
    QUERY_PARSING_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_query_parsing_prompt,
    build_user_prompt,
)


class VLMClient:
    """Interface wrapping different multimodal providers."""

    def __init__(self, provider: str | None = None) -> None:
        self.provider = provider or os.getenv("MODEL_PROVIDER", "mock")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.gemini_model = os.getenv(
            "GEMINI_MODEL", "models/gemini-1.5-flash")
        self.gemini_max_attempts = int(os.getenv("GEMINI_MAX_ATTEMPTS", "3"))
        self._gemini_client = None

    def infer(
        self,
        zip_code: str,
        time_window: Dict[str, str],
        imagery_tiles: List[dict],
        text_snippets: List[str],
        sensor_table: str,
        kb_summary: str,
        tweets: List[dict] = None,
        calls: List[dict] = None,
        sensors: List[dict] = None,
        fema: List[dict] = None,
    ) -> Dict[str, object]:
        """Generate RAG answer using configured provider."""

        context = {
            "imagery_tiles": imagery_tiles,
            "text_snippets": text_snippets,
            "sensor_table": sensor_table,
            "kb_summary": kb_summary,
            "tweets": tweets or [],
            "calls": calls or [],
            "sensors": sensors or [],
            "fema": fema or [],
        }
        if self.provider == "mock":
            return self._mock_response(zip_code, time_window, context)
        if self.provider == "openai":
            return self._openai_stub(zip_code, time_window, context)
        if self.provider == "gemini":
            return self._gemini_response(zip_code, time_window, context)
        raise ValueError(f"Unsupported provider: {self.provider}")

    def parse_query(self, message: str) -> Dict[str, str | None]:
        """Parse natural language query into structured parameters using LLM."""
        if self.provider == "mock":
            # Mock parser just returns None to let regex fallback handle it,
            # or we could implement a simple mock parser here.
            return {"zip": None, "start": None, "end": None}

        if self.provider == "gemini":
            return self._gemini_parse_query(message)

        logger.warning(
            f"Query parsing not implemented for provider {self.provider}")
        return {"zip": None, "start": None, "end": None}

    def _gemini_parse_query(self, message: str) -> Dict[str, str | None]:
        """Use Gemini to parse the query."""
        if not self.gemini_api_key:
            logger.warning(
                "GEMINI_API_KEY not set, skipping LLM query parsing")
            return {"zip": None, "start": None, "end": None}

        try:
            import google.generativeai as genai
        except ImportError:
            return {"zip": None, "start": None, "end": None}

        parser_model = genai.GenerativeModel(
            self.gemini_model,
            system_instruction=QUERY_PARSING_SYSTEM_PROMPT.strip()
        )

        user_prompt = build_query_parsing_prompt(message)
        messages = [{"role": "user", "parts": [{"text": user_prompt}]}]

        try:
            response = parser_model.generate_content(
                messages,
                generation_config={"response_mime_type": "application/json"},
            )
            payload = self._extract_response_text(response)
            parsed = json.loads(payload)
            return parsed
        except Exception as exc:
            logger.error(f"Gemini query parsing failed: {exc}")
            return {"zip": None, "start": None, "end": None}

    def _mock_response(self, zip_code: str, time_window: Dict[str, str], context: Dict[str, object]) -> Dict[str, object]:
        """Return deterministic pseudo-output for tests."""

        tile_ids = [tile["tile_id"]
                    for tile in context.get("imagery_tiles", [])]
        tweet_ids = []
        call_ids = []
        if context.get("text_snippets"):
            tweet_ids = [f"tw_{i}" for i in range(
                len(context["text_snippets"]))]
            call_ids = [f"311_{i}" for i in range(
                len(context["text_snippets"]))]
        sensor_ids = [tile.get("sensor_id", f"sensor_{i}") for i, tile in enumerate(
            context.get("imagery_tiles", []))]
        kb_refs = [f"fema_zip_{zip_code}_2010_2016"] if context.get(
            "kb_summary") else []
        return {
            "zip": zip_code,
            "time_window": time_window,
            "estimates": {
                "structural_damage_pct": round(min(len(tile_ids) * 0.05, 0.9), 2),
                "roads_impacted": [f"Road_{i}" for i in range(min(2, len(tile_ids)))],
                "confidence": 0.75,
            },
            "evidence_refs": {
                "imagery_tile_ids": tile_ids,
                "tweet_ids": tweet_ids[:2],
                "call_311_ids": call_ids[:2],
                "sensor_ids": sensor_ids[:2],
                "kb_refs": kb_refs,
            },
            "natural_language_summary": f"Based on {len(tile_ids)} imagery tiles and {len(tweet_ids)} tweets, we estimate moderate structural damage in zip {zip_code}.",
        }

    def _openai_stub(self, zip_code: str, time_window: Dict[str, str], context: Dict[str, object]) -> Dict[str, object]:
        """Placeholder for OpenAI GPT-4V integration."""

        logger.info(
            "OpenAI provider selected but not implemented; falling back to mock")
        return self._mock_response(zip_code, time_window, context)

    def _fetch_image(self, uri: str) -> bytes | None:
        """Fetch image data from URI (handles example.com mocks)."""
        if not uri:
            return None
        
        if "example.com" in uri:
            # Return a 1x1 black pixel JPEG for testing/mocking
            return b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xdb\x00C\x01\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x03\x01\x22\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04\x04\x00\x00\x01\x7d\x01\x02\x03\x00\x04\x11\x05\x12\x21\x31\x41\x06\x13\x51\x61\x07\x22\x71\x14\x32\x81\x91\xa1\x08\x23\x42\xb1\xc1\x15\x52\xd1\xf0\x24\x33\x62\x72\x82\x09\x0a\x16\x17\x18\x19\x1a\x25\x26\x27\x28\x29\x2a\x34\x35\x36\x37\x38\x39\x3a\x43\x44\x45\x46\x47\x48\x49\x4a\x53\x54\x55\x56\x57\x58\x59\x5a\x63\x64\x65\x66\x67\x68\x69\x6a\x73\x74\x75\x76\x77\x78\x79\x7a\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xff\xc4\x00\x1f\x01\x00\x03\x01\x01\x01\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\xff\xc4\x00\xb5\x11\x00\x02\x01\x02\x04\x04\x03\x04\x07\x05\x04\x04\x00\x01\x02\x77\x00\x01\x02\x03\x11\x04\x05\x21\x31\x06\x12\x41\x51\x07\x61\x71\x13\x22\x32\x81\x08\x14\x42\x91\xa1\xb1\xc1\x09\x23\x33\x52\xf0\x15\x62\x72\xd1\x0a\x16\x24\x34\xe1\x25\xf1\x17\x18\x19\x1a\x26\x27\x28\x29\x2a\x35\x36\x37\x38\x39\x3a\x43\x44\x45\x46\x47\x48\x49\x4a\x53\x54\x55\x56\x57\x58\x59\x5a\x63\x64\x65\x66\x67\x68\x69\x6a\x73\x74\x75\x76\x77\x78\x79\x7a\x82\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00\x3f\x00\xf7\xfa\x28\xa2\x80\x3f\xff\xd9'
            
        try:
            with urllib.request.urlopen(uri, timeout=5) as resp:
                return resp.read()
        except Exception as exc:
            logger.warning(f"Failed to fetch image {uri}: {exc}")
            return None

    def _gemini_response(self, zip_code: str, time_window: Dict[str, str], context: Dict[str, object]) -> Dict[str, object]:
        """Call the Gemini API and parse the structured JSON response."""

        if not self.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set; required for Gemini provider.")
        try:
            import google.generativeai as genai  # type: ignore import-not-found
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "google-generativeai package is required for Gemini provider") from exc

        if self._gemini_client is None:
            genai.configure(api_key=self.gemini_api_key)
            self._gemini_client = genai.GenerativeModel(
                self.gemini_model, system_instruction=SYSTEM_PROMPT.strip())

        user_prompt = build_user_prompt(zip_code, time_window, context)
        
        # Collect image parts
        parts = [{"text": user_prompt}]
        
        # Attach images
        imagery_tiles = context.get("imagery_tiles", [])
        for tile in imagery_tiles:
            uri = tile.get("uri")
            image_data = self._fetch_image(uri)
            if image_data:
                parts.append({
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": image_data
                    }
                })
            if len(parts) >= 16: # Limit max images to avoid payload size errors or limits
                logger.warning("Max image limit reached for Gemini call")
                break

        messages = [
            {"role": "user", "parts": parts},
        ]

        for attempt in range(1, self.gemini_max_attempts + 1):
            start_ts = time.perf_counter()
            try:
                response = self._gemini_client.generate_content(
                    messages,
                    generation_config={
                        "response_mime_type": "application/json"},
                )
                latency_ms = round((time.perf_counter() - start_ts) * 1000, 2)
                payload = self._extract_response_text(response)
                parsed = json.loads(payload)
                usage = getattr(response, "usage_metadata", None)
                usage_info = getattr(usage, "__dict__", usage)
                logger.info("Gemini call succeeded", attempt=attempt,
                            latency_ms=latency_ms, usage=usage_info)
                return parsed
            except json.JSONDecodeError:
                logger.warning(
                    "Gemini returned non-JSON response; falling back to mock output",
                    snippet=payload[:200] if 'payload' in locals() else '',
                )
                return self._mock_response(zip_code, time_window, context)
            except Exception as exc:  # pragma: no cover - network errors
                latency_ms = round((time.perf_counter() - start_ts) * 1000, 2)
                logger.error("Gemini API call failed", attempt=attempt,
                             latency_ms=latency_ms, error=str(exc))
                if attempt == self.gemini_max_attempts:
                    raise
                time.sleep(min(2 ** (attempt - 1), 4))

        # Should not reach here because loop either returns or raises.
        return self._mock_response(zip_code, time_window, context)

    def _extract_response_text(self, response) -> str:
        """Best-effort extraction of text from Gemini responses."""

        text = getattr(response, "text", None)
        if text:
            return text.strip()
        for candidate in getattr(response, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            if not content:
                continue
            for part in getattr(content, "parts", []) or []:
                value = getattr(part, "text", None)
                if value:
                    return value.strip()
        logger.warning(
            "Gemini response had no text content; returning empty payload")
        return ""
