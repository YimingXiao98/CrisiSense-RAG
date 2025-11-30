"""Verifier component for validating RAG responses against context."""

from __future__ import annotations

from typing import Any, Dict, List, Set

from loguru import logger


class Verifier:
    """Verifies RAG responses to ensure faithfulness to the provided context."""

    def _extract_ids(self, records: List[Dict[str, Any]]) -> Set[str]:
        """Extract tweet IDs from structured records."""
        return {str(r.get("tweet_id")) for r in records if r.get("tweet_id")}

    def _extract_call_ids(self, records: List[Dict[str, Any]]) -> Set[str]:
        """Extract 311 call IDs from structured records."""
        return {str(r.get("record_id")) for r in records if r.get("record_id")}

    def _extract_sensor_ids(self, records: List[Dict[str, Any]]) -> Set[str]:
        """Extract sensor IDs from structured records."""
        # Note: context['sensors'] is a list of dicts, but context['sensor_table'] is markdown.
        # We should use the list of dicts if available.
        return {str(r.get("sensor_id")) for r in records if r.get("sensor_id")}

    def verify(
        self, response: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Verify the response against the context.

        Args:
            response: The JSON response from the model.
            context: The context dictionary used to generate the response.

        Returns:
            The verified (and potentially filtered) response.
        """
        verified_response = response.copy()
        evidence_refs = verified_response.get("evidence_refs", {})

        # 1. Validate Tweet IDs
        valid_tweet_ids = self._extract_ids(context.get("tweets", []))
        cited_tweet_ids = evidence_refs.get("tweet_ids", [])
        verified_tweet_ids = [
            tid for tid in cited_tweet_ids if str(tid) in valid_tweet_ids
        ]

        if len(verified_tweet_ids) != len(cited_tweet_ids):
            logger.warning(
                f"Filtered {len(cited_tweet_ids) - len(verified_tweet_ids)} invalid tweet IDs."
            )
        evidence_refs["tweet_ids"] = verified_tweet_ids

        # 2. Validate 311 Call IDs
        valid_call_ids = self._extract_call_ids(context.get("calls", []))
        cited_call_ids = evidence_refs.get("call_311_ids", [])

        verified_call_ids = []
        for cid in cited_call_ids:
            cid_str = str(cid)
            if cid_str in valid_call_ids:
                verified_call_ids.append(cid_str)
            else:
                # Fuzzy match: check if any valid ID is a suffix of the cited ID
                match = next(
                    (vid for vid in valid_call_ids if cid_str.endswith(vid)), None
                )
                if match:
                    logger.info(f"Fuzzy matched call ID {cid} to {match}")
                    verified_call_ids.append(match)
                else:
                    logger.warning(f"Invalid 311 Call ID: {cid}")

        evidence_refs["call_311_ids"] = verified_call_ids

        # 3. Validate Sensor IDs
        # Use structured sensors list if available, else try table parsing (fallback not implemented here for simplicity)
        valid_sensor_ids = self._extract_sensor_ids(context.get("sensors", []))
        cited_sensor_ids = evidence_refs.get("sensor_ids", [])

        verified_sensor_ids = [
            str(sid) for sid in cited_sensor_ids if str(sid) in valid_sensor_ids
        ]

        if len(verified_sensor_ids) != len(cited_sensor_ids):
            logger.warning(
                f"Filtered {len(cited_sensor_ids) - len(verified_sensor_ids)} invalid sensor IDs."
            )

        evidence_refs["sensor_ids"] = verified_sensor_ids

        verified_response["evidence_refs"] = evidence_refs
        return verified_response
