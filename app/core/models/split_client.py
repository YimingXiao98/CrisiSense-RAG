"""Client orchestrating text analysis and verification."""

from __future__ import annotations

import os
from typing import Dict, List, Any

from .text_client import TextAnalysisClient
from .verifier import Verifier


class SplitPipelineClient:
    """
    Orchestrates the split pipeline.
    Currently implements Text-Only + Verification.
    Future: Will include VisualAnalysisClient and FusionEngine.
    """

    def __init__(self, provider: str = None):
        if not provider:
            provider = os.getenv("MODEL_PROVIDER", "gemini")
        self.text_client = TextAnalysisClient(provider=provider)
        self.verifier = Verifier()
        self.provider = provider

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
        """
        Infer method compatible with VLMClient.
        """
        # Construct context dictionary expected by TextAnalysisClient
        context = {
            "imagery_tiles": imagery_tiles,  # Will be ignored/stripped by TextAnalysisClient
            "text_snippets": text_snippets,
            "sensor_table": sensor_table,
            "kb_summary": kb_summary,
            "tweets": tweets or [],
            "calls": calls or [],
            "sensors": sensors or [],
            "fema": fema or [],
        }

        # 1. Text Analysis
        raw_response = self.text_client.analyze(zip_code, time_window, context)

        # 2. Verification
        verified_response = self.verifier.verify(raw_response, context)

        # 3. Add missing fields expected by EvalRunner/GenerationEvaluator
        # TextAnalysisClient returns {reasoning, estimates, evidence_refs}
        # We might need to add 'natural_language_summary' if it's missing or named differently.
        # TextAnalysisClient uses 'reasoning' as the main text.
        # VLMClient usually returns 'natural_language_summary'.
        # Let's check TextAnalysisClient output again.

        # TextAnalysisClient returns whatever the LLM returns.
        # The prompt asks for JSON with "reasoning".
        # VLMClient output has "natural_language_summary".
        # We should map "reasoning" to "natural_language_summary" if needed.

        if "natural_language_summary" not in verified_response:
            verified_response["natural_language_summary"] = verified_response.get(
                "reasoning", ""
            )

        # Ensure other fields exist
        if "estimates" not in verified_response:
            verified_response["estimates"] = {
                "structural_damage_pct": 0.0,
                "confidence": 0.0,
            }

        return verified_response

    def _generate_text(self, prompt: str) -> str:
        """
        Generate text for evaluation purposes (LLM-as-a-judge).
        Delegates to the text client's generate_text method.
        """
        return self.text_client.generate_text(prompt)
