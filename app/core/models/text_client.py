"""Client for text-only analysis using Gemini or OpenAI."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from loguru import logger

from .prompts import TEXT_ONLY_SYSTEM_PROMPT, build_user_prompt


class TextAnalysisClient:
    """Client for performing text-only analysis."""

    def __init__(
        self,
        model_name: str = None,
        api_key: Optional[str] = None,
        provider: str = None,
    ):
        import os

        # Determine provider
        self.provider = provider or os.getenv("MODEL_PROVIDER", "gemini")

        if self.provider == "gemini":
            self._init_gemini(model_name, api_key or os.getenv("GEMINI_API_KEY"))
        elif self.provider == "openai":
            self._init_openai(model_name, api_key or os.getenv("OPENAI_API_KEY"))
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _init_gemini(self, model_name: str, api_key: str):
        """Initialize Gemini client."""
        import os
        import google.generativeai as genai

        if not api_key:
            logger.warning("GEMINI_API_KEY not set. TextAnalysisClient will fail.")

        if not model_name:
            model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name, system_instruction=TEXT_ONLY_SYSTEM_PROMPT.strip()
        )
        self.gemini_model = self.model

    def _init_openai(self, model_name: str, api_key: str):
        """Initialize OpenAI client."""
        import os

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

        if not api_key:
            logger.warning("OPENAI_API_KEY not set. TextAnalysisClient will fail.")

        if not model_name:
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        self.openai_client = OpenAI(api_key=api_key)
        self.openai_model = model_name

    def analyze(
        self,
        zip_code: str,
        time_window: Dict[str, str],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Analyze text evidence for a given ZIP and time window.

        Args:
            zip_code: The ZIP code to analyze.
            time_window: Dict with 'start' and 'end' dates.
            context: Dictionary containing retrieved text snippets (tweets, calls, etc.).
                     NOTE: 'imagery_tiles' should be ignored or empty.

        Returns:
            JSON response with reasoning, estimates, and evidence_refs.
        """
        # Ensure we are in text-only mode by stripping imagery from context view passed to prompt
        text_context = context.copy()
        text_context["imagery_tiles"] = []

        prompt = build_user_prompt(zip_code, time_window, text_context)

        try:
            if self.provider == "gemini":
                return self._analyze_gemini(prompt)
            elif self.provider == "openai":
                return self._analyze_openai(prompt)
        except Exception as e:
            logger.error(f"Text analysis failed: {e}")
            return {
                "reasoning": f"Analysis failed: {e}",
                "estimates": {"structural_damage_pct": 0.0, "confidence": 0.0},
                "evidence_refs": {"tweet_ids": [], "call_311_ids": [], "kb_refs": []},
            }

    def _analyze_gemini(self, prompt: str) -> Dict[str, Any]:
        """Analyze using Gemini."""
        response = self.gemini_model.generate_content(
            prompt, generation_config={"response_mime_type": "application/json"}
        )
        return json.loads(response.text)

    def _analyze_openai(self, prompt: str) -> Dict[str, Any]:
        """Analyze using OpenAI."""
        response = self.openai_client.chat.completions.create(
            model=self.openai_model,
            messages=[
                {"role": "system", "content": TEXT_ONLY_SYSTEM_PROMPT.strip()},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)

    def generate_text(self, prompt: str) -> str:
        """Generate text for evaluation purposes (LLM-as-a-judge)."""
        try:
            if self.provider == "gemini":
                response = self.gemini_model.generate_content(prompt)
                return response.text
            elif self.provider == "openai":
                response = self.openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            return ""
