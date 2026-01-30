"""Client for text-only analysis using Gemini, OpenAI, or Hugging Face models."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from loguru import logger

from .prompts import TEXT_ONLY_SYSTEM_PROMPT, build_user_prompt


# Hugging Face model configurations (via OpenAI-compatible API)
HF_MODEL_CONFIGS = {
    # Meta Llama models
    "llama-3.3-70b": {
        "model": "meta-llama/Llama-3.3-70B-Instruct",
        "provider_suffix": ":together",  # Together AI backend
    },
    "llama-3.1-70b": {
        "model": "meta-llama/Llama-3.1-70B-Instruct",
        "provider_suffix": ":together",
    },
    # Qwen models
    "qwen2.5-72b": {
        "model": "Qwen/Qwen2.5-72B-Instruct",
        "provider_suffix": ":novita",  # Novita AI backend
    },
    "qwen2.5-32b": {
        "model": "Qwen/Qwen2.5-32B-Instruct",
        "provider_suffix": ":novita",
    },
    # Mistral models
    "mistral-large": {
        "model": "mistralai/Mistral-Large-Instruct-2411",
        "provider_suffix": ":together",
    },
    # Google Gemma (open-source)
    "gemma-2-27b": {
        "model": "google/gemma-2-27b-it",
        "provider_suffix": ":nebius",
    },
}


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
        self.model_name = model_name  # Store for metadata

        if self.provider == "gemini":
            self._init_gemini(model_name, api_key or os.getenv("GEMINI_API_KEY"))
        elif self.provider == "openai":
            self._init_openai(model_name, api_key or os.getenv("OPENAI_API_KEY"))
        elif self.provider == "huggingface":
            self._init_huggingface(model_name, api_key or os.getenv("HF_TOKEN"))
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

    def _init_huggingface(self, model_name: str, api_key: str):
        """Initialize Hugging Face client via OpenAI-compatible API."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

        if not api_key:
            logger.warning("HF_TOKEN not set. TextAnalysisClient will fail.")

        # Get model config
        if model_name and model_name in HF_MODEL_CONFIGS:
            config = HF_MODEL_CONFIGS[model_name]
            full_model_name = config["model"] + config["provider_suffix"]
        elif model_name:
            # Allow custom model names (assume :together suffix if not specified)
            full_model_name = model_name if ":" in model_name else f"{model_name}:together"
        else:
            # Default to Llama 3.3 70B
            config = HF_MODEL_CONFIGS["llama-3.3-70b"]
            full_model_name = config["model"] + config["provider_suffix"]
            model_name = "llama-3.3-70b"

        self.hf_client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=api_key,
        )
        self.hf_model = full_model_name
        self.model_name = model_name
        logger.info(f"Initialized HuggingFace client with model: {full_model_name}")

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
            elif self.provider == "huggingface":
                return self._analyze_huggingface(prompt)
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

    def _analyze_huggingface(self, prompt: str) -> Dict[str, Any]:
        """Analyze using Hugging Face models via OpenAI-compatible API."""
        import re
        
        # HuggingFace models may not support response_format, so we ask for JSON in the prompt
        json_instruction = (
            "\n\nIMPORTANT: Respond with valid JSON only. "
            "Do not include any text before or after the JSON object. "
            "Use double quotes for all strings and property names. "
            "Do not use trailing commas."
        )
        
        response = self.hf_client.chat.completions.create(
            model=self.hf_model,
            messages=[
                {"role": "system", "content": TEXT_ONLY_SYSTEM_PROMPT.strip()},
                {"role": "user", "content": prompt + json_instruction},
            ],
            max_tokens=4096,
        )
        
        raw_text = response.choices[0].message.content
        
        # Try to extract JSON from the response (handle markdown code blocks)
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', raw_text)
        if json_match:
            raw_text = json_match.group(1)
        
        # Try to find JSON object in the response
        json_start = raw_text.find('{')
        json_end = raw_text.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            raw_text = raw_text[json_start:json_end]
        
        # Fix common JSON issues from open-source models
        # 1. Replace single quotes with double quotes (but not inside strings)
        # 2. Remove trailing commas before } or ]
        raw_text = re.sub(r",\s*([}\]])", r"\1", raw_text)
        
        # Try to parse, if it fails try more aggressive fixes
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            # More aggressive: replace single quotes with double quotes
            # This is risky but sometimes necessary for models that use single quotes
            fixed_text = raw_text.replace("'", '"')
            try:
                return json.loads(fixed_text)
            except json.JSONDecodeError:
                # Last resort: try to extract key fields manually
                logger.warning(f"JSON parsing failed, attempting manual extraction from: {raw_text[:500]}...")
                return self._extract_fields_manually(raw_text)
    
    def _extract_fields_manually(self, raw_text: str) -> Dict[str, Any]:
        """Extract key fields from malformed JSON response."""
        import re
        
        result = {
            "reasoning": "",
            "estimates": {
                "flood_extent_pct": 0.0,
                "damage_severity_pct": 0.0,
                "confidence": 0.0,
            },
            "evidence_refs": {
                "tweet_ids": [],
                "call_311_ids": [],
            },
        }
        
        # Try to extract flood_extent_pct
        extent_match = re.search(r'"?flood_extent_pct"?\s*:\s*(\d+(?:\.\d+)?)', raw_text)
        if extent_match:
            result["estimates"]["flood_extent_pct"] = float(extent_match.group(1))
        
        # Try to extract damage_severity_pct
        damage_match = re.search(r'"?damage_severity_pct"?\s*:\s*(\d+(?:\.\d+)?)', raw_text)
        if damage_match:
            result["estimates"]["damage_severity_pct"] = float(damage_match.group(1))
        
        # Try to extract confidence
        conf_match = re.search(r'"?confidence"?\s*:\s*(\d+(?:\.\d+)?)', raw_text)
        if conf_match:
            result["estimates"]["confidence"] = float(conf_match.group(1))
        
        # Try to extract reasoning
        reasoning_match = re.search(r'"?reasoning"?\s*:\s*"([^"]*)"', raw_text)
        if reasoning_match:
            result["reasoning"] = reasoning_match.group(1)
        
        logger.info(f"Manually extracted: extent={result['estimates']['flood_extent_pct']}, damage={result['estimates']['damage_severity_pct']}")
        return result

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
            elif self.provider == "huggingface":
                response = self.hf_client.chat.completions.create(
                    model=self.hf_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=4096,
                )
                return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            return ""
