"""Multi-judge evaluation client supporting various LLM providers."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from loguru import logger
from openai import OpenAI


# Judge configurations
JUDGE_CONFIGS = {
    # OpenAI judges
    "gpt-4o": {
        "provider": "openai",
        "model": "gpt-4o",
        "base_url": None,  # Uses default
    },
    "gpt-4o-mini": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "base_url": None,
    },
    # Google judges (via OpenAI-compatible API)
    "gemini-2.0-flash": {
        "provider": "gemini",
        "model": "gemini-2.0-flash-exp",
        "base_url": None,  # Uses Gemini SDK
    },
    "gemini-2.5-flash": {
        "provider": "gemini",
        "model": "gemini-2.5-flash-preview-05-20",
        "base_url": None,
    },
    # HuggingFace models via router
    "llama-3.2-90b": {
        "provider": "huggingface",
        "model": "meta-llama/Llama-3.2-90B-Vision-Instruct:together",
        "base_url": "https://router.huggingface.co/v1",
    },
    "gemma-3-27b": {
        "provider": "huggingface",
        "model": "google/gemma-3-27b-it:nebius",
        "base_url": "https://router.huggingface.co/v1",
    },
    "qwen3-vl-235b": {
        "provider": "huggingface",
        "model": "Qwen/Qwen3-VL-235B-A22B-Instruct:novita",
        "base_url": "https://router.huggingface.co/v1",
    },
}


class JudgeClient:
    """Client for LLM-as-a-Judge with support for multiple providers."""

    def __init__(self, judge_name: str) -> None:
        """Initialize judge client with specific model.
        
        Args:
            judge_name: Key from JUDGE_CONFIGS (e.g., 'gpt-4o', 'gemini-2.5-flash', 'llama-3.2-90b')
        """
        if judge_name not in JUDGE_CONFIGS:
            raise ValueError(f"Unknown judge: {judge_name}. Available: {list(JUDGE_CONFIGS.keys())}")
        
        self.judge_name = judge_name
        self.config = JUDGE_CONFIGS[judge_name]
        self.provider = self.config["provider"]
        self.model = self.config["model"]
        
        # Initialize appropriate client
        if self.provider == "openai":
            self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        elif self.provider == "huggingface":
            self.client = OpenAI(
                base_url=self.config["base_url"],
                api_key=os.environ.get("HF_TOKEN"),
            )
        elif self.provider == "gemini":
            # Use Google's SDK for Gemini (uses GEMINI_API_KEY to match project convention)
            try:
                import google.generativeai as genai
                genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
                self.gemini_model = genai.GenerativeModel(self.model)
                self.client = None  # Will use Gemini SDK directly
            except ImportError:
                raise ImportError("google-generativeai package required for Gemini models")
        
        logger.info(f"Initialized judge: {judge_name} ({self.provider}/{self.model})")

    def generate_text(self, prompt: str, max_retries: int = 3) -> str:
        """Generate text completion from the judge model with retry logic."""
        import time
        
        for attempt in range(max_retries):
            try:
                if self.provider == "gemini":
                    # Use Gemini SDK
                    response = self.gemini_model.generate_content(prompt)
                    return response.text
                else:
                    # Use OpenAI-compatible API
                    completion = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=1024,
                        temperature=0.1,  # Low temperature for consistent judging
                    )
                    return completion.choices[0].message.content
            except Exception as e:
                error_str = str(e)
                # Check for rate limit error
                if "429" in error_str or "rate" in error_str.lower():
                    wait_time = (2 ** attempt) * 2  # 2, 4, 8 seconds
                    logger.warning(f"Rate limit hit for {self.judge_name}, waiting {wait_time}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Judge {self.judge_name} failed: {e}")
                    raise
        
        raise RuntimeError(f"Judge {self.judge_name} failed after {max_retries} retries")

    def evaluate_faithfulness(
        self,
        context: Dict[str, Any],
        answer: str,
        query_params: Dict[str, str] = None,
    ) -> Dict[str, Any]:
        """Evaluate faithfulness of answer to context.
        
        Returns:
            Dict with 'score', 'reasoning', 'supported_claims', 'unsupported_claims'
        """
        # Prepare context summary
        snippets = "\n".join(context.get("text_snippets", []))
        kb_summary = context.get("kb_summary", "")
        
        # Add imagery metadata
        image_ids = [t.get("tile_id", "unknown") for t in context.get("imagery_tiles", [])]
        imagery_context = f"Available Imagery Tiles: {', '.join(image_ids)}" if image_ids else ""
        
        # Add query context
        query_context = ""
        if query_params:
            query_context = f"Query Parameters: ZIP={query_params.get('zip')}, Start={query_params.get('start')}, End={query_params.get('end')}"

        prompt = f"""You are a strict fact-checking judge. 
Your task is to verify if the following ANSWER is supported by the provided CONTEXT.

CONTEXT:
{query_context}
{imagery_context}
{snippets}
{kb_summary}

ANSWER:
{answer}

INSTRUCTIONS:
1. Read the Answer carefully and identify all factual claims.
2. For each claim, check if it is supported by the Context.
3. IMPORTANT: The Answer was generated by a VLM that CAN see the images listed in 'Available Imagery Tiles'. If the Answer describes visual details of these images (e.g. flooding, debris), assume those details are SUPPORTED by the existence of the image ID, unless directly contradicted by other text context.
4. ALLOW reasonable inferences based on general knowledge (e.g., "floodwaters recede slowly", "heavy rain causes flooding").
5. Calculate a faithfulness score from 0.0 to 1.0:
   - 1.0 = All claims are fully supported by the context
   - 0.7-0.9 = Most claims supported, minor unsupported details
   - 0.4-0.6 = Mix of supported and unsupported claims
   - 0.1-0.3 = Few claims supported, mostly unsupported
   - 0.0 = Completely unsupported or contradicts context

Respond in the following JSON format:
{{
    "reasoning": "Explain which claims are supported/unsupported and why...",
    "supported_claims": ["list of supported claims"],
    "unsupported_claims": ["list of unsupported claims"],
    "score": <float between 0.0 and 1.0>
}}"""

        try:
            response = self.generate_text(prompt)
            logger.debug(f"Faithfulness [{self.judge_name}]: {response[:200]}...")
            
            # Clean up markdown code blocks if present
            cleaned = response.replace("```json", "").replace("```", "").strip()
            data = json.loads(cleaned)
            
            return {
                "score": float(data["score"]),
                "reasoning": data.get("reasoning", ""),
                "supported_claims": data.get("supported_claims", []),
                "unsupported_claims": data.get("unsupported_claims", []),
            }
        except Exception as e:
            logger.warning(f"Faithfulness evaluation failed [{self.judge_name}]: {e}")
            return {"score": 0.0, "reasoning": f"Error: {e}", "supported_claims": [], "unsupported_claims": []}

    def evaluate_relevance(self, query: str, answer: str) -> Dict[str, Any]:
        """Evaluate relevance of answer to query.
        
        Returns:
            Dict with 'score' and optional 'reasoning'
        """
        prompt = f"""You are a helpful assistant evaluator.
Your task is to rate how well the ANSWER addresses the USER QUERY.

USER QUERY:
{query}

ANSWER:
{answer}

INSTRUCTIONS:
Rate the relevance on a scale of 0.0 to 1.0:
- 1.0: Perfectly answers the query with specific details.
- 0.7-0.9: Good answer but could be more specific.
- 0.4-0.6: Partially answers the query but misses key aspects.
- 0.1-0.3: Barely relevant, mostly off-topic.
- 0.0: Irrelevant or fails to answer the query.

Respond with ONLY a JSON object:
{{"score": <float between 0.0 and 1.0>, "reasoning": "Brief explanation"}}"""

        try:
            response = self.generate_text(prompt)
            logger.debug(f"Relevance [{self.judge_name}]: {response[:200]}...")
            
            # Try to parse as JSON first
            cleaned = response.replace("```json", "").replace("```", "").strip()
            try:
                data = json.loads(cleaned)
                return {"score": float(data["score"]), "reasoning": data.get("reasoning", "")}
            except json.JSONDecodeError:
                # Fall back to parsing just a number
                score = float(cleaned.strip())
                return {"score": score, "reasoning": ""}
        except Exception as e:
            logger.warning(f"Relevance evaluation failed [{self.judge_name}]: {e}")
            return {"score": 0.0, "reasoning": f"Error: {e}"}

    def evaluate(
        self,
        query: str,
        context: Dict[str, Any],
        answer: str,
        query_params: Dict[str, str] = None,
    ) -> Dict[str, Any]:
        """Full evaluation: Faithfulness + Relevance.
        
        Returns:
            Dict with 'faithfulness', 'relevance', and detailed sub-scores
        """
        faithfulness_result = self.evaluate_faithfulness(context, answer, query_params)
        relevance_result = self.evaluate_relevance(query, answer)
        
        return {
            "judge": self.judge_name,
            "faithfulness": faithfulness_result["score"],
            "relevance": relevance_result["score"],
            "faithfulness_detail": faithfulness_result,
            "relevance_detail": relevance_result,
        }


def get_available_judges() -> list[str]:
    """Return list of available judge names."""
    return list(JUDGE_CONFIGS.keys())


def create_judge(judge_name: str) -> JudgeClient:
    """Factory function to create a judge client."""
    return JudgeClient(judge_name)
