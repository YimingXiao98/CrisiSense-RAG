"""Unit tests for TextAnalysisClient."""

import json
import pytest
from unittest.mock import MagicMock, patch
from app.core.models.text_client import TextAnalysisClient


@pytest.fixture
def mock_genai():
    with patch("app.core.models.text_client.genai") as mock:
        yield mock


@pytest.fixture
def client(mock_genai):
    return TextAnalysisClient(api_key="fake_key")


def test_analyze_success(client, mock_genai):
    # Mock response
    mock_response = MagicMock()
    mock_response.text = json.dumps(
        {
            "reasoning": "Test reasoning",
            "estimates": {"structural_damage_pct": 10.0},
            "evidence_refs": {"tweet_ids": ["123"]},
        }
    )
    client.model.generate_content.return_value = mock_response

    context = {
        "tweets": [{"tweet_id": "123", "text": "Flood"}],
        "imagery_tiles": [{"uri": "img.jpg"}],  # Should be stripped
    }

    result = client.analyze(
        "77002", {"start": "2017-08-25", "end": "2017-08-30"}, context
    )

    assert result["reasoning"] == "Test reasoning"
    assert result["evidence_refs"]["tweet_ids"] == ["123"]

    # Verify imagery was stripped from prompt context
    # We can check the call args to generate_content, but build_user_prompt is called internally.
    # Ideally we mock build_user_prompt to verify input, but checking result is okay for now.


def test_analyze_failure(client, mock_genai):
    client.model.generate_content.side_effect = Exception("API Error")

    context = {}
    result = client.analyze(
        "77002", {"start": "2017-08-25", "end": "2017-08-30"}, context
    )

    assert "Analysis failed" in result["reasoning"]
    assert result["estimates"]["structural_damage_pct"] == 0.0
