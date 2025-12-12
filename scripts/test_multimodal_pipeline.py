"""Test script for Multimodal Split-Pipeline orchestration."""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.models.split_client import SplitPipelineClient


class TestMultimodalPipeline(unittest.TestCase):
    def setUp(self):
        import os
        os.environ["OPENAI_API_KEY"] = "dummy-key"
        
        # Mock components
        self.mock_text_client = MagicMock()
        self.mock_visual_client = MagicMock()
        self.mock_fusion_engine = MagicMock()
        self.mock_verifier = MagicMock()

        # Setup return values
        self.mock_text_client.analyze.return_value = {
            "reasoning": "Text reasoning",
            "estimates": {"structural_damage_pct": 50.0},
            "evidence_refs": {"tweet_ids": ["t1"]}
        }
        self.mock_visual_client.analyze.return_value = {
            "overall_assessment": {"structural_damage_pct": 60.0},
            "evidence_refs": {"imagery_tile_ids": ["img1"]}
        }
        self.mock_fusion_engine.fuse.return_value = {
            "reasoning": "Fused reasoning",
            "estimates": {"structural_damage_pct": 55.0},
            "evidence_refs": {"tweet_ids": ["t1"], "imagery_tile_ids": ["img1"]}
        }
        self.mock_verifier.verify.return_value = {
            "reasoning": "Verified reasoning",
            "estimates": {"structural_damage_pct": 55.0},
            "evidence_refs": {"tweet_ids": ["t1"], "imagery_tile_ids": ["img1"]},
            "natural_language_summary": "Verified summary"
        }

    @patch("app.core.models.split_client.TextAnalysisClient")
    @patch("app.core.models.visual_client.VisualAnalysisClient")
    @patch("app.core.models.fusion_engine.FusionEngine")
    @patch("app.core.models.split_client.Verifier")
    def test_pipeline_orchestration(self, MockVerifier, MockFusion, MockVisual, MockText):
        # Configure mocks to return our instances
        MockText.return_value = self.mock_text_client
        MockVisual.return_value = self.mock_visual_client
        MockFusion.return_value = self.mock_fusion_engine
        MockVerifier.return_value = self.mock_verifier

        # Initialize client
        client = SplitPipelineClient(provider="openai", enable_visual=True)
        
        # Force initialization of visual components
        client._init_visual_components()
        # Inject our mocks (since _init_visual_components might create new instances)
        client._visual_client = self.mock_visual_client
        client._fusion_engine = self.mock_fusion_engine

        # Test data
        zip_code = "77002"
        time_window = {"start": "2017-08-25", "end": "2017-08-30"}
        imagery_tiles = [{"tile_id": "img1", "uri": "path/to/img1.jpg"}]
        context = {
            "imagery_tiles": imagery_tiles,
            "tweets": [{"doc_id": "t1", "text": "flood"}],
            "calls": [],
            "sensors": [],
            "fema": []
        }

        # Run inference
        result = client.infer(
            zip_code=zip_code,
            time_window=time_window,
            imagery_tiles=imagery_tiles,
            text_snippets=[],
            sensor_table="",
            kb_summary="",
            tweets=context["tweets"],
            project_root=Path("/tmp")
        )

        # Verify calls
        self.mock_text_client.analyze.assert_called_once()
        self.mock_visual_client.analyze.assert_called_once()
        self.mock_fusion_engine.fuse.assert_called_once()
        self.mock_verifier.verify.assert_called_once()

        # Verify result
        self.assertEqual(result["natural_language_summary"], "Verified summary")
        self.assertEqual(result["estimates"]["structural_damage_pct"], 55.0)
        print("\nTest passed: Pipeline successfully orchestrated Text -> Visual -> Fusion -> Verification")


if __name__ == "__main__":
    unittest.main()
