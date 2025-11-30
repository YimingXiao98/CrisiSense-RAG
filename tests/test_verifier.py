"""Unit tests for Verifier component."""

import pytest
from app.core.models.verifier import Verifier


@pytest.fixture
def verifier():
    return Verifier()


def test_verify_tweet_ids(verifier):
    context = {
        "tweets": [
            {"tweet_id": "123", "text": "Flood here"},
            {"tweet_id": "456", "text": "Storm coming"},
        ]
    }
    response = {"evidence_refs": {"tweet_ids": ["123", "789", "456"]}}

    verified = verifier.verify(response, context)
    assert verified["evidence_refs"]["tweet_ids"] == ["123", "456"]


def test_verify_call_ids_exact(verifier):
    context = {
        "calls": [
            {"record_id": "1001", "description": "Flooding"},
            {"record_id": "1002", "description": "Damage"},
        ]
    }
    response = {"evidence_refs": {"call_311_ids": ["1001", "9999"]}}

    verified = verifier.verify(response, context)
    assert verified["evidence_refs"]["call_311_ids"] == ["1001"]


def test_verify_call_ids_fuzzy(verifier):
    context = {"calls": [{"record_id": "1001", "description": "Flooding"}]}
    # Simulate the hallucinated prefix issue
    response = {"evidence_refs": {"call_311_ids": ["12345-1001"]}}

    verified = verifier.verify(response, context)
    assert verified["evidence_refs"]["call_311_ids"] == ["1001"]


def test_verify_sensor_ids(verifier):
    context = {
        "sensors": [{"sensor_id": 10, "value": 5.0}, {"sensor_id": 20, "value": 0.0}]
    }
    response = {"evidence_refs": {"sensor_ids": [10, 30]}}

    verified = verifier.verify(response, context)
    # Note: IDs are converted to strings in Verifier
    assert verified["evidence_refs"]["sensor_ids"] == ["10"]


def test_verify_empty_context(verifier):
    context = {}
    response = {"evidence_refs": {"tweet_ids": ["123"]}}
    verified = verifier.verify(response, context)
    assert verified["evidence_refs"]["tweet_ids"] == []
