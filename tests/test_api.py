from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.core.dataio.storage import DataLocator
from app.core.retrieval.retriever import Retriever
from app.core.models.vlm_client import VLMClient


def setup_module(module):
    locator = DataLocator(Path(__file__).resolve().parents[1] / "data")
    app.state.locator = locator
    app.state.retriever = Retriever(locator)
    app.state.vlm_client = VLMClient("mock")


def test_healthz():
    client = TestClient(app)
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_query_endpoint():
    client = TestClient(app)
    payload = {"zip": "77002", "start": "2017-08-28", "end": "2017-09-03", "k_tiles": 4, "n_text": 5}
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["zip"] == "77002"
    assert data["evidence_refs"]["imagery_tile_ids"], "Imagery references required"


def test_chat_endpoint_parses_language():
    client = TestClient(app)
    message = 'Please summarize Harvey impacts for zip 77002 between Aug 28 2017 and Sept 3 2017.'
    response = client.post('/chat', json={'message': message, 'k_tiles': 4, 'n_text': 5})
    assert response.status_code == 200
    body = response.json()
    assert body['query']['zip'] == '77002'
    assert body['answer']['zip'] == '77002'
    assert body['answer']['evidence_refs']['imagery_tile_ids'], 'Imagery references required'


def test_chat_endpoint_requires_zip():
    client = TestClient(app)
    response = client.post('/chat', json={'message': 'Tell me about Harvey flooding'})
    assert response.status_code == 400
    assert 'ZIP' in response.json()['detail']
