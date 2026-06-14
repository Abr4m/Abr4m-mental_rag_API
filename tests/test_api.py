"""
Tests: API Integration
Uses TestClient with a mocked AppContainer (no real models loaded).
"""


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_tags(client):
    resp = client.get("/tags")
    assert resp.status_code == 200
    assert "All" in resp.json()["tags"]


def test_query_success(client):
    resp = client.post("/query", json={"query": "I feel anxious all the time"})
    assert resp.status_code == 200
    body = resp.json()
    assert "answer" in body
    assert isinstance(body["sources"], list)


def test_query_validation_too_short(client):
    resp = client.post("/query", json={"query": "hi"})
    assert resp.status_code == 422


def test_query_n_tips_out_of_range(client):
    resp = client.post("/query", json={"query": "I feel anxious", "n_tips": 99})
    assert resp.status_code == 422
