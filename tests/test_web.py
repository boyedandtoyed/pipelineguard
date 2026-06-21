import io
import pytest
from starlette.testclient import TestClient

from pipelineguard.web import app

client = TestClient(app)


def test_index_returns_dashboard():
    """GET / serves the HTML dashboard."""
    response = client.get("/")
    assert response.status_code == 200
    assert "PipelineGuard" in response.text


def test_health_is_ok():
    """GET /health reports ok status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_validate_endpoint_runs_checks():
    """POST /api/validate accepts a CSV and returns a checks list."""
    csv_bytes = b"col1,col2\n1,2\n3,4\n"
    response = client.post(
        "/api/validate",
        files={"file": ("sample.csv", io.BytesIO(csv_bytes), "text/csv")},
        data={"missing_threshold": "0.1"},
    )
    assert response.status_code == 200
    body = response.json()
    assert "checks" in body
    assert isinstance(body["checks"], list)
    assert len(body["checks"]) > 0
    # missing_values + duplicates should be present
    names = {c["name"] for c in body["checks"]}
    assert "missing_values" in names
    assert "duplicates" in names
    # summary fields
    assert body["total_rows"] == 2
    assert body["total_columns"] == 2
    assert "passed" in body


def test_validate_with_label():
    """POST /api/validate runs class_balance when a label is given.

    Uses string-valued labels so the library's counts dict has string keys
    (the CheckResult.details field is typed ``dict[str, Any]``).
    """
    csv_bytes = b"a,b,label\n1,2,yes\n3,4,no\n5,6,yes\n7,8,no\n"
    response = client.post(
        "/api/validate",
        files={"file": ("labeled.csv", io.BytesIO(csv_bytes), "text/csv")},
        data={"label": "label"},
    )
    assert response.status_code == 200
    names = {c["name"] for c in response.json()["checks"]}
    assert "class_balance" in names


def test_validate_bad_csv_returns_422():
    """POST /api/validate returns 422 on unparseable input."""
    response = client.post(
        "/api/validate",
        files={"file": ("bad.csv", io.BytesIO(b"\xff\xfe not a csv"), "text/csv")},
    )
    assert response.status_code == 422


def test_drift_endpoint():
    """POST /api/drift compares two datasets and returns a drift report."""
    ref = b"x\n" + b"\n".join(b"%d" % i for i in range(20)) + b"\n"
    cur = b"x\n" + b"\n".join(b"%d" % (i + 100) for i in range(20)) + b"\n"
    response = client.post(
        "/api/drift",
        files={
            "reference": ("ref.csv", io.BytesIO(ref), "text/csv"),
            "current": ("cur.csv", io.BytesIO(cur), "text/csv"),
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert "checks" in body and isinstance(body["checks"], list)
    names = {c["name"] for c in body["checks"]}
    assert "covariate_drift_ks" in names
