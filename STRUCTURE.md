# PipelineGuard — Project Structure

ML pipeline validation and data drift detection. Library + CLI + FastAPI web UI.

## `src/pipelineguard/`

| File | Purpose |
|------|---------|
| `__init__.py` | Package entrypoint. Exports `DataValidator`, `DriftDetector`, `ValidationReport`; defines `__version__`. |
| `models.py` | Pydantic v2 data models: `Severity` enum (`info`/`warning`/`error`/`critical`), `CheckResult` (name/passed/severity/message/details), `ValidationReport` (dataset metadata + `checks` list) with computed `.passed`, `.errors`, `.warnings` properties. |
| `validator.py` | `DataValidator` class — chained data-quality checks: `check_missing_values`, `check_class_balance`, `check_schema`, `check_duplicates`, `check_feature_ranges`, `.validate(df) -> ValidationReport`. |
| `drift.py` | `DriftDetector(reference)` class — `detect_covariate_drift` (Kolmogorov–Smirnov), `detect_label_drift` (chi-squared), `.report(name) -> ValidationReport`. |
| `report.py` | `print_report(report)` — rich-formatted terminal rendering of a `ValidationReport`. |
| `cli.py` | Typer CLI `app` with two commands: `validate` (single CSV, optional `--label`/`--schema`) and `drift` (reference + current CSV). Registered as `pipelineguard` console script. |
| `web.py` | **FastAPI app** (`app`). Endpoints: `GET /` (dashboard), `GET /health`, `POST /api/validate`, `POST /api/drift`. Reuses `DataValidator`/`DriftDetector`; wraps CSV parsing with 422 error handling; serializes reports via `_serialize` (attaches `passed`, JSON-encodes `Severity` enums). |
| `ui/index.html` | Self-contained dark dashboard (vanilla JS/CSS, no CDN). Validate tab (CSV upload + optional label + missing-threshold slider), Drift tab (two CSVs + alpha slider), status pill, pass/fail banner, checks table, drifted-columns list. |

## `tests/`

| File | Purpose |
|------|---------|
| `__init__.py` | Empty test-package marker. |
| `test_validator.py` | `DataValidator` checks: missing values (pass/fail), class balance, schema, duplicates. |
| `test_drift.py` | `DriftDetector` covariate and label drift behaviour. |
| `test_web.py` | FastAPI tests via `starlette.testclient.TestClient(app)`: `GET /`, `GET /health`, `POST /api/validate` (CSV + checks list), label branch, 422 on bad CSV, `POST /api/drift`. |

## Root files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Project metadata + dependencies (numpy, pandas, scipy, scikit-learn, rich, typer, pydantic, fastapi, uvicorn[standard], python-multipart, httpx). Defines `pipelineguard` console script and pytest config. |
| `Dockerfile` | `python:3.11-slim` image; installs deps from `pyproject.toml`, copies `src/`, sets `PYTHONPATH=/app/src`, exposes **8000**, runs `uvicorn pipelineguard.web:app`. |
| `docker-compose.yml` | Single `pipelineguard` service mapping host **3002** → container **8000**, `restart: unless-stopped`, `/health` healthcheck. |
| `README.md` | Project overview, install, CLI + Python API examples. |
| `STRUCTURE.md` | This file. |

## How to run

### Install (editable)
```bash
pip install -e .
```

### CLI
```bash
pipelineguard validate data/train.csv --label target
pipelineguard drift data/train.csv data/production.csv --label target --alpha 0.05
```

### Local web server
```bash
uvicorn pipelineguard.web:app --port 8000
# open http://127.0.0.1:8000
```
Endpoints: `GET /` (UI), `GET /health`, `POST /api/validate`, `POST /api/drift`.

### Docker
```bash
docker compose up -d --build
# open http://127.0.0.1:3002
```
The container listens on port **8000** internally; `docker-compose.yml` maps the host port **3002** to it. So the dashboard is served at **http://127.0.0.1:3002**.

### Tests
```bash
pytest tests/ -v
```
