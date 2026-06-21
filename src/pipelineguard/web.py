"""FastAPI web layer for PipelineGuard.

Exposes a small HTML dashboard plus JSON endpoints that wrap the existing
``DataValidator`` and ``DriftDetector`` libraries. The library and CLI modules
are not modified; this module only imports and reuses them.
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path
import pandas as pd
import io

from .validator import DataValidator
from .drift import DriftDetector
from .models import ValidationReport

app = FastAPI(title="PipelineGuard", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_UI = Path(__file__).parent / "ui" / "index.html"


def _serialize(report: ValidationReport) -> dict:
    """Serialize a ValidationReport to a JSON-safe dict.

    ``model_dump(mode="json")`` converts the ``Severity`` enums to their string
    values. ``passed`` is a computed property so it is attached explicitly.
    """
    out = report.model_dump(mode="json")
    out["passed"] = report.passed
    return out


async def _read_csv(file: UploadFile) -> pd.DataFrame:
    """Parse an uploaded CSV; raises HTTPException(422) on bad input."""
    try:
        raw = await file.read()
        return pd.read_csv(io.BytesIO(raw))
    except Exception as exc:  # noqa: BLE001 - surface any parse error as 422
        raise HTTPException(
            status_code=422,
            detail=f"Could not parse CSV '{file.filename}': {exc}",
        )


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(_UI)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/validate")
async def validate(
    file: UploadFile = File(...),
    label: str | None = Form(default=None),
    missing_threshold: float = Form(default=0.1),
):
    """Validate a single uploaded CSV.

    Runs ``check_missing_values`` and ``check_duplicates`` always, plus
    ``check_class_balance`` when a label column is provided.
    """
    df = await _read_csv(file)

    validator = DataValidator(name=file.filename or "upload")
    validator.check_missing_values(df, threshold=missing_threshold)
    validator.check_duplicates(df)
    if label:
        validator.check_class_balance(df, label_col=label)

    report = validator.validate(df)
    return _serialize(report)


@app.post("/api/drift")
async def drift(
    reference: UploadFile = File(...),
    current: UploadFile = File(...),
    label: str | None = Form(default=None),
    alpha: float = Form(default=0.05),
):
    """Detect drift between a reference and a current dataset.

    Runs KS covariate drift on shared numeric columns, and chi-squared label
    drift when a label column is provided.
    """
    ref_df = await _read_csv(reference)
    cur_df = await _read_csv(current)

    detector = DriftDetector(reference=ref_df)
    detector.detect_covariate_drift(cur_df, alpha=alpha)
    if label:
        detector.detect_label_drift(cur_df, label_col=label, alpha=alpha)

    report = detector.report(name=f"{reference.filename} vs {current.filename}")
    return _serialize(report)
