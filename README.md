# PipelineGuard

ML pipeline validation and data drift detection library. Catches data quality issues and distribution shifts before they silently degrade your models.

## Features
- Missing value, duplicate, schema, range, class imbalance checks
- Kolmogorov-Smirnov covariate drift detection
- Chi-squared label distribution drift
- Rich terminal reports
- CLI (`pipelineguard validate / drift`) + Python API

## Installation & Running

### Prerequisites
- Python 3.11+

### Quick Start
```bash
git clone https://github.com/boyedandtoyed/pipelineguard.git
cd pipelineguard

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -e .

# CLI usage
pipelineguard validate data/train.csv --label target
pipelineguard drift data/train.csv data/production.csv --label target
```

### Python API
```python
import pandas as pd
from pipelineguard import DataValidator, DriftDetector

df = pd.read_csv("data/train.csv")
report = (
    DataValidator("my_dataset")
    .check_missing_values(df)
    .check_class_balance(df, label_col="target")
    .check_duplicates(df)
    .validate(df)
)
```

### Running Tests
```bash
pip install pytest
pytest tests/ -v
```

## Tech Stack
NumPy · Pandas · SciPy · scikit-learn · Rich · Typer · Pydantic v2
