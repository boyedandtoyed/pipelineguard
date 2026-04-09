import pandas as pd
import numpy as np
import pytest
from pipelineguard.validator import DataValidator
from pipelineguard.models import Severity

@pytest.fixture
def clean_df():
    np.random.seed(42)
    return pd.DataFrame({
        "age": np.random.randint(18, 80, 200).astype(float),
        "income": np.random.uniform(20000, 100000, 200),
        "label": np.random.choice([0, 1], 200, p=[0.5, 0.5]),
    })

def test_missing_values_passes(clean_df):
    v = DataValidator("test")
    v.check_missing_values(clean_df)
    report = v.validate(clean_df)
    check = next(c for c in report.checks if c.name == "missing_values")
    assert check.passed

def test_missing_values_fails():
    df = pd.DataFrame({"a": [1, None, None, None, None], "b": [1, 2, 3, 4, 5]})
    v = DataValidator("test")
    v.check_missing_values(df, threshold=0.5)
    report = v.validate(df)
    check = next(c for c in report.checks if c.name == "missing_values")
    assert not check.passed
    assert check.severity == Severity.ERROR

def test_class_balance_passes(clean_df):
    v = DataValidator("test")
    v.check_class_balance(clean_df, label_col="label", max_ratio=3.0)
    report = v.validate(clean_df)
    check = next(c for c in report.checks if c.name == "class_balance")
    assert check.passed

def test_class_balance_fails():
    df = pd.DataFrame({"label": [0]*95 + [1]*5})
    v = DataValidator("test")
    v.check_class_balance(df, "label", max_ratio=5.0)
    report = v.validate(df)
    check = next(c for c in report.checks if c.name == "class_balance")
    assert not check.passed

def test_schema_check_passes(clean_df):
    v = DataValidator("test")
    v.check_schema(clean_df, {"age": "float", "income": "float", "label": "int"})
    report = v.validate(clean_df)
    check = next(c for c in report.checks if c.name == "schema")
    # Schema may warn on dtype details but missing_cols check should pass
    assert "missing_columns" not in check.details or len(check.details["missing_columns"]) == 0

def test_duplicates_detected():
    df = pd.DataFrame({"a": [1, 1, 2, 3, 4]})
    v = DataValidator("test")
    v.check_duplicates(df, threshold=0.0)
    report = v.validate(df)
    check = next(c for c in report.checks if c.name == "duplicates")
    assert not check.passed
