import pandas as pd
import numpy as np
from pipelineguard.drift import DriftDetector

def test_no_drift():
    np.random.seed(0)
    ref = pd.DataFrame({"x": np.random.normal(0, 1, 500), "y": np.random.normal(5, 2, 500)})
    cur = pd.DataFrame({"x": np.random.normal(0, 1, 200), "y": np.random.normal(5, 2, 200)})
    detector = DriftDetector(ref)
    detector.detect_covariate_drift(cur, alpha=0.01)
    report = detector.report()
    check = next(c for c in report.checks if c.name == "covariate_drift_ks")
    assert check.passed

def test_drift_detected():
    np.random.seed(42)
    ref = pd.DataFrame({"x": np.random.normal(0, 1, 500)})
    cur = pd.DataFrame({"x": np.random.normal(5, 1, 200)})  # shifted mean
    detector = DriftDetector(ref)
    detector.detect_covariate_drift(cur, alpha=0.05)
    report = detector.report()
    check = next(c for c in report.checks if c.name == "covariate_drift_ks")
    assert not check.passed
    assert "x" in check.details["drifted_columns"]
