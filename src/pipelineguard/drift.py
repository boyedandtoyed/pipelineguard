import numpy as np
import pandas as pd
from scipy import stats
from .models import CheckResult, Severity, ValidationReport

class DriftDetector:
    """Detects data drift between a reference (train) and production dataset."""

    def __init__(self, reference: pd.DataFrame):
        self.reference = reference
        self._checks: list[CheckResult] = []

    def detect_covariate_drift(
        self,
        current: pd.DataFrame,
        columns: list[str] | None = None,
        alpha: float = 0.05,
    ) -> "DriftDetector":
        """Kolmogorov-Smirnov test for each numeric column."""
        cols = columns or list(self.reference.select_dtypes(include="number").columns)
        drifted: dict[str, float] = {}
        stable: list[str] = []
        for col in cols:
            if col not in current.columns or col not in self.reference.columns:
                continue
            ref_vals = self.reference[col].dropna().values
            cur_vals = current[col].dropna().values
            if len(ref_vals) < 5 or len(cur_vals) < 5:
                continue
            _, p_val = stats.ks_2samp(ref_vals, cur_vals)
            if p_val < alpha:
                drifted[col] = round(p_val, 6)
            else:
                stable.append(col)
        passed = len(drifted) == 0
        self._checks.append(CheckResult(
            name="covariate_drift_ks",
            passed=passed,
            severity=Severity.WARNING if not passed else Severity.INFO,
            message=f"KS drift test (α={alpha}): {len(drifted)} drifted, {len(stable)} stable",
            details={"drifted_columns": drifted, "stable_columns": stable},
        ))
        return self

    def detect_label_drift(
        self,
        current: pd.DataFrame,
        label_col: str,
        alpha: float = 0.05,
    ) -> "DriftDetector":
        """Chi-squared test on label distribution."""
        if label_col not in self.reference.columns or label_col not in current.columns:
            self._checks.append(CheckResult(
                name="label_drift",
                passed=False,
                severity=Severity.ERROR,
                message=f"Label column '{label_col}' not found in both datasets",
            ))
            return self
        ref_counts = self.reference[label_col].value_counts().sort_index()
        cur_counts = current[label_col].value_counts().sort_index()
        all_labels = ref_counts.index.union(cur_counts.index)
        ref_freq = np.array([ref_counts.get(l, 0) for l in all_labels], dtype=float)
        cur_freq = np.array([cur_counts.get(l, 0) for l in all_labels], dtype=float)
        ref_freq = ref_freq / ref_freq.sum()
        cur_freq_abs = cur_freq
        _, p_val = stats.chisquare(f_obs=cur_freq_abs, f_exp=ref_freq * cur_freq_abs.sum())
        passed = p_val >= alpha
        self._checks.append(CheckResult(
            name="label_drift",
            passed=passed,
            severity=Severity.WARNING if not passed else Severity.INFO,
            message=f"Label distribution drift (χ² p={p_val:.4f}, α={alpha})",
            details={"p_value": float(p_val), "reference_dist": ref_freq.tolist(), "current_dist": (cur_freq/cur_freq.sum()).tolist()},
        ))
        return self

    def report(self, name: str = "drift_report") -> ValidationReport:
        ref = self.reference
        return ValidationReport(
            dataset_name=name,
            total_rows=len(ref),
            total_columns=len(ref.columns),
            checks=list(self._checks),
        )
