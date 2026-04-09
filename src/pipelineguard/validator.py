import pandas as pd
import numpy as np
from .models import CheckResult, Severity, ValidationReport

class DataValidator:
    """Validates a DataFrame against a set of configurable rules."""

    def __init__(self, name: str = "dataset"):
        self.name = name
        self._checks: list[CheckResult] = []

    def _add(self, result: CheckResult) -> None:
        self._checks.append(result)

    def check_missing_values(self, df: pd.DataFrame, threshold: float = 0.1) -> "DataValidator":
        """Flag columns with more than `threshold` fraction of nulls."""
        null_rates = df.isnull().mean()
        violations = null_rates[null_rates > threshold]
        if violations.empty:
            self._add(CheckResult(
                name="missing_values",
                passed=True,
                severity=Severity.INFO,
                message=f"No columns exceed {threshold:.0%} null threshold",
                details={"max_null_rate": float(null_rates.max())},
            ))
        else:
            self._add(CheckResult(
                name="missing_values",
                passed=False,
                severity=Severity.ERROR,
                message=f"{len(violations)} column(s) exceed {threshold:.0%} null threshold",
                details={col: f"{rate:.2%}" for col, rate in violations.items()},
            ))
        return self

    def check_class_balance(self, df: pd.DataFrame, label_col: str, max_ratio: float = 10.0) -> "DataValidator":
        """Check that no class is more than max_ratio times more frequent than another."""
        if label_col not in df.columns:
            self._add(CheckResult(
                name="class_balance",
                passed=False,
                severity=Severity.ERROR,
                message=f"Label column '{label_col}' not found",
            ))
            return self
        counts = df[label_col].value_counts()
        ratio = counts.iloc[0] / counts.iloc[-1] if len(counts) > 1 else 1.0
        passed = ratio <= max_ratio
        self._add(CheckResult(
            name="class_balance",
            passed=passed,
            severity=Severity.WARNING if not passed else Severity.INFO,
            message=f"Class imbalance ratio: {ratio:.1f}x (threshold {max_ratio:.1f}x)",
            details=counts.to_dict(),
        ))
        return self

    def check_schema(self, df: pd.DataFrame, expected: dict[str, str]) -> "DataValidator":
        """Validate column presence and dtype compatibility."""
        missing_cols = set(expected) - set(df.columns)
        wrong_types: dict[str, str] = {}
        for col, expected_dtype in expected.items():
            if col in df.columns and not str(df[col].dtype).startswith(expected_dtype):
                wrong_types[col] = f"expected {expected_dtype}, got {df[col].dtype}"
        passed = not missing_cols and not wrong_types
        details: dict = {}
        if missing_cols:
            details["missing_columns"] = list(missing_cols)
        if wrong_types:
            details["wrong_types"] = wrong_types
        self._add(CheckResult(
            name="schema",
            passed=passed,
            severity=Severity.CRITICAL if missing_cols else (Severity.WARNING if wrong_types else Severity.INFO),
            message="Schema valid" if passed else f"Schema violations: {len(missing_cols)} missing, {len(wrong_types)} wrong types",
            details=details,
        ))
        return self

    def check_duplicates(self, df: pd.DataFrame, threshold: float = 0.01) -> "DataValidator":
        """Check for duplicate rows."""
        dup_rate = df.duplicated().mean()
        passed = dup_rate <= threshold
        self._add(CheckResult(
            name="duplicates",
            passed=passed,
            severity=Severity.WARNING if not passed else Severity.INFO,
            message=f"Duplicate row rate: {dup_rate:.2%}",
            details={"duplicate_rows": int(df.duplicated().sum()), "rate": float(dup_rate)},
        ))
        return self

    def check_feature_ranges(self, df: pd.DataFrame, ranges: dict[str, tuple[float, float]]) -> "DataValidator":
        """Check that numeric features stay within expected [min, max] bounds."""
        violations: dict[str, str] = {}
        for col, (lo, hi) in ranges.items():
            if col not in df.columns:
                violations[col] = "column not found"
                continue
            actual_min, actual_max = df[col].min(), df[col].max()
            if actual_min < lo or actual_max > hi:
                violations[col] = f"range [{actual_min:.3f}, {actual_max:.3f}] outside [{lo}, {hi}]"
        passed = not violations
        self._add(CheckResult(
            name="feature_ranges",
            passed=passed,
            severity=Severity.ERROR if not passed else Severity.INFO,
            message=f"Feature range check: {len(violations)} violation(s)",
            details=violations,
        ))
        return self

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        return ValidationReport(
            dataset_name=self.name,
            total_rows=len(df),
            total_columns=len(df.columns),
            checks=list(self._checks),
        )
