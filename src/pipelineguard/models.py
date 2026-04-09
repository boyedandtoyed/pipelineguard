from pydantic import BaseModel
from typing import Any
from enum import Enum

class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class CheckResult(BaseModel):
    name: str
    passed: bool
    severity: Severity
    message: str
    details: dict[str, Any] = {}

class ValidationReport(BaseModel):
    dataset_name: str
    total_rows: int
    total_columns: int
    checks: list[CheckResult] = []

    @property
    def passed(self) -> bool:
        return all(c.passed or c.severity in (Severity.INFO, Severity.WARNING) for c in self.checks)

    @property
    def errors(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed and c.severity in (Severity.ERROR, Severity.CRITICAL)]

    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed and c.severity == Severity.WARNING]
