import typer
import pandas as pd
import json
from pathlib import Path
from rich.console import Console
from .validator import DataValidator
from .drift import DriftDetector
from .report import print_report

app = typer.Typer(name="pipelineguard", help="ML pipeline validation and drift detection")
console = Console()

@app.command()
def validate(
    data: Path = typer.Argument(..., help="Path to CSV dataset"),
    label: str = typer.Option(None, "--label", "-l", help="Label column name"),
    missing_threshold: float = typer.Option(0.1, "--missing", help="Max null fraction per column"),
    schema_file: Path = typer.Option(None, "--schema", "-s", help="JSON file with expected column dtypes"),
    max_imbalance: float = typer.Option(10.0, "--max-imbalance", help="Max class imbalance ratio"),
) -> None:
    """Validate a dataset CSV against quality checks."""
    if not data.exists():
        console.print(f"[red]File not found: {data}[/red]")
        raise typer.Exit(1)

    df = pd.read_csv(data)
    console.print(f"[dim]Loaded {len(df):,} rows × {len(df.columns)} columns from {data.name}[/dim]")

    v = DataValidator(name=data.name)
    v.check_missing_values(df, threshold=missing_threshold)
    v.check_duplicates(df)
    if label:
        v.check_class_balance(df, label_col=label, max_ratio=max_imbalance)
    if schema_file and schema_file.exists():
        schema = json.loads(schema_file.read_text())
        v.check_schema(df, expected=schema)

    report = v.validate(df)
    print_report(report)
    raise typer.Exit(0 if report.passed else 1)

@app.command()
def drift(
    reference: Path = typer.Argument(..., help="Reference (training) CSV"),
    current: Path = typer.Argument(..., help="Current (production) CSV"),
    label: str = typer.Option(None, "--label", "-l", help="Label column"),
    alpha: float = typer.Option(0.05, "--alpha", help="Statistical significance level"),
) -> None:
    """Detect data drift between reference and current datasets."""
    for p in [reference, current]:
        if not p.exists():
            console.print(f"[red]File not found: {p}[/red]")
            raise typer.Exit(1)

    ref_df = pd.read_csv(reference)
    cur_df = pd.read_csv(current)

    detector = DriftDetector(reference=ref_df)
    detector.detect_covariate_drift(cur_df, alpha=alpha)
    if label:
        detector.detect_label_drift(cur_df, label_col=label, alpha=alpha)

    report = detector.report(name=f"{reference.name} vs {current.name}")
    print_report(report)
    raise typer.Exit(0 if report.passed else 1)

if __name__ == "__main__":
    app()
