from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from .models import ValidationReport, Severity

console = Console()

SEVERITY_COLORS = {
    Severity.INFO: "cyan",
    Severity.WARNING: "yellow",
    Severity.ERROR: "red",
    Severity.CRITICAL: "bold red",
}

def print_report(report: ValidationReport) -> None:
    status = "[green]✓ PASSED[/green]" if report.passed else "[red]✗ FAILED[/red]"
    console.print(Panel(
        f"[bold]{report.dataset_name}[/bold]  |  {report.total_rows:,} rows  ×  {report.total_columns} columns  |  {status}",
        title="[bold blue]PipelineGuard Validation Report[/bold blue]",
        border_style="blue",
    ))
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold blue")
    table.add_column("Check", style="white", min_width=24)
    table.add_column("Status", justify="center", width=10)
    table.add_column("Severity", justify="center", width=10)
    table.add_column("Message", style="dim")
    for check in report.checks:
        status_icon = "[green]✓[/green]" if check.passed else "[red]✗[/red]"
        sev_color = SEVERITY_COLORS.get(check.severity, "white")
        table.add_row(
            check.name,
            status_icon,
            f"[{sev_color}]{check.severity.value}[/{sev_color}]",
            check.message,
        )
    console.print(table)
    if report.errors:
        console.print("\n[bold red]Errors:[/bold red]")
        for e in report.errors:
            console.print(f"  • [red]{e.name}[/red]: {e.message}")
            if e.details:
                for k, v in e.details.items():
                    console.print(f"    [dim]{k}:[/dim] {v}")
    console.print()
