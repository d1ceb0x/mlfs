from pathlib import Path

import typer
from rich import print as rprint
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.table import Table

console = Console()


def _format_size(bytes: int) -> str:
    """Human-readable file size."""
    for unit in ("B", "KB", "MB", "GB"):
        if bytes < 1024:
            return f"{bytes:.1f} {unit}"
        bytes /= 1024
    return f"{bytes:.1f} TB"


def scan_directory(path: Path) -> tuple[list[tuple[Path, int]], list[Path]]:
    """Recursively scan a directory. Returns (files, skipped)."""
    files = []
    skipped = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]Scanning[/] {task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        transient=True,
        console=console,
    ) as progress:
        task = progress.add_task(str(path), total=None)

        for item in path.rglob("*"):
            try:
                if item.is_file():
                    size = item.stat().st_size
                    files.append((item, size))
                    progress.update(task, description=str(item.parent.name))
            except PermissionError:
                skipped.append(item)

    return files, skipped


def run_scan(path: Path):
    """Core scan logic — separated for testability."""
    if not path.exists():
        rprint(f"[red]Error:[/] path does not exist: {path}")
        raise typer.Exit(1)

    if not path.is_dir():
        rprint(f"[red]Error:[/] not a directory: {path}")
        raise typer.Exit(1)

    rprint(f"\n[bold cyan]mlfs scan[/] → [green]{path}[/]\n")

    files, skipped = scan_directory(path)

    if not files:
        rprint("[yellow]No files found.[/]")
        return

    # Sort by size descending
    files.sort(key=lambda x: x[1], reverse=True)

    # Build table
    table = Table(show_header=True, header_style="bold cyan", expand=False)
    table.add_column("File", style="white", no_wrap=False, ratio=4)
    table.add_column("Size", style="green", justify="right", ratio=1)

    for file_path, size in files[:50]:  # cap display at 50
        try:
            rel = file_path.relative_to(path)
        except ValueError:
            rel = file_path
        table.add_row(str(rel), _format_size(size))

    console.print(table)

    if len(files) > 50:
        rprint(f"[dim]... and {len(files) - 50} more files[/]")

    # Summary
    total_size = sum(s for _, s in files)
    rprint(
        f"\n[bold]Total:[/] [cyan]{len(files)}[/] files · [green]{_format_size(total_size)}[/]"
    )

    if skipped:
        rprint(f"[yellow]⚠ Skipped {len(skipped)} path(s) due to permission errors[/]")


def register(app: typer.Typer):
    @app.command()
    def scan(
        path: Path = typer.Argument(..., help="Directory to scan"),
    ):
        """Recursively scan a directory and report files by size."""
        run_scan(path)
