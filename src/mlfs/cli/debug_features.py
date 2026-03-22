import typer
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich import print as rprint

from mlfs.core.features import extract_all, ExtensionRegistry, FEATURE_NAMES

console = Console()


def register(app: typer.Typer):
    @app.command(name="debug-features")
    def debug_features(
        path: Path = typer.Argument(..., help="Directory to extract features from"),
        limit: int = typer.Option(20, "--limit", "-n", help="Max files to display"),
    ):
        """Extract and display raw feature vectors for files in a directory."""

        if not path.exists() or not path.is_dir():
            rprint(f"[red]Error:[/] invalid directory: {path}")
            raise typer.Exit(1)

        rprint(f"\n[bold cyan]mlfs debug-features[/] → [green]{path}[/]\n")

        # Collect files
        files = [p for p in path.rglob("*") if p.is_file()]
        if not files:
            rprint("[yellow]No files found.[/]")
            return

        # Extract features
        registry = ExtensionRegistry()
        matrix, registry = extract_all(files[:limit], root=path, registry=registry)

        if matrix.shape[0] == 0:
            rprint("[yellow]Could not extract features from any files.[/]")
            return

        # Build table
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("File", style="white", no_wrap=True, max_width=40)
        for name in FEATURE_NAMES:
            table.add_column(name, justify="right", style="green")

        for i, file_path in enumerate(files[:matrix.shape[0]]):
            vec = matrix[i]
            try:
                label = str(file_path.relative_to(path))
            except ValueError:
                label = file_path.name

            # Truncate long paths for display
            if len(label) > 38:
                label = "..." + label[-35:]

            table.add_row(label, *[f"{v:.1f}" for v in vec])

        console.print(table)

        # Extension mapping
        rprint(f"\n[bold]Extension registry:[/]")
        for ext, eid in sorted(registry.mapping.items(), key=lambda x: x[1]):
            ext_display = ext if ext else "(no ext)"
            rprint(f"  [dim]{eid:3d}[/] → [cyan]{ext_display}[/]")

        rprint(f"\n[bold]Matrix shape:[/] [cyan]{matrix.shape}[/] "
               f"([green]{matrix.shape[0]}[/] files × "
               f"[green]{matrix.shape[1]}[/] features)")
