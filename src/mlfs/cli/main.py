import typer
from rich import print as rprint
from rich.console import Console

from mlfs.cli import debug_features, scan

app = typer.Typer(
    name="mlfs",
    help="[bold cyan]mlfs[/] — local, privacy-first file intelligence CLI.",
    rich_markup_mode="rich",
    add_completion=False,
)
scan.register(app)
debug_features.register(app)
console = Console()


@app.callback(invoke_without_command=True)
def root(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        rprint(
            "\n[bold cyan]mlfs[/] — ML-powered file system intelligence\n\n"
            "Run [green]mlfs --help[/] to see available commands.\n"
        )


@app.command()
def version():
    """Show the current mlfs version."""
    from mlfs import __version__

    rprint(f"[bold cyan]mlfs[/] v{__version__}")


def main():
    app()


if __name__ == "__main__":
    main()
