from typer.testing import CliRunner
from mlfs.cli.main import app
from mlfs import __version__

runner = CliRunner()

def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "mlfs" in result.output

def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output
