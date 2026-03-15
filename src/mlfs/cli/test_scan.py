from pathlib import Path

import pytest
from typer.testing import CliRunner

from mlfs.cli.main import app

runner = CliRunner()


def test_scan_basic(tmp_path):
    # Create some test files
    (tmp_path / "file_a.txt").write_text("hello world")
    (tmp_path / "file_b.py").write_text("print('hi')" * 100)
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "nested.txt").write_text("nested content")

    result = runner.invoke(app, ["scan", str(tmp_path)])
    assert result.exit_code == 0
    assert "file_a.txt" in result.output
    assert "Total" in result.output


def test_scan_missing_path():
    result = runner.invoke(app, ["scan", "/nonexistent/path/xyz"])
    assert result.exit_code == 1


def test_scan_not_a_directory(tmp_path):
    f = tmp_path / "just_a_file.txt"
    f.write_text("hello")
    result = runner.invoke(app, ["scan", str(f)])
    assert result.exit_code == 1


def test_scan_empty_directory(tmp_path):
    result = runner.invoke(app, ["scan", str(tmp_path)])
    assert result.exit_code == 0
    assert "No files found" in result.output
