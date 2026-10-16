import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from modelsight.cli import app, load_env_file

runner = CliRunner()


def test_dry_run_prints_the_plan_and_launches_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "exp.yaml"
    config.write_text(
        "target: modal\ngpu: B200\nimage: lmsysorg/sglang:test\n"
        "server: python -m sglang.launch_server --model-path Qwen/Qwen3.8-27B-FP8\n"
        "bench: aiperf profile --concurrency 32\n"
    )
    result = runner.invoke(app, ["run", "-f", str(config), "--dry-run"])
    assert result.exit_code == 0, result.stdout
    assert "target: modal" in result.stdout
    assert "gpu: B200" in result.stdout
    assert "lmsysorg/sglang:test" in result.stdout
    assert "sglang.launch_server" in result.stdout
    assert "aiperf profile" in result.stdout
    assert not (tmp_path / "results").exists()


def test_invalid_config_exits_nonzero(tmp_path: Path) -> None:
    config = tmp_path / "exp.yaml"
    config.write_text("target: local\nserver: sleep 1\n")
    result = runner.invoke(app, ["run", "-f", str(config), "--dry-run"])
    assert result.exit_code == 1
    assert "bench" in result.stderr


def test_env_file_sets_missing_variables_and_does_not_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("ALREADY_SET", "from-shell")
    (tmp_path / ".env").write_text(
        "# comment\n\nexport HF_TOKEN=\"hf_example\"\nALREADY_SET=from-file\nEMPTY=\n"
    )
    load_env_file(tmp_path / ".env")
    assert os.environ["HF_TOKEN"] == "hf_example"
    assert os.environ["ALREADY_SET"] == "from-shell"


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip()


def _local_experiment(tmp_path: Path, bench: str) -> Path:
    path = tmp_path / "exp.yaml"
    path.write_text(f"target: local\nserver: sleep 1\nbench: {bench}\n")
    return path


def test_dry_run_reports_missing_aiperf(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    config = _local_experiment(tmp_path, "aiperf profile")
    result = runner.invoke(app, ["run", "-f", str(config), "--dry-run"])
    assert result.exit_code == 0
    assert "aiperf: not found on PATH" in result.stdout


def test_run_stops_before_launching_when_aiperf_is_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    config = _local_experiment(tmp_path, "aiperf profile")
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["run", "-f", str(config)])
    assert result.exit_code == 1
    assert "uv tool install aiperf" in result.output
    assert not (tmp_path / "results").exists()


def test_dry_run_skips_check_for_other_bench_tools(tmp_path: Path) -> None:
    config = _local_experiment(tmp_path, "echo hi")
    result = runner.invoke(app, ["run", "-f", str(config), "--dry-run"])
    assert "aiperf: not used by this bench command" in result.stdout
