import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from modelsight.build import BuildError
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


SHA = "c" * 40
DEEPINFER_YAML = """target: modal
gpu: B200:8
dockerfile: Dockerfile
secrets: [HF_TOKEN]
report: [uname -a]
cache:
  prefetch: [org/model]
sources:
  DYNAMO:
    repo: https://github.com/usationlabs/dynamo
    ref: feat/deepinfer
server: serve
bench: bench
"""


def _deepinfer_config(tmp_path: Path) -> Path:
    (tmp_path / "Dockerfile").write_text("FROM scratch\n")
    config = tmp_path / "exp.yaml"
    config.write_text(DEEPINFER_YAML)
    return config


@pytest.fixture
def fake_refs(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    asked: list[tuple[str, str]] = []

    def resolve(repo: str, ref: str) -> str:
        asked.append((repo, ref))
        return SHA

    monkeypatch.setattr("modelsight.build.resolve_ref", resolve)
    return asked


def test_dry_run_shows_the_image_plan_and_resolved_refs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_refs
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")
    config = _deepinfer_config(tmp_path)
    result = runner.invoke(app, ["run", "-f", str(config), "--dry-run"])
    assert result.exit_code == 0, result.stdout
    assert "image_fingerprint:" in result.stdout
    assert "dockerfile:" in result.stdout
    assert f"source DYNAMO: https://github.com/usationlabs/dynamo feat/deepinfer -> {SHA}" in result.stdout
    assert "cache: volume=modelsight-cache mount=/cache prefetch=org/model" in result.stdout
    assert "secrets: HF_TOKEN (set)" in result.stdout
    assert "report: 1 commands" in result.stdout
    assert "hf_secret_value" not in result.stdout
    assert fake_refs == [("https://github.com/usationlabs/dynamo", "feat/deepinfer")]


def test_dry_run_flags_missing_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_refs
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.chdir(tmp_path)  # ms reads ./.env, and a real one could set HF_TOKEN
    config = _deepinfer_config(tmp_path)
    result = runner.invoke(app, ["run", "-f", str(config), "--dry-run"])
    assert result.exit_code == 0
    assert "secrets: HF_TOKEN (missing)" in result.stdout


def test_unresolvable_ref_stops_before_anything_launches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(repo: str, ref: str) -> str:
        raise BuildError(f"{ref!r} is not a branch or tag of {repo}")

    monkeypatch.setattr("modelsight.build.resolve_ref", fail)
    config = _deepinfer_config(tmp_path)
    result = runner.invoke(app, ["run", "-f", str(config)])
    assert result.exit_code == 1
    assert "not a branch or tag" in result.stderr


def test_run_refuses_to_start_without_required_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_refs
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.chdir(tmp_path)  # ms reads ./.env, and a real one could set HF_TOKEN
    launched = []
    monkeypatch.setattr("modelsight.cli.modal.run", lambda *a, **k: launched.append(a))
    config = _deepinfer_config(tmp_path)
    result = runner.invoke(app, ["run", "-f", str(config)])
    assert result.exit_code == 1
    assert "HF_TOKEN" in result.stderr
    assert launched == []


def test_run_passes_the_prepared_image_to_modal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_refs
) -> None:
    monkeypatch.setenv("HF_TOKEN", "x")
    seen = {}

    def fake_run(exp, text, name, prepared, detach=False):
        seen["prepared"] = prepared
        seen["detach"] = detach
        return tmp_path / "results"

    monkeypatch.setattr("modelsight.cli.modal.run", fake_run)
    config = _deepinfer_config(tmp_path)
    result = runner.invoke(app, ["run", "-f", str(config)])
    assert result.exit_code == 0, result.stdout
    assert len(seen["prepared"].fingerprint) == 12
    assert ("DYNAMO_SHA", SHA) in seen["prepared"].build_args


def test_build_dry_run_prints_the_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_refs
) -> None:
    monkeypatch.setenv("HF_TOKEN", "x")
    config = _deepinfer_config(tmp_path)
    result = runner.invoke(app, ["build", "-f", str(config), "--dry-run"])
    assert result.exit_code == 0, result.stdout
    assert "image_fingerprint:" in result.stdout


def test_build_calls_modal_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_refs
) -> None:
    monkeypatch.setenv("HF_TOKEN", "x")
    built = []
    monkeypatch.setattr(
        "modelsight.cli.modal.build",
        lambda exp, prepared, detach=True: built.append((prepared, detach)),
    )
    config = _deepinfer_config(tmp_path)
    result = runner.invoke(app, ["build", "-f", str(config)])
    assert result.exit_code == 0, result.stdout
    assert len(built) == 1


def test_build_rejects_local_target_and_empty_plans(tmp_path: Path) -> None:
    local = tmp_path / "local.yaml"
    local.write_text("target: local\nserver: s\nbench: b\n")
    result = runner.invoke(app, ["build", "-f", str(local)])
    assert result.exit_code == 1
    assert "target: modal" in result.stderr

    plain = tmp_path / "plain.yaml"
    plain.write_text("target: modal\ngpu: B200\nimage: x\nserver: s\nbench: b\n")
    result = runner.invoke(app, ["build", "-f", str(plain)])
    assert result.exit_code == 1
    assert "nothing to build" in result.stderr


def test_run_is_not_detached_unless_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_refs
) -> None:
    monkeypatch.setenv("HF_TOKEN", "x")
    seen = []
    monkeypatch.setattr(
        "modelsight.cli.modal.run",
        lambda exp, text, name, prepared, detach=False: seen.append(detach)
        or tmp_path / "results",
    )
    config = _deepinfer_config(tmp_path)
    assert runner.invoke(app, ["run", "-f", str(config)]).exit_code == 0
    assert runner.invoke(app, ["run", "-f", str(config), "--detach"]).exit_code == 0
    assert seen == [False, True]


def test_build_is_detached_unless_told_not_to(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_refs
) -> None:
    monkeypatch.setenv("HF_TOKEN", "x")
    seen = []
    monkeypatch.setattr(
        "modelsight.cli.modal.build",
        lambda exp, prepared, detach=True: seen.append(detach),
    )
    config = _deepinfer_config(tmp_path)
    assert runner.invoke(app, ["build", "-f", str(config)]).exit_code == 0
    assert runner.invoke(app, ["build", "-f", str(config), "--no-detach"]).exit_code == 0
    assert seen == [True, False]


def test_detach_is_rejected_for_local_runs(tmp_path: Path) -> None:
    config = tmp_path / "local.yaml"
    config.write_text("target: local\nserver: s\nbench: b\n")
    result = runner.invoke(app, ["run", "-f", str(config), "--detach"])
    assert result.exit_code == 1
    assert "target: modal" in result.stderr


def _saved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str = "my-run", kind: str = "run"):
    from modelsight.runs import RunState, save_state

    monkeypatch.chdir(tmp_path)
    state = RunState(
        kind=kind,
        name=name,
        app_id="ap-1",
        function_call_id="fc-1",
        profile=None,
        output=str(tmp_path / "results"),
        volume="modelsight-results",
        started_at="2026-10-21T19:49:51",
    )
    save_state(state)
    return state


def test_attach_without_a_name_lists_the_saved_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _saved(tmp_path, monkeypatch, "run-one")
    _saved(tmp_path, monkeypatch, "build-two", kind="build")
    result = runner.invoke(app, ["attach"])
    assert result.exit_code == 0
    assert "run-one  run  app ap-1" in result.stdout
    assert "build-two  build" in result.stdout


def test_attach_with_nothing_saved_says_so(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["attach"])
    assert result.exit_code == 0
    assert "no detached runs saved here" in result.stdout


def test_attach_reconnects_and_prints_where_the_results_went(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _saved(tmp_path, monkeypatch)
    seen = []
    monkeypatch.setattr(
        "modelsight.cli.modal.attach",
        lambda s, logs=True: seen.append((s, logs)) or tmp_path / "results" / "my-run",
    )
    result = runner.invoke(app, ["attach", "my-run", "--no-logs"])
    assert result.exit_code == 0, result.stdout
    assert seen == [(state, False)]
    assert f"results: {tmp_path / 'results' / 'my-run'}" in result.stdout


def test_attach_to_a_build_prints_no_results_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _saved(tmp_path, monkeypatch, "build-x", kind="build")
    monkeypatch.setattr("modelsight.cli.modal.attach", lambda s, logs=True: None)
    result = runner.invoke(app, ["attach", "build-x"])
    assert result.exit_code == 0
    assert "results:" not in result.stdout


def test_attach_to_an_unknown_run_fails_clearly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["attach", "nope"])
    assert result.exit_code == 1
    assert "no saved run 'nope'" in result.stderr


def test_attach_reports_a_lost_connection_as_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from modelsight.runner import RunError

    _saved(tmp_path, monkeypatch)

    def lost(state, logs=True):
        raise RunError("lost the connection to Modal. Reconnect with: ms attach my-run")

    monkeypatch.setattr("modelsight.cli.modal.attach", lost)
    result = runner.invoke(app, ["attach", "my-run"])
    assert result.exit_code == 1
    assert "ms attach my-run" in result.stderr
