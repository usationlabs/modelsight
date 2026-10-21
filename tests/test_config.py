import re
from pathlib import Path

import pytest

from modelsight.config import (
    Cache,
    ConfigError,
    GitSource,
    experiment_from_mapping,
    load_experiment,
    parse_duration,
)

ROOT = Path(__file__).resolve().parents[1]


def test_parse_duration_units() -> None:
    assert parse_duration("90") == 90
    assert parse_duration("90s") == 90
    assert parse_duration("20m") == 1200
    assert parse_duration("2h") == 7200
    assert parse_duration("1h30m") == 5400


def test_parse_duration_rejects_junk() -> None:
    with pytest.raises(ConfigError):
        parse_duration("soon")
    with pytest.raises(ConfigError):
        parse_duration("0s")


def test_example_yaml_loads() -> None:
    experiment, text = load_experiment(ROOT / "examples" / "qwen27b-fp8.yaml")
    assert experiment.target == "modal"
    assert experiment.gpu == "B200"
    assert "sglang.launch_server" in experiment.server
    assert "--ui none" in experiment.bench
    assert "target: modal" in text


def test_local_yaml_fills_defaults(tmp_path: Path) -> None:
    path = tmp_path / "exp.yaml"
    path.write_text("target: local\nserver: sleep 1\nbench: echo ok\n")
    experiment, _ = load_experiment(path)
    assert experiment.ready_url == "http://localhost:30000/health"
    assert experiment.ready_timeout == "20m"
    assert experiment.timeout == "2h"
    assert experiment.output == "results/"
    assert experiment.gpu is None


def test_modal_requires_gpu_and_image(tmp_path: Path) -> None:
    path = tmp_path / "exp.yaml"
    path.write_text("target: modal\nserver: true\nbench: true\n")
    with pytest.raises(ConfigError, match="gpu"):
        load_experiment(path)


def test_unknown_field_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "exp.yaml"
    path.write_text("target: local\nserver: true\nbench: true\nsweep: []\n")
    with pytest.raises(ConfigError, match="unknown field"):
        load_experiment(path)


def test_bad_ready_url_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "exp.yaml"
    path.write_text(
        "target: local\nserver: true\nbench: true\nready_url: localhost:30000/health\n"
    )
    with pytest.raises(ConfigError, match="ready_url"):
        load_experiment(path)


MODAL_BASE = "target: modal\ngpu: B200:8\nimage: nvidia/cuda:13.2.1-devel-ubuntu24.04\nserver: serve\nbench: bench\n"


def _load(tmp_path: Path, text: str):
    path = tmp_path / "exp.yaml"
    path.write_text(text)
    return load_experiment(path)


def test_new_sections_are_optional(tmp_path: Path) -> None:
    experiment, _ = _load(tmp_path, MODAL_BASE)
    assert experiment.build is None
    assert experiment.cache is None
    assert experiment.secrets == ()
    assert experiment.report == ()


def test_build_steps_accept_commands_and_git_sources(tmp_path: Path) -> None:
    experiment, _ = _load(
        tmp_path,
        MODAL_BASE
        + """build:
  env:
    FOO: bar
    COUNT: 3
  gpu: B200
  secrets: [GH_TOKEN]
  steps:
    - apt-get install -y git
    - git:
        repo: https://github.com/usationlabs/dynamo
        path: /opt/dynamo
        ref: feat/deepinfer
    - git:
        repo: https://github.com/deepseek-ai/DeepInfer
        path: /opt/DeepInfer
""",
    )
    build = experiment.build
    assert build is not None
    assert build.steps[0] == "apt-get install -y git"
    assert build.steps[1] == GitSource(
        "https://github.com/usationlabs/dynamo", "/opt/dynamo", "feat/deepinfer"
    )
    assert build.steps[2].ref == "main"  # the default
    assert build.env == (("FOO", "bar"), ("COUNT", "3"))
    assert build.gpu == "B200"
    assert build.secrets == ("GH_TOKEN",)


def test_cache_defaults(tmp_path: Path) -> None:
    experiment, _ = _load(tmp_path, MODAL_BASE + "cache:\n  prefetch: [org/model]\n")
    assert experiment.cache == Cache(("org/model",), "modelsight-cache", "/cache")


def test_secrets_and_report(tmp_path: Path) -> None:
    experiment, _ = _load(
        tmp_path,
        MODAL_BASE + "secrets: [HF_TOKEN]\nreport:\n  - git -C /opt/x rev-parse HEAD\n",
    )
    assert experiment.secrets == ("HF_TOKEN",)
    assert experiment.report == ("git -C /opt/x rev-parse HEAD",)


@pytest.mark.parametrize(
    "extra, message",
    [
        ("build: {}\n", "build.steps"),
        ("build:\n  steps: []\n", "build.steps"),
        ("build:\n  steps: [ls]\n  nope: 1\n", "build.nope"),
        ("build:\n  steps: [3]\n", "build.steps[0]"),
        ("build:\n  steps:\n    - git: {repo: git@github.com:a/b, path: /x}\n", "https://"),
        ("build:\n  steps:\n    - git: {repo: https://github.com/a/b, path: x}\n", "absolute"),
        ("build:\n  steps:\n    - git: {repo: https://github.com/a/b}\n", "path"),
        ("build:\n  steps:\n    - git: {repo: https://github.com/a/b, path: /x, sha: 1}\n", "sha"),
        ("build:\n  steps: [ls]\n  env: {'bad-name': 1}\n", "bad-name"),
        ("build:\n  steps: [ls]\n  secrets: ['not a name']\n", "environment variable"),
        ("cache:\n  mount: relative\n", "absolute"),
        ("cache:\n  size: 1\n", "cache.size"),
        ("secrets: HF_TOKEN\n", "list of strings"),
        ("report: [3]\n", "non-empty strings"),
    ],
)
def test_invalid_new_sections_are_rejected(tmp_path: Path, extra: str, message: str) -> None:
    with pytest.raises(ConfigError, match=re.escape(message)):
        _load(tmp_path, MODAL_BASE + extra)


@pytest.mark.parametrize(
    "extra", ["build:\n  steps: [ls]\n", "cache:\n  prefetch: [a/b]\n", "secrets: [HF_TOKEN]\n"]
)
def test_local_target_rejects_modal_only_sections(tmp_path: Path, extra: str) -> None:
    with pytest.raises(ConfigError, match="only supported when target is modal"):
        _load(tmp_path, "target: local\nserver: s\nbench: b\n" + extra)


def test_report_is_allowed_locally(tmp_path: Path) -> None:
    experiment, _ = _load(tmp_path, "target: local\nserver: s\nbench: b\nreport: [uname -a]\n")
    assert experiment.report == ("uname -a",)


def test_to_mapping_round_trips(tmp_path: Path) -> None:
    experiment, _ = _load(
        tmp_path,
        MODAL_BASE
        + """secrets: [HF_TOKEN]
report: [uname -a]
cache:
  prefetch: [org/model]
  volume: weights
  mount: /w
build:
  env: {A: b}
  gpu: B200
  secrets: [GH_TOKEN]
  steps:
    - echo hi
    - git: {repo: https://github.com/a/b, path: /opt/b, ref: dev}
""",
    )
    assert experiment_from_mapping(experiment.to_mapping()) == experiment


def test_plain_experiment_round_trips(tmp_path: Path) -> None:
    experiment, _ = _load(tmp_path, "target: local\nserver: s\nbench: b\n")
    assert experiment_from_mapping(experiment.to_mapping()) == experiment
