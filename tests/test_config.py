import re
from pathlib import Path

import pytest

from modelsight.config import (
    Cache,
    ConfigError,
    Source,
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
DOCKERFILE_BASE = "target: modal\ngpu: B200:8\nserver: serve\nbench: bench\n"


def _load(tmp_path: Path, text: str):
    path = tmp_path / "exp.yaml"
    path.write_text(text)
    return load_experiment(path)


def _with_dockerfile(tmp_path: Path, extra: str = "", where: str = "Dockerfile"):
    dockerfile = tmp_path / where
    dockerfile.parent.mkdir(parents=True, exist_ok=True)
    dockerfile.write_text("FROM scratch\n")
    return _load(tmp_path, DOCKERFILE_BASE + f"dockerfile: {where}\n" + extra)


def test_new_sections_are_optional(tmp_path: Path) -> None:
    experiment, _ = _load(tmp_path, MODAL_BASE)
    assert experiment.dockerfile is None
    assert experiment.build_args == ()
    assert experiment.sources == ()
    assert experiment.cache is None
    assert experiment.secrets == ()
    assert experiment.report == ()


def test_dockerfile_is_resolved_relative_to_the_experiment_file(tmp_path: Path) -> None:
    experiment, _ = _with_dockerfile(tmp_path, where="images/Dockerfile")
    assert experiment.dockerfile == str((tmp_path / "images" / "Dockerfile").resolve())
    assert experiment.image is None


def test_missing_dockerfile_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        _load(tmp_path, DOCKERFILE_BASE + "dockerfile: Nope\n")


def test_modal_needs_image_or_dockerfile_but_not_both(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="image or dockerfile is required"):
        _load(tmp_path, "target: modal\ngpu: B200\nserver: s\nbench: b\n")
    (tmp_path / "Dockerfile").write_text("FROM scratch\n")
    with pytest.raises(ConfigError, match="not both"):
        _load(tmp_path, MODAL_BASE + "dockerfile: Dockerfile\n")


def test_sources_and_build_args(tmp_path: Path) -> None:
    experiment, _ = _with_dockerfile(
        tmp_path,
        """add_python: "3.12"
build_args:
  JOBS: 8
  NOTE: hello
sources:
  DYNAMO:
    repo: https://github.com/usationlabs/dynamo
    ref: feat/deepinfer
  ENGINE: {repo: https://github.com/org/engine}
""",
    )
    assert experiment.add_python == "3.12"
    assert experiment.build_args == (("JOBS", "8"), ("NOTE", "hello"))
    assert experiment.sources == (
        Source("DYNAMO", "https://github.com/usationlabs/dynamo", "feat/deepinfer"),
        Source("ENGINE", "https://github.com/org/engine", "main"),  # the default ref
    )


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
        ("sources: [a]\n", "sources must be a mapping"),
        ("sources:\n  lower: {repo: https://x/r}\n", "upper case"),
        ("sources:\n  A: {repo: git@github.com:a/b}\n", "https://"),
        ("sources:\n  A: {ref: main}\n", "repo"),
        ("sources:\n  A: {repo: https://x/r, sha: 1}\n", "sources.A.sha"),
        ("build_args: [a]\n", "build_args must be a mapping"),
        ("build_args: {'bad-name': 1}\n", "bad-name"),
        ("build_args: {A_SHA: x}\nsources:\n  A: {repo: https://x/r}\n", "A_SHA"),
        ("cache:\n  mount: relative\n", "absolute"),
        ("cache:\n  size: 1\n", "cache.size"),
        ("secrets: HF_TOKEN\n", "list of strings"),
        ("secrets: ['not a name']\n", "environment variable"),
        ("report: [3]\n", "non-empty strings"),
    ],
)
def test_invalid_sections_are_rejected(tmp_path: Path, extra: str, message: str) -> None:
    (tmp_path / "Dockerfile").write_text("FROM scratch\n")
    with pytest.raises(ConfigError, match=re.escape(message)):
        _load(tmp_path, DOCKERFILE_BASE + "dockerfile: Dockerfile\n" + extra)


def test_sources_need_a_dockerfile(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="dockerfile is required"):
        _load(tmp_path, MODAL_BASE + "sources:\n  A: {repo: https://x/r}\n")


@pytest.mark.parametrize(
    "extra",
    [
        "dockerfile: Dockerfile\n",
        "build_args: {A: b}\n",
        "add_python: '3.12'\n",
        "cache:\n  prefetch: [a/b]\n",
        "secrets: [HF_TOKEN]\n",
    ],
)
def test_local_target_rejects_modal_only_sections(tmp_path: Path, extra: str) -> None:
    with pytest.raises(ConfigError, match="only supported when target is modal"):
        _load(tmp_path, "target: local\nserver: s\nbench: b\n" + extra)


def test_report_is_allowed_locally(tmp_path: Path) -> None:
    experiment, _ = _load(tmp_path, "target: local\nserver: s\nbench: b\nreport: [uname -a]\n")
    assert experiment.report == ("uname -a",)


def test_to_mapping_round_trips(tmp_path: Path) -> None:
    experiment, _ = _with_dockerfile(
        tmp_path,
        """add_python: "3.12"
secrets: [HF_TOKEN]
report: [uname -a]
cache:
  prefetch: [org/model]
  volume: weights
  mount: /w
build_args: {A: b}
sources:
  DYNAMO: {repo: https://github.com/a/b, ref: dev}
""",
    )
    assert experiment_from_mapping(experiment.to_mapping()) == experiment


def test_plain_experiment_round_trips(tmp_path: Path) -> None:
    experiment, _ = _load(tmp_path, "target: local\nserver: s\nbench: b\n")
    assert experiment_from_mapping(experiment.to_mapping()) == experiment
