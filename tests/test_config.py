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
    resolve_nodes,
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
    assert (experiment.nodes, experiment.rdma, experiment.bench_rank) == (1, False, None)
    assert experiment.ready_urls == () and not experiment.clustered


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
        ("nodes: 9\n", "between 1 and 8"),
        ("nodes: two\n", "nodes must be an integer"),
        ("rdma: true\n", "rdma needs nodes > 1"),
        ("nodes: 2\nrdma: yes please\n", "rdma must be true or false"),
        ("nodes: 2\nbench_rank: 2\n", "bench_rank must be in 0..1"),
        ("ready_url: http://{node1}:1/\n", "{node1}, but the run has 1 node"),
        ("nodes: 2\nready_urls: [localhost:1]\n", "ready_urls must start"),
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
nodes: 2
rdma: true
bench_rank: 1
ready_url: http://localhost:8200/health
ready_urls: ["http://{node0}:8100/health"]
""",
    )
    assert experiment_from_mapping(experiment.to_mapping()) == experiment
    assert experiment.clustered and experiment.bench_rank_resolved == 1
    assert experiment.ready_urls == ("http://{node0}:8100/health",)


def test_plain_experiment_round_trips(tmp_path: Path) -> None:
    experiment, _ = _load(tmp_path, "target: local\nserver: s\nbench: b\n")
    assert experiment_from_mapping(experiment.to_mapping()) == experiment
    # a single-node mapping carries none of the cluster fields
    assert not {"nodes", "rdma", "bench_rank", "ready_urls"} & set(experiment.to_mapping())


def test_bench_rank_defaults_to_the_last_node_and_placeholders_resolve(tmp_path: Path) -> None:
    experiment, _ = _load(tmp_path, MODAL_BASE + "nodes: 3\nready_url: http://{node2}:1/\n")
    assert experiment.bench_rank is None and experiment.bench_rank_resolved == 2
    assert resolve_nodes(experiment.ready_url, ["a", "b", "10.0.0.3"]) == "http://10.0.0.3:1/"


H6_YAMLS = sorted((ROOT / "examples" / "kv-transfer").glob("h6*.yaml"))


@pytest.mark.parametrize("path", H6_YAMLS, ids=lambda p: p.name)
def test_h6_yamls_load_as_two_node_rdma_runs(path: Path) -> None:
    experiment, _ = load_experiment(path)

    assert (experiment.nodes, experiment.rdma, experiment.bench_rank_resolved) == (2, True, 1)
    assert Path(experiment.dockerfile).name == "Dockerfile.h6" and experiment.gpu == "H100"
    assert experiment.ready_urls == ("http://{node0}:8100/health",)
    assert "kv_peer_stage_load" in experiment.bench and "/opt/h6_server.sh" in experiment.server


def test_dockerfile_h6_provides_every_opt_path_the_h6_files_use() -> None:
    import re

    kv = ROOT / "examples" / "kv-transfer"
    dockerfile = (kv / "Dockerfile.h6").read_text()
    used = set()
    for path in [*H6_YAMLS, kv / "h6_server.sh"]:
        used |= set(re.findall(r"/opt/[A-Za-z0-9_/-]+(?:\.[A-Za-z0-9_]+)*", path.read_text()))
    assert "/opt/h6_server.sh" in used
    missing = sorted(p for p in used if p not in dockerfile)
    assert not missing, f"Dockerfile.h6 never mentions {missing}"
    # a path the image gets from a COPY must have its source in the build context
    for source, dest in re.findall(r"^COPY (\S+) (/opt/\S+)", dockerfile, re.M):
        assert (kv / source).exists() or source.endswith(".tar.gz"), source
