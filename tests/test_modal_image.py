"""The Modal target turns the plan into image calls; Modal itself is faked."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from modelsight.build import install_commands, prepare
from modelsight.config import Cache, Experiment, Source
from modelsight.targets import modal as target

SHA = "d" * 40


class FakeSecret:
    def __init__(self, values: dict[str, str]) -> None:
        self.values = values

    @classmethod
    def from_dict(cls, values: dict[str, str]) -> "FakeSecret":
        return cls(values)


class FakeImage:
    def __init__(self, calls: list) -> None:
        self.calls = calls

    @classmethod
    def from_registry(cls, ref: str, add_python=None) -> "FakeImage":
        return cls([("from_registry", ref, add_python)])

    @classmethod
    def from_dockerfile(cls, path, *, context_dir, build_args, add_python=None) -> "FakeImage":
        return cls([("from_dockerfile", Path(path), Path(context_dir), build_args, add_python)])

    def env(self, values: dict[str, str]) -> "FakeImage":
        return FakeImage([*self.calls, ("env", values)])

    def run_commands(self, command: str) -> "FakeImage":
        return FakeImage([*self.calls, ("run", command)])

    def add_local_python_source(self, name: str) -> "FakeImage":
        return FakeImage([*self.calls, ("mount", name)])


@pytest.fixture(autouse=True)
def fake_modal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        target, "modal", SimpleNamespace(Image=FakeImage, Secret=FakeSecret)
    )


def _experiment(**overrides) -> Experiment:
    values = dict(
        target="modal",
        server="serve",
        bench="bench",
        gpu="B200:8",
        image="nvidia/cuda:13.2.1-devel-ubuntu24.04",
        ready_url="http://localhost:1/",
        ready_timeout="1m",
        timeout="1h",
        output="results/",
    )
    values.update(overrides)
    return Experiment(**values)


def test_registry_image_gets_ms_tooling_and_the_modelsight_mount() -> None:
    exp = _experiment()
    calls = target.build_image(exp, prepare(exp)).calls
    assert calls[0] == ("from_registry", exp.image, None)
    assert calls[-1] == ("mount", "modelsight")
    assert [call[1] for call in calls if call[0] == "run"] == install_commands()


def test_dockerfile_is_built_with_its_directory_as_context_and_resolved_args(
    tmp_path: Path,
) -> None:
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM scratch\n")
    exp = _experiment(
        image=None,
        dockerfile=str(dockerfile),
        add_python="3.12",
        build_args=(("JOBS", "8"),),
        sources=(Source("DYNAMO", "https://x/dynamo", "feat/deepinfer"),),
        cache=Cache(),
    )
    calls = target.build_image(exp, prepare(exp, lambda repo, ref: SHA)).calls
    kind, path, context, args, add_python = calls[0]
    assert kind == "from_dockerfile"
    assert path == dockerfile and context == tmp_path
    assert args == {
        "JOBS": "8",
        "DYNAMO_REPO": "https://x/dynamo",
        "DYNAMO_REF": "feat/deepinfer",
        "DYNAMO_SHA": SHA,
    }
    assert add_python == "3.12"
    assert calls[-2] == ("env", {"HF_HOME": "/cache/huggingface"})
    assert calls[-1] == ("mount", "modelsight")


def test_runtime_secrets_come_from_local_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_value")
    [secret] = target._runtime_secrets(_experiment(secrets=("HF_TOKEN",)))
    assert secret.values == {"HF_TOKEN": "hf_value"}
