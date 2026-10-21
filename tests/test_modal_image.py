"""The Modal target turns planned layers into image calls; Modal itself is faked."""

from types import SimpleNamespace

import pytest

from modelsight.build import install_commands, prepare
from modelsight.config import Build, Cache, Experiment, GitSource
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
    def from_registry(cls, ref: str) -> "FakeImage":
        return cls([("from_registry", ref)])

    def env(self, values: dict[str, str]) -> "FakeImage":
        return FakeImage([*self.calls, ("env", values)])

    def run_commands(self, command: str, secrets=None, gpu=None) -> "FakeImage":
        names = [list(secret.values) for secret in secrets or []]
        return FakeImage([*self.calls, ("run", command, names, gpu)])

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


def test_image_starts_from_the_base_and_mounts_modelsight_last() -> None:
    exp = _experiment()
    calls = target.build_image(exp, prepare(exp)).calls
    assert calls[0] == ("from_registry", exp.image)
    assert calls[-1] == ("mount", "modelsight")
    commands = [call[1] for call in calls if call[0] == "run"]
    assert commands == install_commands()


def test_build_steps_become_layers_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GH_TOKEN", "ghp_value")
    exp = _experiment(
        build=Build(
            steps=("echo one", 'curl -H "T: $GH_TOKEN" x', GitSource("https://x/r", "/opt/r", "dev")),
            env=(("A", "b"),),
            gpu="B200",
            secrets=("GH_TOKEN",),
        ),
        cache=Cache(),
    )
    calls = target.build_image(exp, prepare(exp, lambda repo, ref: SHA)).calls
    tooling = len(install_commands())
    own = calls[1 + tooling : -1]
    assert own[0] == ("env", {"A": "b"})
    assert own[1] == ("run", "echo one", [], "B200")
    assert own[2] == ("run", 'curl -H "T: $GH_TOKEN" x', [["GH_TOKEN"]], "B200")
    assert own[3][0] == "run" and SHA in own[3][1]
    assert own[4] == ("env", {"HF_HOME": "/cache/huggingface"})


def test_runtime_secrets_come_from_local_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_value")
    [secret] = target._runtime_secrets(_experiment(secrets=("HF_TOKEN",)))
    assert secret.values == {"HF_TOKEN": "hf_value"}
