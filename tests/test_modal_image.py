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

    @classmethod
    def from_name(cls, name: str, required_keys=None) -> "FakeSecret":
        return cls({})


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


class FakeVolume:
    @classmethod
    def from_name(cls, name: str, create_if_missing: bool = False) -> "FakeVolume":
        return cls()

    def commit(self) -> None:
        pass

    def reload(self) -> None:
        pass

    def listdir(self, *args, **kwargs):
        return []


class FakeFunction:
    def __init__(self, fn) -> None:
        self.fn = fn

    def remote(self, *args, **kwargs) -> None:
        return None


class FakeApp:
    """Records how the app was run; function() hands back something callable."""

    runs: list[dict] = []
    app_id = "ap-fake"

    def __init__(self, name: str) -> None:
        pass

    def function(self, **kwargs):
        return FakeFunction

    def run(self, **kwargs):
        FakeApp.runs.append(kwargs)
        return _NullContext()


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None


@pytest.fixture
def fake_app(monkeypatch: pytest.MonkeyPatch):
    FakeApp.runs = []
    monkeypatch.setattr(
        target,
        "modal",
        SimpleNamespace(
            Image=FakeImage,
            Secret=FakeSecret,
            Volume=FakeVolume,
            App=FakeApp,
            enable_output=lambda: _NullContext(),
        ),
    )
    return FakeApp


def test_build_is_detached_by_default_so_a_network_blip_cannot_cancel_a_download(
    fake_app, capsys: pytest.CaptureFixture[str]
) -> None:
    exp = _experiment(cache=Cache(prefetch=("org/model",)))
    target.build(exp, prepare(exp))
    assert fake_app.runs == [{"detach": True}]
    assert "modal app stop ap-fake" in capsys.readouterr().out


def test_build_can_be_attached(fake_app) -> None:
    exp = _experiment(cache=Cache(prefetch=("org/model",)))
    target.build(exp, prepare(exp), detach=False)
    assert fake_app.runs == [{"detach": False}]


def test_run_is_attached_by_default_and_says_what_a_disconnect_does(
    fake_app, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    exp = _experiment(output=str(tmp_path))
    monkeypatch.setattr(target, "_download", lambda *a, **k: True)
    target.run(exp, "yaml", "name", prepare(exp))
    assert fake_app.runs == [{"detach": False}]
    out = capsys.readouterr().out
    assert "not detached" in out and "--detach" in out
    assert "even if this laptop disconnects" not in out


def test_run_can_be_detached(
    fake_app, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    exp = _experiment(output=str(tmp_path))
    monkeypatch.setattr(target, "_download", lambda *a, **k: True)
    target.run(exp, "yaml", "name", prepare(exp), detach=True)
    assert fake_app.runs == [{"detach": True}]
    out = capsys.readouterr().out
    assert "running detached" in out and "modal app stop ap-fake" in out
