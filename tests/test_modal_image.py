"""The Modal target turns the plan into image calls; Modal itself is faked."""

from pathlib import Path
from types import SimpleNamespace

import modal as real_modal
import pytest

from modelsight.build import install_commands, prepare
from modelsight.config import Cache, Experiment, Source
from modelsight.runs import RunState, load_state
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


class FakeCall:
    """A spawned function call: get() returns, or raises what the test sets."""

    object_id = "fc-fake"
    outcome: BaseException | None = None
    gets = 0

    def get(self, timeout=None):
        FakeCall.gets += 1
        if FakeCall.outcome is not None:
            raise FakeCall.outcome

    @classmethod
    def from_id(cls, function_call_id: str) -> "FakeCall":
        cls.attached = function_call_id
        return cls()


class FakeFunction:
    def __init__(self, fn) -> None:
        self.fn = fn

    def remote(self, *args, **kwargs) -> None:
        FakeApp.remote_calls += 1

    def spawn(self, *args, **kwargs) -> FakeCall:
        FakeApp.spawn_calls += 1
        return FakeCall()


class FakeApp:
    """Records how the app was run; function() hands back something callable."""

    runs: list[dict] = []
    remote_calls = 0
    spawn_calls = 0
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
def fake_app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Fake Modal for build()/run()/attach(); saved run state goes to a temp dir."""
    monkeypatch.chdir(tmp_path)
    FakeApp.runs = []
    FakeApp.remote_calls = FakeApp.spawn_calls = 0
    FakeCall.outcome = None
    FakeCall.gets = 0
    FakeCall.attached = None
    monkeypatch.setattr(
        target,
        "modal",
        SimpleNamespace(
            Image=FakeImage,
            Secret=FakeSecret,
            Volume=FakeVolume,
            App=FakeApp,
            FunctionCall=FakeCall,
            enable_output=lambda: _NullContext(),
            exception=real_modal.exception,
            config=SimpleNamespace(_profile="work"),
        ),
    )
    monkeypatch.setattr(target, "_download", lambda *a, **k: True)
    return FakeApp


def _prefetch_experiment(**overrides) -> Experiment:
    return _experiment(cache=Cache(prefetch=("org/model",)), **overrides)


def test_build_is_detached_by_default_so_a_network_blip_cannot_cancel_a_download(
    fake_app, capsys: pytest.CaptureFixture[str]
) -> None:
    exp = _prefetch_experiment()
    target.build(exp, prepare(exp))
    assert fake_app.runs == [{"detach": True}]
    assert fake_app.spawn_calls == 1 and fake_app.remote_calls == 0
    out = capsys.readouterr().out
    assert "ms attach build-" in out and "modal app stop ap-fake" in out


def test_build_can_be_attached(fake_app) -> None:
    exp = _prefetch_experiment()
    target.build(exp, prepare(exp), detach=False)
    assert fake_app.runs == [{"detach": False}]
    assert fake_app.remote_calls == 1 and fake_app.spawn_calls == 0


def test_run_is_attached_by_default_and_says_what_a_disconnect_does(
    fake_app, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exp = _experiment(output=str(tmp_path))
    target.run(exp, "yaml", "name", prepare(exp))
    assert fake_app.runs == [{"detach": False}]
    assert fake_app.remote_calls == 1 and fake_app.spawn_calls == 0
    out = capsys.readouterr().out
    assert "not detached" in out and "--detach" in out
    assert "even if this laptop disconnects" not in out


def test_a_detached_run_is_spawned_and_remembered(
    fake_app, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exp = _experiment(output=str(tmp_path / "out"))
    target.run(exp, "yaml", "my-run", prepare(exp), detach=True)
    assert fake_app.runs == [{"detach": True}]
    assert fake_app.spawn_calls == 1 and fake_app.remote_calls == 0
    state = load_state("my-run")
    assert (state.kind, state.app_id, state.function_call_id) == ("run", "ap-fake", "fc-fake")
    assert state.profile == "work"
    assert state.output == str(tmp_path / "out")
    assert state.volume == target.VOLUME_NAME
    assert "ms attach my-run" in capsys.readouterr().out


def test_losing_the_connection_during_a_detached_run_points_at_attach(
    fake_app, tmp_path: Path
) -> None:
    FakeCall.outcome = real_modal.exception.ConnectionError("dns failure")
    exp = _experiment(output=str(tmp_path))
    with pytest.raises(target.LostConnection, match="ms attach my-run"):
        target.run(exp, "yaml", "my-run", prepare(exp), detach=True)
    assert load_state("my-run").function_call_id == "fc-fake"  # still reachable


def test_a_remote_failure_is_not_reported_as_a_lost_connection(
    fake_app, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeCall.outcome = real_modal.exception.RemoteError("container crashed")
    downloads = []
    monkeypatch.setattr(target, "_download", lambda *a, **k: downloads.append(a) or True)
    exp = _experiment(output=str(tmp_path))
    with pytest.raises(target.RunError) as raised:
        target.run(exp, "yaml", "my-run", prepare(exp), detach=True)
    assert not isinstance(raised.value, target.LostConnection)
    assert downloads  # results are fetched so the failure can be inspected


def test_a_detached_build_that_loses_the_connection_points_at_attach(fake_app) -> None:
    FakeCall.outcome = TimeoutError("timed out")
    exp = _prefetch_experiment()
    with pytest.raises(target.LostConnection, match="ms attach build-"):
        target.build(exp, prepare(exp))


def _state(kind: str = "run", profile: str | None = "work", output: str = "out") -> RunState:
    return RunState(
        kind=kind,
        name="my-run",
        app_id="ap-fake",
        function_call_id="fc-saved",
        profile=profile,
        output=output,
        volume="modelsight-results",
        started_at="2026-10-21T19:49:51",
    )


class FakeLogs:
    def __init__(self, command, env=None) -> None:
        self.command, self.env = command, env
        self.terminated = False
        FakeLogs.last = self

    def poll(self):
        return None if not self.terminated else 0

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout=None) -> int:
        return 0

    def kill(self) -> None:
        self.terminated = True


@pytest.fixture
def fake_logs(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(target.subprocess, "Popen", FakeLogs)
    monkeypatch.setattr(target.time, "sleep", lambda s: None)
    return FakeLogs


def test_attach_waits_for_the_saved_call_and_fetches_results(
    fake_app, fake_logs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    downloads = []
    monkeypatch.setattr(target, "_download", lambda vol, name, dest: downloads.append((name, dest)) or True)
    results = target.attach(_state(output=str(tmp_path / "out")))
    assert FakeCall.attached == "fc-saved" and FakeCall.gets == 1
    assert downloads == [("my-run", tmp_path / "out" / "my-run")]
    assert results == tmp_path / "out" / "my-run"


def test_attach_streams_the_apps_logs_in_the_runs_profile_and_stops_them(
    fake_app, fake_logs
) -> None:
    target.attach(_state())
    command = fake_logs.last.command
    assert command[1:] == ["-m", "modal", "app", "logs", "-f", "ap-fake"]
    assert fake_logs.last.env["MODAL_PROFILE"] == "work"
    assert fake_logs.last.terminated


def test_attach_without_logs_does_not_start_a_log_process(fake_app, fake_logs) -> None:
    fake_logs.last = None
    target.attach(_state(), logs=False)
    assert fake_logs.last is None


def test_attaching_to_a_build_returns_nothing(fake_app, fake_logs) -> None:
    assert target.attach(_state(kind="build")) is None


def test_attach_refuses_the_wrong_profile_and_says_how_to_fix_it(fake_app, fake_logs) -> None:
    with pytest.raises(target.RunError, match="MODAL_PROFILE=other ms attach my-run"):
        target.attach(_state(profile="other"))
    assert FakeCall.attached is None  # never touched Modal


def test_attach_reports_a_lost_connection(fake_app, fake_logs) -> None:
    FakeCall.outcome = real_modal.exception.ConnectionError("dns failure")
    with pytest.raises(target.LostConnection, match="ms attach my-run"):
        target.attach(_state())
    assert fake_logs.last.terminated


def test_attach_reports_a_failed_run(fake_app, fake_logs) -> None:
    FakeCall.outcome = real_modal.exception.RemoteError("boom")
    with pytest.raises(target.RunError, match="the run failed"):
        target.attach(_state())


def test_attach_falls_back_to_the_volume_when_modal_no_longer_has_the_result(
    fake_app, fake_logs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeCall.outcome = real_modal.exception.OutputExpiredError("expired")
    downloads = []
    monkeypatch.setattr(target, "_download", lambda vol, name, dest: downloads.append(name) or True)
    target.attach(_state(output=str(tmp_path)))
    assert downloads == ["my-run"]


def test_attach_stops_watching_on_ctrl_c_but_says_the_run_continues(
    fake_app, fake_logs, capsys: pytest.CaptureFixture[str]
) -> None:
    FakeCall.outcome = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        target.attach(_state())
    assert "still going" in capsys.readouterr().out
    assert fake_logs.last.terminated
