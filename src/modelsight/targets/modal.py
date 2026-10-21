"""Run the same commands inside one Modal container, then copy the results back.

The server and the bench run in the same container, so the bench command talks
to localhost. AIPerf is installed with uv into its own environment, leaving the
image's engine environment alone. The image comes from `modelsight.build`:
a Dockerfile (or registry image) is the base, and ms adds its own layers on top.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import modal

from modelsight.build import BuildError, Prepared, prepare
from modelsight.config import Experiment
from modelsight.runner import MODAL_TIMEOUT_BUFFER_SECONDS, RunError
from modelsight.runs import RunState, save_state

VOLUME_NAME = "modelsight-results"
RESULTS_MOUNT = "/results"
HF_SECRET = "huggingface"
APP_NAME = "modelsight"
# Downloading a large checkpoint takes a while; this only caps a hung download.
PREFETCH_TIMEOUT_SECONDS = 4 * 3600


def _local_secret(names: tuple[str, ...]) -> modal.Secret:
    """A Modal secret holding the values of local environment variables."""
    return modal.Secret.from_dict({name: os.environ[name] for name in names})


def _runtime_secrets(exp: Experiment) -> list[modal.Secret]:
    if exp.secrets:
        return [_local_secret(exp.secrets)]
    # Without a `secrets` list, keep using the named Modal secret.
    return [modal.Secret.from_name(HF_SECRET, required_keys=["HF_TOKEN"])]


def build_image(exp: Experiment, prepared: Prepared) -> modal.Image:
    """The base (registry image or Dockerfile), then the layers ms adds. Modal
    caches each layer."""
    if exp.dockerfile:
        dockerfile = Path(exp.dockerfile)
        image = modal.Image.from_dockerfile(
            dockerfile,
            context_dir=dockerfile.parent,
            build_args=dict(prepared.build_args),
            add_python=exp.add_python,
        )
    else:
        assert exp.image
        image = modal.Image.from_registry(exp.image, add_python=exp.add_python)
    for layer in prepared.layers:
        if layer.kind == "env":
            image = image.env(dict(layer.env))
        else:
            image = image.run_commands(layer.command)
    # Mounted at runtime, so changing ms does not rebuild the engine image.
    return image.add_local_python_source("modelsight")


class LostConnection(RunError):
    """This machine lost its connection to Modal. A detached run keeps going."""


def _lost_connection(exc: BaseException) -> bool:
    """True for a failure to reach Modal, as opposed to the remote code failing."""
    lost = (
        modal.exception.ConnectionError,
        modal.exception.TimeoutError,
        ConnectionError,
        TimeoutError,
        OSError,
    )
    return isinstance(exc, lost)


def active_profile() -> str | None:
    return getattr(modal.config, "_profile", None) or os.environ.get("MODAL_PROFILE") or None


def _absolute_output(exp: Experiment) -> str:
    path = Path(exp.output)
    return str(path if path.is_absolute() else Path.cwd() / path)


def _record(kind: str, name: str, app, call, exp: Experiment, volume: str | None) -> RunState:
    """Save what `ms attach` needs and tell the user how to reach the run."""
    state = RunState(
        kind=kind,
        name=name,
        app_id=app.app_id,
        function_call_id=call.object_id,
        profile=active_profile(),
        output=_absolute_output(exp),
        volume=volume,
        started_at=datetime.now().isoformat(timespec="seconds"),
    )
    path = save_state(state)
    print(
        f"running detached (app {state.app_id}). Reconnect from this directory with: "
        f"ms attach {name}\nstate saved in {path}\nstop it with: modal app stop {state.app_id}",
        flush=True,
    )
    return state


def _reattach_message(state: RunState) -> str:
    return (
        "lost the connection to Modal. The run is detached, so it keeps going there. "
        f"Reconnect with: ms attach {state.name}"
    )


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _prepared(exp: Experiment, prepared: Prepared | None) -> Prepared:
    if prepared is not None:
        return prepared
    try:
        return prepare(exp)
    except BuildError as exc:
        raise RunError(str(exc)) from exc


def build(
    exp: Experiment, prepared: Prepared | None = None, detach: bool = True
) -> None:
    """Build the image now, and download `cache.prefetch` models, on CPU.

    Modal caches the layers, so a later `ms run` with the same build section
    starts without rebuilding. Weights land on the cache volume.

    Detached by default: Modal stops an app whose client has disconnected, and a
    brief network failure on this machine would otherwise cancel a download that
    takes a long time. This function only uses CPUs and has its own timeout.
    """
    prepared = _prepared(exp, prepared)
    models = list(exp.cache.prefetch) if exp.cache else []
    print(f"building image {prepared.fingerprint} on Modal (no GPU)", flush=True)

    cache_volume = (
        modal.Volume.from_name(exp.cache.volume, create_if_missing=True)
        if exp.cache
        else None
    )
    volumes = {exp.cache.mount: cache_volume} if exp.cache and cache_volume else {}
    app = modal.App(APP_NAME)

    @app.function(
        image=build_image(exp, prepared),
        timeout=PREFETCH_TIMEOUT_SECONDS,
        volumes=volumes,
        secrets=[_local_secret(exp.secrets)] if exp.secrets else [],
        serialized=True,
    )
    def prepare_remote(models: list[str]) -> None:
        import subprocess
        import time

        for model in models:
            print(f"downloading {model}", flush=True)
            subprocess.run(
                ["uvx", "--from", "huggingface_hub", "hf", "download", model],
                check=True,
            )
        if cache_volume is not None:
            # Not bounded: this is what makes the download permanent.
            print("committing the cache volume; this can take a while after a large download", flush=True)
            started = time.monotonic()
            cache_volume.commit()
            print(f"committed the cache volume in {time.monotonic() - started:.0f}s", flush=True)

    state = None
    try:
        with modal.enable_output(), app.run(detach=detach):
            if detach:
                call = prepare_remote.spawn(models)
                state = _record(
                    "build", f"build-{prepared.fingerprint}-{_stamp()}", app, call, exp, None
                )
                call.get()
            else:
                prepare_remote.remote(models)
    except LostConnection:
        raise
    except Exception as exc:
        if state is not None and _lost_connection(exc):
            raise LostConnection(_reattach_message(state)) from exc
        raise RunError(f"modal build failed: {exc}") from exc
    print(f"image {prepared.fingerprint} is built and cached", flush=True)


def run(
    exp: Experiment,
    config_text: str,
    run_name: str,
    prepared: Prepared | None = None,
    detach: bool = False,
) -> Path:
    if not exp.gpu or not (exp.image or exp.dockerfile):
        raise RunError("modal target requires gpu and an image or dockerfile")
    prepared = _prepared(exp, prepared)

    local_dir = _local_destination(exp.output, run_name)
    print(
        f"submitting to Modal: gpu={exp.gpu} image={exp.image or exp.dockerfile} "
        f"fingerprint={prepared.fingerprint} volume={VOLUME_NAME}",
        flush=True,
    )
    if detach:
        print(
            "detached: the container keeps going if this machine loses its connection "
            "to Modal, and still stops at timeout. Results are stored on the volume "
            f"(or fetch them by hand: modal volume get {VOLUME_NAME} {run_name} {local_dir}).",
            flush=True,
        )
    else:
        print(
            "not detached: if this machine loses its connection to Modal for more than "
            "a minute or so, Modal stops the app and the run is lost. Use --detach to "
            "let it finish regardless.",
            flush=True,
        )

    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    volumes = {RESULTS_MOUNT: volume}
    cache_volume = None
    if exp.cache:
        cache_volume = modal.Volume.from_name(exp.cache.volume, create_if_missing=True)
        volumes[exp.cache.mount] = cache_volume
    app = modal.App(APP_NAME)
    image = build_image(exp, prepared)
    info_text = prepared.info_text(exp)

    @app.function(
        image=image,
        gpu=exp.gpu,
        timeout=int(exp.timeout_seconds) + MODAL_TIMEOUT_BUFFER_SECONDS,
        volumes=volumes,
        secrets=_runtime_secrets(exp),
        serialized=True,
    )
    def remote(mapping: dict, config_text: str, run_name: str, info_text: str) -> None:
        # Imports stay inside the function so the container only needs modelsight
        # and the standard library, not the client's CLI dependencies.
        from pathlib import Path

        from modelsight.config import experiment_from_mapping
        from modelsight.runner import commit_with_timeout, execute_run

        experiment = experiment_from_mapping(mapping)
        try:
            execute_run(
                experiment,
                Path(RESULTS_MOUNT) / run_name,
                config_text,
                extra_env=info_text,
            )
        finally:
            # The results first, then the cache. Each commit is bounded and timed so
            # a slow one cannot keep the GPUs idle once the run is over.
            for label, mounted in (("results", volume), ("cache", cache_volume)):
                if mounted is not None:
                    commit_with_timeout(label, mounted)

    state = None
    try:
        with modal.enable_output(), app.run(detach=detach):
            if detach:
                call = remote.spawn(exp.to_mapping(), config_text, run_name, info_text)
                state = _record("run", run_name, app, call, exp, VOLUME_NAME)
                call.get()
            else:
                remote.remote(exp.to_mapping(), config_text, run_name, info_text)
    except LostConnection:
        raise
    except RunError:
        _download(volume, run_name, local_dir)
        raise
    except Exception as exc:
        if state is not None and _lost_connection(exc):
            # Offline, so there is nothing to download; the run is still going.
            raise LostConnection(_reattach_message(state)) from exc
        _download(volume, run_name, local_dir)
        if "Secret" in type(exc).__name__ or HF_SECRET in str(exc):
            raise RunError(
                f"Modal secret {HF_SECRET!r} is required. Create it with:\n"
                f"  modal secret create {HF_SECRET} HF_TOKEN=$HF_TOKEN\n{exc}"
            ) from exc
        raise RunError(f"modal run failed: {exc}\nresults: {local_dir}") from exc

    if not _download(volume, run_name, local_dir):
        raise RunError(
            f"the Modal run finished, but no results were found on volume {VOLUME_NAME} at {run_name}"
        )
    return local_dir


def attach(state: RunState, logs: bool = True) -> Path | None:
    """Reconnect to a detached run: stream its logs, wait for it, fetch results.

    Returns the folder results were copied to, or None for a build.
    """
    current = active_profile()
    if state.profile and current != state.profile:
        raise RunError(
            f"this run started under Modal profile {state.profile!r} but {current!r} is "
            f"active. Use: MODAL_PROFILE={state.profile} ms attach {state.name}"
        )
    call = modal.FunctionCall.from_id(state.function_call_id)
    log_process = _follow_logs(state) if logs else None
    finished = False
    try:
        call.get()
        finished = True
    except KeyboardInterrupt:
        print(
            f"\nstopped watching; the run is still going. Reconnect with: ms attach {state.name}",
            flush=True,
        )
        raise
    except (modal.exception.OutputExpiredError, modal.exception.NotFoundError):
        print(
            "Modal no longer has this call's result; looking for results on the volume.",
            flush=True,
        )
        finished = True
    except Exception as exc:
        if _lost_connection(exc):
            raise LostConnection(_reattach_message(state)) from exc
        raise RunError(f"the {state.kind} failed: {exc}") from exc
    finally:
        _stop_logs(log_process, flush=finished)
    print(f"the {state.kind} has finished", flush=True)
    if state.kind != "run":
        return None

    volume = modal.Volume.from_name(state.volume or VOLUME_NAME)
    local_dir = _local_destination(state.output, state.name)
    if not _download(volume, state.name, local_dir):
        raise RunError(
            f"the run finished, but no results were found on volume {state.volume} at {state.name}"
        )
    return local_dir


def _follow_logs(state: RunState) -> subprocess.Popen | None:
    """Stream the app's logs to this terminal with Modal's own CLI."""
    env = {**os.environ}
    if state.profile:
        env["MODAL_PROFILE"] = state.profile
    try:
        return subprocess.Popen(
            [sys.executable, "-m", "modal", "app", "logs", "-f", state.app_id], env=env
        )
    except OSError as exc:
        print(f"could not stream logs: {exc}", flush=True)
        return None


def _stop_logs(process: subprocess.Popen | None, flush: bool) -> None:
    if process is None:
        return
    if flush and process.poll() is None:
        time.sleep(2)  # let the last lines arrive
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


def _local_destination(output: str, run_name: str) -> Path:
    root = Path(output)
    if not root.is_absolute():
        root = Path.cwd() / root
    destination = root / run_name
    suffix = 2
    while destination.exists():
        destination = root / f"{run_name}-{suffix}"
        suffix += 1
    return destination


def _download(volume: modal.Volume, run_name: str, destination: Path) -> bool:
    try:
        volume.reload()
    except Exception:
        # Outside a container, a freshly committed volume is already readable.
        pass
    try:
        entries = list(volume.listdir(run_name, recursive=True))
    except Exception as exc:
        print(f"could not read results from volume {VOLUME_NAME}: {exc}", flush=True)
        return False
    files = [entry for entry in entries if _is_file(entry)]
    if not files:
        print(f"no results found at {VOLUME_NAME}/{run_name}", flush=True)
        return False
    destination.mkdir(parents=True, exist_ok=True)
    for entry in files:
        relative = result_relative_path(entry.path, run_name)
        if relative is None:
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            for chunk in volume.read_file(entry.path):
                handle.write(chunk)
    print(f"downloaded results to {destination}", flush=True)
    return True


def result_relative_path(remote_path: str, run_name: str) -> Path | None:
    """Path of a volume file relative to the run directory."""
    relative = Path(remote_path)
    if relative.is_absolute():
        relative = Path(*relative.parts[1:])
    parts = [part for part in relative.parts if part not in ("", ".")]
    if not parts or ".." in parts:
        return None
    if parts[0] == run_name:
        parts = parts[1:]
    if not parts:
        return None
    return Path(*parts)


def _is_file(entry: object) -> bool:
    kind = getattr(entry, "type", None)
    name = getattr(kind, "name", str(kind))
    return "FILE" in name.upper() and "DIR" not in name.upper()
