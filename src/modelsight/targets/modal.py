"""Run the same commands inside one Modal container, then copy the results back.

The server and the bench run in the same container, so the bench command talks
to localhost. AIPerf is installed with uv into its own environment, leaving the
image's engine environment alone. The image comes from `modelsight.build`:
a Dockerfile (or registry image) is the base, and ms adds its own layers on top.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

from modelsight.build import BuildError, Prepared, prepare
from modelsight.config import Experiment
from modelsight.runner import MODAL_TIMEOUT_BUFFER_SECONDS, RunError

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

        for model in models:
            print(f"downloading {model}", flush=True)
            subprocess.run(
                ["uvx", "--from", "huggingface_hub", "hf", "download", model],
                check=True,
            )
        if cache_volume is not None:
            cache_volume.commit()

    try:
        with modal.enable_output(), app.run(detach=detach):
            if detach:
                print(
                    "running detached: it keeps going if this machine loses its "
                    f"connection. Stop it with: modal app stop {app.app_id}",
                    flush=True,
                )
            prepare_remote.remote(models)
    except Exception as exc:
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
            "running detached: the container keeps going if this machine loses its "
            "connection to Modal, and still stops at timeout. Results are stored on "
            f"the volume; retrieve them with: modal volume get {VOLUME_NAME} {run_name} {local_dir}",
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
        from modelsight.runner import execute_run

        experiment = experiment_from_mapping(mapping)
        try:
            execute_run(
                experiment,
                Path(RESULTS_MOUNT) / run_name,
                config_text,
                extra_env=info_text,
            )
        finally:
            for mounted in (volume, cache_volume):
                if mounted is None:
                    continue
                try:
                    mounted.commit()
                except Exception as exc:
                    print(f"warning: could not commit a volume: {exc}", flush=True)

    try:
        with modal.enable_output(), app.run(detach=detach):
            if detach:
                print(f"stop it with: modal app stop {app.app_id}", flush=True)
            remote.remote(exp.to_mapping(), config_text, run_name, info_text)
    except RunError:
        _download(volume, run_name, local_dir)
        raise
    except Exception as exc:
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
