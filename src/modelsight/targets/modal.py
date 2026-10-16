"""Run the same commands inside one Modal container, then copy the results back.

The server and the bench run in the same container, so the bench command talks
to localhost. AIPerf is installed with uv into its own environment, leaving the
image's SGLang environment alone.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import modal

from modelsight.config import Experiment
from modelsight.runner import MODAL_TIMEOUT_BUFFER_SECONDS, RunError

VOLUME_NAME = "modelsight-results"
RESULTS_MOUNT = "/results"
HF_SECRET = "huggingface"
# Pinned so image builds stay reproducible. Bump deliberately.
UV_VERSION = "0.12.5"
APP_NAME = "modelsight"


def install_commands() -> list[str]:
    """Shell commands that put a pinned uv and AIPerf on the image PATH."""
    installer = f"https://astral.sh/uv/{UV_VERSION}/install.sh"
    return [
        "DEBIAN_FRONTEND=noninteractive apt-get update && "
        "DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends curl ca-certificates",
        f"curl -LsSf {installer} | env UV_UNMANAGED_INSTALL=/usr/local/bin sh",
        "uv python install 3.12",
        "uv venv /opt/aiperf --python 3.12",
        "uv pip install --python /opt/aiperf/bin/python aiperf",
        "ln -sf /opt/aiperf/bin/aiperf /usr/local/bin/aiperf",
    ]


def build_image(image_ref: str) -> modal.Image:
    image = modal.Image.from_registry(image_ref)
    for command in install_commands():
        image = image.run_commands(command)
    # Mounted at runtime, so changing ms does not rebuild the SGLang image.
    return image.add_local_python_source("modelsight")


def run(exp: Experiment, config_text: str, run_name: str) -> Path:
    if not exp.gpu or not exp.image:
        raise RunError("modal target requires gpu and image")

    local_dir = _local_destination(exp.output, run_name)
    print(
        f"submitting to Modal: gpu={exp.gpu} image={exp.image} volume={VOLUME_NAME}",
        flush=True,
    )
    print(
        "results are stored on the volume even if this laptop disconnects; "
        f"retrieve them with: modal volume get {VOLUME_NAME} {run_name} {local_dir}",
        flush=True,
    )

    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    app = modal.App(APP_NAME)
    image = build_image(exp.image)

    @app.function(
        image=image,
        gpu=exp.gpu,
        timeout=int(exp.timeout_seconds) + MODAL_TIMEOUT_BUFFER_SECONDS,
        volumes={RESULTS_MOUNT: volume},
        secrets=[modal.Secret.from_name(HF_SECRET, required_keys=["HF_TOKEN"])],
        serialized=True,
    )
    def remote(payload: dict, config_text: str, run_name: str) -> None:
        # Imports stay inside the function so the container only needs modelsight
        # and the standard library, not the client's CLI dependencies.
        from pathlib import Path

        from modelsight.config import Experiment
        from modelsight.runner import execute_run

        experiment = Experiment(**payload)
        try:
            execute_run(experiment, Path(RESULTS_MOUNT) / run_name, config_text)
        finally:
            try:
                volume.commit()
            except Exception as exc:
                print(f"warning: could not commit results volume: {exc}", flush=True)

    try:
        with modal.enable_output(), app.run():
            remote.remote(asdict(exp), config_text, run_name)
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
