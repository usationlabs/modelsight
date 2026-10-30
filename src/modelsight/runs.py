"""Remember detached Modal runs so `ms attach` can reconnect to them later.

A detached run keeps going on Modal when this machine loses its connection, but
the client that started it is gone. The app and function-call IDs are all that is
needed to find it again, so they are written to a small file when the run starts.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

STATE_DIR = Path(".ms") / "runs"


class StateError(Exception):
    """A saved run cannot be found or read."""


@dataclass(frozen=True)
class RunState:
    kind: str  # "run" or "build"
    name: str
    app_id: str
    function_call_id: str
    # The Modal profile (workspace) the run started under; None if unknown.
    profile: str | None
    # Absolute directory results are copied into, and the volume they are read from.
    output: str
    volume: str | None
    started_at: str


def _root(root: Path | None) -> Path:
    return root if root is not None else Path.cwd() / STATE_DIR


def save_state(state: RunState, root: Path | None = None) -> Path:
    directory = _root(root)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{state.name}.json"
    path.write_text(json.dumps(asdict(state), indent=2) + "\n")
    return path


def load_state(ref: str, root: Path | None = None) -> RunState:
    """Load by run name, or by the path of a saved file."""
    candidate = Path(ref)
    path = candidate if candidate.is_file() else _root(root) / f"{ref}.json"
    try:
        data = json.loads(path.read_text())
        return RunState(**data)
    except FileNotFoundError as exc:
        raise StateError(
            f"no saved run {ref!r} in {_root(root)}; "
            "run this from the directory where the run was started"
        ) from exc
    except (OSError, ValueError, TypeError) as exc:
        raise StateError(f"could not read {path}: {exc}") from exc


def list_states(root: Path | None = None) -> list[RunState]:
    directory = _root(root)
    if not directory.is_dir():
        return []
    states = []
    for path in sorted(directory.glob("*.json")):
        try:
            states.append(RunState(**json.loads(path.read_text())))
        except (OSError, ValueError, TypeError):
            continue
    return sorted(states, key=lambda state: state.started_at)
