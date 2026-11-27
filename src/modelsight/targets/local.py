"""Run the server and bench commands as processes on this machine."""

from __future__ import annotations

from pathlib import Path

from modelsight.config import Experiment
from modelsight.runner import RunError, allocate_run_dir, execute_run


def run(exp: Experiment, config_text: str, run_name: str) -> Path:
    if exp.clustered:
        raise RunError(f"multi-node runs need the modal target (nodes: {exp.nodes})")
    run_dir = allocate_run_dir(exp.output, run_name)
    print(f"running locally in {run_dir}", flush=True)
    try:
        execute_run(exp, run_dir, config_text)
    except RunError as exc:
        raise RunError(f"{exc}\nresults: {run_dir}") from exc
    return run_dir
