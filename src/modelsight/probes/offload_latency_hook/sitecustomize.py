"""Timestamp every CPU->GPU load job of vLLM's OffloadingConnector.

SCRATCH, not committed. Put this directory on PYTHONPATH of the vLLM server
(examples/kv-transfer/offload_load_latency.yaml does). Python imports
`sitecustomize` at startup in the server and in every process it spawns, and the
patches are applied when the target modules are first imported, so vLLM is not
imported here.

Three points of one load job's life, all time.monotonic() (CLOCK_MONOTONIC, one
clock for every process on the host):

  submit  CPUOffloadingWorker.submit_load is called (vllm/v1/kv_offload/cpu/
          gpu_worker.py), which builds the copy descriptors and enqueues the
          copy on the load stream
  seen    CPUOffloadingWorker.get_finished returns the job: the worker has
          polled the end event of the transfer and found it done. That poll runs
          once per engine step, so this includes up to one step of waiting
  report  OffloadingConnectorScheduler._update_connector_output receives the
          job in the worker metadata's completed_jobs (offloading/scheduler.py):
          the scheduler learns the job finished

`report - submit` is the number the design's M0 rule is about. Worker and
scheduler share a process only with `--distributed-executor-backend uni`, which
the experiment sets; the hook needs it to join the three points.

One JSON line per load job is appended to $OFFLOAD_LATENCY_LOG (default
/tmp/offload_load_jobs.jsonl), and one tagged line goes to stderr:

  [LOADLAT] job=12 bytes=... copy_ms=... seen_ms=... report_ms=...

Every wrapped function is checked at patch time: a missing class or method
raises, so the server dies at startup instead of running unmeasured.
"""

from __future__ import annotations

import importlib.abc
import json
import os
import sys
import threading
import time

LOG_PATH = os.environ.get("OFFLOAD_LATENCY_LOG", "/tmp/offload_load_jobs.jsonl")
TAG = "[LOADLAT]"

WORKER_MODULE = "vllm.v1.kv_offload.cpu.gpu_worker"
SCHEDULER_MODULE = "vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler"

_lock = threading.Lock()
# job id -> {"submit": t, "submit_returned": t, "expedited": bool, ...}
_jobs: dict[int, dict] = {}


def _say(message: str) -> None:
    print(f"{TAG} {message}", file=sys.stderr, flush=True)


def _require(module, class_name: str, method: str):
    cls = getattr(module, class_name, None)
    if cls is None:
        raise RuntimeError(f"{TAG} {module.__name__} has no class {class_name}")
    fn = getattr(cls, method, None)
    if not callable(fn):
        raise RuntimeError(f"{TAG} {class_name} has no method {method}")
    return cls, fn


def _patch_worker(module) -> None:
    cls, submit_load = _require(module, "CPUOffloadingWorker", "submit_load")
    _, get_finished = _require(module, "CPUOffloadingWorker", "get_finished")

    def timed_submit_load(self, job_id, *args, **kwargs):
        started = time.monotonic()
        try:
            return submit_load(self, job_id, *args, **kwargs)
        finally:
            done = time.monotonic()
            expedited = kwargs.get("expedited", args[2] if len(args) > 2 else False)
            with _lock:
                _jobs[job_id] = {
                    "submit": started,
                    "submit_returned": done,
                    "expedited": bool(expedited),
                }

    def timed_get_finished(self):
        results = get_finished(self)
        seen = time.monotonic()
        with _lock:
            for result in results:
                record = _jobs.get(result.job_id)
                if record is not None and "seen" not in record:
                    record["seen"] = seen
                    record["bytes"] = result.transfer_size
                    record["copy_s"] = result.transfer_time
                    record["success"] = result.success
        return results

    cls.submit_load = timed_submit_load
    cls.get_finished = timed_get_finished
    _say(f"patched {cls.__name__}.submit_load and .get_finished (pid {os.getpid()})")


def _patch_scheduler(module) -> None:
    cls, update = _require(
        module, "OffloadingConnectorScheduler", "_update_connector_output"
    )

    def timed_update(self, connector_output):
        received = time.monotonic()
        meta = getattr(connector_output, "kv_connector_worker_meta", None)
        completed = getattr(meta, "completed_jobs", None) or {}
        lines = []
        with _lock:
            for job_id in completed:
                record = _jobs.pop(job_id, None)
                if record is None:
                    continue  # a store job
                record["report"] = received
                lines.append((job_id, record))
        for job_id, record in lines:
            _emit(job_id, record)
        return update(self, connector_output)

    cls._update_connector_output = timed_update
    _say(f"patched {cls.__name__}._update_connector_output (pid {os.getpid()})")


def _emit(job_id: int, record: dict) -> None:
    submit = record["submit"]
    seen = record.get("seen")
    out = {
        "job": job_id,
        "pid": os.getpid(),
        "submit": submit,
        "seen": seen,
        "report": record["report"],
        "expedited": record["expedited"],
        "bytes": record.get("bytes"),
        "copy_s": record.get("copy_s"),
        "success": record.get("success"),
        "submit_call_s": record["submit_returned"] - submit,
        "submit_to_seen_s": None if seen is None else seen - submit,
        "submit_to_report_s": record["report"] - submit,
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(out) + "\n")
    copy_ms = "-" if out["copy_s"] is None else f"{out['copy_s'] * 1e3:.1f}"
    seen_ms = "-" if seen is None else f"{(seen - submit) * 1e3:.1f}"
    _say(
        f"job={job_id} bytes={out['bytes']} copy_ms={copy_ms} "
        f"seen_ms={seen_ms} report_ms={out['submit_to_report_s'] * 1e3:.1f} "
        f"expedited={out['expedited']}"
    )


PATCHES = {WORKER_MODULE: _patch_worker, SCHEDULER_MODULE: _patch_scheduler}


class _Finder(importlib.abc.MetaPathFinder):
    """Runs a patch right after its target module finishes importing."""

    def find_spec(self, fullname, path, target=None):
        if fullname not in PATCHES:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None:
                break
        else:
            return None
        loader = spec.loader
        exec_module = loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            PATCHES[fullname](module)

        loader.exec_module = exec_and_patch
        return spec


def install() -> None:
    for name, patch in PATCHES.items():
        if name in sys.modules:  # imported before this ran
            patch(sys.modules[name])
    sys.meta_path.insert(0, _Finder())


install()
