"""Test hooks for the peer stage end-to-end run (examples/kv-transfer/peer_stage_e2e.yaml).

SCRATCH, not committed. Put this directory on PYTHONPATH of every engine of the
run. Python imports `sitecustomize` at startup in the server and in every process
it spawns, and each patch is applied when its target module is first imported,
so vLLM is not imported here. There is no test flag in vLLM itself: everything
below lives in this file, and every hook is inert until the engine's control
file ($PEER_STAGE_CTL, a JSON object the probe rewrites between scenarios) asks
for it. Every wrapped function is checked at patch time: a missing class or
method raises, so the engine dies at startup instead of running unhooked.

Also loads /opt/nixl-error-mode/sitecustomize.py (UCX error handling "none",
which the image's own sitecustomize sets and which this module shadows).

Hooks, with the control key that arms each:

  A, vllm.v1.core.kv_cache_manager.KVCacheManager
    free                 always on: remembers the block hashes of each finished
                         request, keyed by its first 8 prompt token ids
    reset_prefix_cache   "evict": {"key": [8 token ids], "lo": i, "hi": j}, once.
                         Instead of a reset, drops the GPU prefix cache entries
                         of blocks i..j-1 of that request's prompt (blocks nobody
                         holds). The blocks stay in the CPU offload tier. This is
                         how a GPU / CPU / GPU layout is built: LRU only ever
                         evicts a tail first, so the middle cannot be displaced by
                         traffic.
  A, vllm.v1.kv_offload.cpu.gpu_worker.CPUOffloadingWorker
    submit_load / get_finished
                         "fail_loads": true     report every CPU->GPU load job
                                                as failed (the worker-level
                                                failure the stage must survive)
                         "slow_loads_s": 1.2    hold a finished load job's result
                                                back until that long after submit
                                                (a slow copy, so a deadline can
                                                really expire with the job in flight)
  B, ...nixl.peer_lookup_client.PeerLookupClient
    submit               "drop_reply_budget": true  send the lookup without
                                                reply_budget_s, as an engine that
                                                predates the field does
    submit / poll        always on: tagged log lines of each lookup and its reply,
                         which is where B's view of a partial hit's miss reason
                         shows (vLLM logs it only for a full miss)

Tagged lines, to stderr (the engine's log):

  [PSTAGE-HOOK] evicted positions 16-40 n=24 missing=0 held=0
  [PSTAGE-HOOK] failed load job=7
  [PSTAGE-HOOK] held load job=7 for 1200 ms
  [PSTAGE-HOOK] lookup submit key=cmpl-... reply_budget_s=2.0
  [PSTAGE-HOOK] lookup reply key=cmpl-... hit_tokens=1024 miss=none pin=peer-pin-... error=None
"""

from __future__ import annotations

import collections
import importlib.abc
import json
import os
import sys
import threading
import time

TAG = "[PSTAGE-HOOK]"
CTL_PATH = os.environ.get("PEER_STAGE_CTL")
KEY_TOKENS = 8

MANAGER_MODULE = "vllm.v1.core.kv_cache_manager"
WORKER_MODULE = "vllm.v1.kv_offload.cpu.gpu_worker"
CLIENT_MODULE = "vllm.distributed.kv_transfer.kv_connector.v1.nixl.peer_lookup_client"

# The image's UCX error-mode patch, which this module shadows (one `sitecustomize`
# per interpreter).
_NIXL_MODE = "/opt/nixl-error-mode/sitecustomize.py"
if os.path.exists(_NIXL_MODE):
    with open(_NIXL_MODE) as _f:
        exec(compile(_f.read(), _NIXL_MODE, "exec"), {"__name__": "nixl_error_mode"})

_lock = threading.Lock()


def _say(message: str) -> None:
    print(f"{TAG} {message}", file=sys.stderr, flush=True)


def _read_ctl() -> dict:
    if not CTL_PATH:
        return {}
    try:
        with open(CTL_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _write_ctl(ctl: dict) -> None:
    tmp = f"{CTL_PATH}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(ctl, f)
    os.replace(tmp, CTL_PATH)


def _require(module, class_name: str, method: str):
    cls = getattr(module, class_name, None)
    if cls is None:
        raise RuntimeError(f"{TAG} {module.__name__} has no class {class_name}")
    fn = getattr(cls, method, None)
    if not callable(fn):
        raise RuntimeError(f"{TAG} {class_name} has no method {method}")
    return cls, fn


# KV cache manager ------------------------------------------------------------

# prompt key -> block hashes of the request that finished with that prompt
_recent: collections.OrderedDict = collections.OrderedDict()


def _evict(manager, spec: dict) -> None:
    key = tuple(spec["key"])
    lo, hi = int(spec["lo"]), int(spec["hi"])
    hashes = _recent.get(key)
    if hashes is None or hi > len(hashes):
        _say(
            f"evict failed: no hashes for key {key[:3]}... "
            f"(have {None if hashes is None else len(hashes)} blocks, want {hi})"
        )
        return
    pool = manager.block_pool
    evicted = missing = held = 0
    for position in range(lo, hi):
        blocks = pool.get_cached_block(hashes[position], [0])
        if not blocks:
            missing += 1
            continue
        block = blocks[0]
        if block.ref_cnt != 0:
            held += 1
            continue
        pool._maybe_evict_cached_block(block)
        evicted += 1
    _say(f"evicted positions {lo}-{hi} n={evicted} missing={missing} held={held}")


def _patch_manager(module) -> None:
    cls, free = _require(module, "KVCacheManager", "free")
    _, reset = _require(module, "KVCacheManager", "reset_prefix_cache")

    def hooked_free(self, request):
        try:
            key = tuple(request.prompt_token_ids[:KEY_TOKENS])
            with _lock:
                _recent[key] = list(request.block_hashes)
                while len(_recent) > 256:
                    _recent.popitem(last=False)
        except Exception as exc:  # never break serving for a test hook
            _say(f"could not record hashes: {exc!r}")
        return free(self, request)

    def hooked_reset(self, *args, **kwargs):
        ctl = _read_ctl()
        spec = ctl.pop("evict", None)
        if spec:
            _write_ctl(ctl)
            _evict(self, spec)
            return True
        return reset(self, *args, **kwargs)

    cls.free = hooked_free
    cls.reset_prefix_cache = hooked_reset
    _say(f"patched KVCacheManager.free and .reset_prefix_cache (pid {os.getpid()})")


# CPU offload worker ----------------------------------------------------------

_load_submit: dict[int, float] = {}  # load job id -> monotonic submit time
_held: list[tuple[float, object]] = []  # (due time, TransferResult)


def _patch_worker(module) -> None:
    cls, submit_load = _require(module, "CPUOffloadingWorker", "submit_load")
    _, get_finished = _require(module, "CPUOffloadingWorker", "get_finished")

    def hooked_submit_load(self, job_id, *args, **kwargs):
        with _lock:
            _load_submit[job_id] = time.monotonic()
        return submit_load(self, job_id, *args, **kwargs)

    def hooked_get_finished(self):
        results = get_finished(self)
        ctl = _read_ctl()
        fail = bool(ctl.get("fail_loads"))
        delay = float(ctl.get("slow_loads_s") or 0.0)
        now = time.monotonic()
        out = []
        with _lock:
            for result in results:
                submitted = _load_submit.get(result.job_id)
                if submitted is None:
                    out.append(result)  # a store
                    continue
                if fail and result.success:
                    result.success = False
                    _say(f"failed load job={result.job_id}")
                if delay and now - submitted < delay:
                    _held.append((submitted + delay, result))
                    continue
                del _load_submit[result.job_id]
                out.append(result)
            still = []
            for due, result in _held:
                if due <= now or not delay:
                    _say(f"released load job={result.job_id} after {1e3 * (now - (due - delay)):.0f} ms")
                    _load_submit.pop(result.job_id, None)
                    out.append(result)
                else:
                    still.append((due, result))
            _held[:] = still
        return out

    cls.submit_load = hooked_submit_load
    cls.get_finished = hooked_get_finished
    _say(f"patched CPUOffloadingWorker.submit_load and .get_finished (pid {os.getpid()})")


# Peer lookup client (consumer) -------------------------------------------------


def _patch_client(module) -> None:
    cls, submit = _require(module, "PeerLookupClient", "submit")
    _, poll = _require(module, "PeerLookupClient", "poll")

    def hooked_submit(self, key, host, port, request, engine_id=None):
        if _read_ctl().get("drop_reply_budget"):
            request.reply_budget_s = None
        _say(f"lookup submit key={key} reply_budget_s={request.reply_budget_s}")
        return submit(self, key, host, port, request, engine_id=engine_id)

    def hooked_poll(self, key):
        result = poll(self, key)
        if result is not None:
            reply = result.reply
            if reply is None:
                _say(f"lookup reply key={key} hit_tokens=None miss=None pin=None error={result.error}")
            else:
                miss = getattr(reply.miss_reason, "value", reply.miss_reason)
                _say(
                    f"lookup reply key={key} hit_tokens={reply.hit_tokens} "
                    f"miss={miss} pin={reply.remote_request_id} error=None"
                )
        return result

    cls.submit = hooked_submit
    cls.poll = hooked_poll
    _say(f"patched PeerLookupClient.submit and .poll (pid {os.getpid()})")


PATCHES = {
    MANAGER_MODULE: _patch_manager,
    WORKER_MODULE: _patch_worker,
    CLIENT_MODULE: _patch_client,
}


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
