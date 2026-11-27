"""Start a server, wait until it responds, run the bench command, always stop the server.

A cluster run (`nodes > 1`) does the same on every rank: each starts `server`,
the bench rank waits for every ready URL, runs `bench` and says it is done, and
the other ranks keep serving until then (see `execute_cluster_run`).
"""

from __future__ import annotations

import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import IO

from modelsight import __version__
from modelsight.config import Experiment, resolve_nodes
from modelsight.control import DEFAULT_PORT, ControlServer, peer_state

# How long a server gets to exit after SIGTERM before it is killed.
_STOP_GRACE_SECONDS = 10
_POLL_SECONDS = 1
# Modal's own timeout is this much longer than the experiment timeout, so a run
# that hits the cap can still stop the server and flush results before Modal
# kills the container.
MODAL_TIMEOUT_BUFFER_SECONDS = 120
# A volume commit that takes longer than this is abandoned, so a slow one cannot
# hold a GPU container idle after the run is over.
COMMIT_TIMEOUT_SECONDS = 180
# A non-bench rank gives up once the bench rank has not answered for this long.
PEER_LOST_SECONDS = 120
# After it finishes, a rank waits at most this long for the others to see it.
LINGER_SECONDS = 30


class RunError(Exception):
    """The run failed. Partial results are already on disk."""


def allocate_run_dir(output: str, run_name: str) -> Path:
    root = Path(output)
    if not root.is_absolute():
        root = Path.cwd() / root
    run_dir = root / run_name
    suffix = 2
    while run_dir.exists():
        run_dir = root / f"{run_name}-{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True)
    return run_dir


def rank_env(
    rank: int, ips: Sequence[str], run_dir: Path, control_port: int = DEFAULT_PORT
) -> dict[str, str]:
    """What `server` and `bench` learn about the cluster, on every rank.

    A single-node run is rank 0 of 1 at 127.0.0.1.
    """
    env = {
        "MS_RANK": str(rank),
        "MS_NODES": str(len(ips)),
        "MS_NODE_IPS": ",".join(ips),
        "MS_NODE_IP": ips[rank],
        "MS_RUN_DIR": str(run_dir),
        "MS_RANK_DIR": str(run_dir / f"rank-{rank}"),
        "MS_CONTROL_PORT": str(control_port),
    }
    for n, ip in enumerate(ips):
        env[f"MS_NODE{n}_IP"] = ip
    return env


def ready_urls(exp: Experiment, ips: Sequence[str]) -> list[str]:
    return [resolve_nodes(url, ips) for url in (exp.ready_url, *exp.ready_urls)]


def execute_run(
    exp: Experiment, run_dir: Path, config_text: str, extra_env: str = ""
) -> None:
    """Run one experiment inside `run_dir`. Used on this machine and inside Modal.

    `extra_env` is extra text for env.txt, such as what the Modal image was
    built from.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "exp.yaml").write_text(config_text)
    (run_dir / "commands.txt").write_text(f"server: {exp.server}\nbench: {exp.bench}\n")
    (run_dir / "env.txt").write_text(
        environment_report(exp.image) + extra_env + report_section(exp.report)
    )

    ips = ["127.0.0.1"]
    env = rank_env(0, ips, run_dir)
    deadline = time.monotonic() + exp.timeout_seconds
    server, server_log = start_process(
        exp.server, run_dir / "server.log", cwd=Path.cwd(), prefix="[server] ", env=env
    )
    bench = None
    bench_log = None
    try:
        urls = ready_urls(exp, ips)
        print(f"waiting for {', '.join(urls)}", flush=True)
        wait_until_ready(
            exp,
            server,
            run_dir / "server.log",
            deadline=min(time.monotonic() + exp.ready_timeout_seconds, deadline),
            overall_deadline=deadline,
            urls=urls,
        )
        print("server is ready, starting bench", flush=True)
        bench, bench_log = start_process(
            exp.bench, run_dir / "bench.log", cwd=run_dir, prefix="[bench] ", env=env
        )
        wait_for_exit(bench, deadline, exp.timeout)
        if bench.returncode != 0:
            raise RunError(f"bench command exited with status {bench.returncode}")
        print("bench finished", flush=True)
    finally:
        if bench is not None and bench.poll() is None:
            stop_process(bench)
        if bench_log is not None:
            bench_log.close()
        print("stopping server", flush=True)
        stop_process(server)
        server_log.close()


def execute_cluster_run(
    exp: Experiment,
    run_dir: Path,
    config_text: str,
    rank: int,
    ips: Sequence[str],
    extra_env: str = "",
    control_ports: Sequence[int] | None = None,
    peer_lost_seconds: float = PEER_LOST_SECONDS,
    linger_seconds: float = LINGER_SECONDS,
) -> None:
    """One rank of a cluster run. Every rank of the cluster calls this.

    Every rank starts `server`, logging to run_dir/rank-<r>/server.log, and serves
    its state and logs on its control port. The bench rank waits for the ready
    URLs, runs `bench` in run_dir, marks itself done and stops its server. The
    other ranks serve until the bench rank is done, fails, or has not answered
    for `peer_lost_seconds`. A server that exits early fails its rank, and the
    other ranks stop when they see that. `control_ports` (one per rank) exist so
    a test can run two ranks on one host.
    """
    nodes = len(ips)
    if nodes != exp.nodes:
        raise RunError(f"the cluster has {nodes} containers, but the run asks for {exp.nodes}")
    ports = list(control_ports) if control_ports else [DEFAULT_PORT] * nodes
    bench_rank = exp.bench_rank_resolved
    is_bench = rank == bench_rank
    rank_dir = run_dir / f"rank-{rank}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    if is_bench:
        (run_dir / "exp.yaml").write_text(config_text)
        (run_dir / "commands.txt").write_text(f"server: {exp.server}\nbench: {exp.bench}\n")
    (rank_dir / "env.txt").write_text(
        f"rank: {rank} of {nodes}, bench rank {bench_rank}, ips {','.join(ips)}\n"
        + environment_report(exp.image) + extra_env + report_section(exp.report)
    )

    control = ControlServer(rank, rank_dir, port=ports[rank]).start()
    env = rank_env(rank, ips, run_dir, ports[rank])
    others = [r for r in range(nodes) if r != rank]
    deadline = time.monotonic() + exp.timeout_seconds
    print(f"rank {rank} of {nodes} at {ips[rank]}"
          f"{' (bench)' if is_bench else ''}", flush=True)  # fmt: skip
    server, server_log = start_process(
        exp.server, rank_dir / "server.log", cwd=Path.cwd(), prefix="[server] ", env=env
    )
    control.set("serving")
    try:
        if is_bench:
            _bench_rank(exp, run_dir, rank_dir, server, env, deadline, control,
                        [(ips[r], ports[r], r) for r in others], ips)  # fmt: skip
        else:
            _serving_rank(exp, server, rank_dir, deadline,
                          (ips[bench_rank], ports[bench_rank]), peer_lost_seconds)  # fmt: skip
        control.set("done")
    except BaseException:
        control.set("failed")
        raise
    finally:
        print("stopping server", flush=True)
        stop_process(server)
        server_log.close()
        if is_bench:
            # Stay reachable until the other ranks have seen done or failed.
            _linger(control, [(ips[r], ports[r]) for r in others], linger_seconds)
        elif control.state == "failed":
            _linger_until(lambda: peer_state(ips[bench_rank], ports[bench_rank], 1)
                          in (None, "done", "failed"), linger_seconds)  # fmt: skip
        control.stop()


def _bench_rank(exp, run_dir, rank_dir, server, env, deadline, control, peers, ips) -> None:
    def peer_failed() -> str | None:
        for ip, port, r in peers:
            if peer_state(ip, port, 2) == "failed":
                return f"rank {r} failed"
        return None

    urls = ready_urls(exp, ips)
    print(f"waiting for {', '.join(urls)}", flush=True)
    wait_until_ready(
        exp,
        server,
        rank_dir / "server.log",
        deadline=min(time.monotonic() + exp.ready_timeout_seconds, deadline),
        overall_deadline=deadline,
        urls=urls,
        abort=peer_failed,
    )
    print("servers are ready, starting bench", flush=True)
    control.set("running_bench")
    bench, bench_log = start_process(
        exp.bench, run_dir / "bench.log", cwd=run_dir, prefix="[bench] ", env=env
    )
    try:
        while bench.poll() is None:
            if time.monotonic() >= deadline:
                raise RunError(f"run exceeded timeout of {exp.timeout}")
            if server.poll() is not None:
                raise RunError(f"server exited during the bench (status {server.returncode})\n"
                               + _tail(rank_dir / "server.log"))  # fmt: skip
            reason = peer_failed()
            if reason:
                raise RunError(f"{reason} during the bench")
            time.sleep(_POLL_SECONDS)
        if bench.returncode != 0:
            raise RunError(f"bench command exited with status {bench.returncode}")
        print("bench finished", flush=True)
    finally:
        if bench.poll() is None:
            stop_process(bench)
        bench_log.close()


def _serving_rank(exp, server, rank_dir, deadline, bench_at, peer_lost_seconds) -> None:
    ip, port = bench_at
    last_seen = time.monotonic()
    while True:
        if server.poll() is not None:
            raise RunError(f"server exited early (status {server.returncode})\n"
                           + _tail(rank_dir / "server.log"))  # fmt: skip
        now = time.monotonic()
        if now >= deadline:
            raise RunError(f"run exceeded timeout of {exp.timeout}")
        state = peer_state(ip, port, 2)
        if state is not None:
            last_seen = now
        if state == "done":
            print("the bench rank is done", flush=True)
            return
        if state == "failed":
            raise RunError("the bench rank failed")
        if state is None and now - last_seen > peer_lost_seconds:
            raise RunError(f"lost the bench rank at {ip}:{port} for {peer_lost_seconds:g}s")
        time.sleep(_POLL_SECONDS)


def _linger(control: ControlServer, peers: list[tuple[str, int]], seconds: float) -> None:
    def all_seen() -> bool:
        return all(peer_state(ip, port, 1) in (None, "done", "failed") for ip, port in peers)

    _linger_until(all_seen, seconds)


def _linger_until(done: Callable[[], bool], seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline and not done():
        time.sleep(0.2)


def commit_with_timeout(label: str, volume, timeout: float = COMMIT_TIMEOUT_SECONDS) -> bool:
    """Commit a Modal volume, but give up waiting after `timeout` seconds.

    Prints how long it took, so a slow commit is visible in the run's log.
    Returns True only if the commit finished cleanly.
    """
    outcome: list[BaseException | None] = []

    def work() -> None:
        try:
            volume.commit()
            outcome.append(None)
        except Exception as exc:  # noqa: BLE001 - reported, never fatal
            outcome.append(exc)

    started = time.monotonic()
    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    thread.join(timeout)
    took = time.monotonic() - started
    if thread.is_alive():
        print(
            f"warning: committing the {label} volume is still running after {timeout:g}s; "
            "not waiting any longer",
            flush=True,
        )
        return False
    if outcome and outcome[0] is not None:
        print(f"warning: could not commit the {label} volume: {outcome[0]}", flush=True)
        return False
    print(f"committed the {label} volume in {took:.1f}s", flush=True)
    return True


def aiperf_status(exp: Experiment) -> str:
    """Where `aiperf` will come from, or why it will not be found."""
    try:
        words = shlex.split(exp.bench)
    except ValueError:
        words = []
    if not words or Path(words[0]).name != "aiperf":
        return "not used by this bench command"
    if exp.target == "modal":
        return "installed in the Modal image"
    found = shutil.which("aiperf")
    if found:
        return f"found at {found}"
    return "not found on PATH (install it with: uv tool install aiperf)"


def environment_report(image: str | None) -> str:
    lines = [
        f"ms_version: {__version__}",
        f"image: {image or 'local (no image)'}",
        f"aiperf: {command_version(['aiperf', '--version'])}",
        "nvidia-smi:",
        nvidia_smi(),
    ]
    return "\n".join(lines) + "\n"


def report_section(commands: tuple[str, ...], timeout: float = 60) -> str:
    """Output of the experiment's `report` commands, for env.txt."""
    if not commands:
        return ""
    lines = ["report:"]
    for command in commands:
        lines.append(f"$ {command}")
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                stdin=subprocess.DEVNULL,
            )
            text = (result.stdout + result.stderr).strip()
            if result.returncode != 0:
                text = f"{text}\n(exit status {result.returncode})".strip()
        except subprocess.TimeoutExpired:
            text = f"(timed out after {timeout:g}s)"
        lines.append(text or "(no output)")
    return "\n".join(lines) + "\n"


def command_version(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except FileNotFoundError:
        return f"{command[0]} not available"
    except subprocess.TimeoutExpired:
        return f"{command[0]} timed out"
    text = (result.stdout or result.stderr).strip()
    return text or f"{command[0]} produced no output"


def nvidia_smi() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except FileNotFoundError:
        return "nvidia-smi not available"
    except subprocess.TimeoutExpired:
        return "nvidia-smi timed out"
    text = result.stdout if result.returncode == 0 else result.stderr
    return text.strip() or "nvidia-smi produced no output"


class _LogTee:
    """Copies a process's output to a log file and to this process's stdout."""

    def __init__(self, stream: IO[str], log_file: IO[str], prefix: str) -> None:
        self._stream = stream
        self._log_file = log_file
        self._prefix = prefix
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        for line in self._stream:
            self._log_file.write(line)
            try:
                sys.stdout.write(self._prefix + line)
                sys.stdout.flush()
            except Exception:  # a closed terminal must not stop the log file filling
                pass

    def close(self) -> None:
        # A grandchild that outlives the process can hold the pipe open, so the
        # wait is bounded.
        self._thread.join(timeout=_STOP_GRACE_SECONDS)
        self._stream.close()
        self._log_file.close()


def start_process(
    command: str,
    log_path: Path,
    cwd: Path,
    prefix: str = "",
    env: dict[str, str] | None = None,
) -> tuple[subprocess.Popen[str], _LogTee]:
    """Start `command` in its own process group; `env` adds to os.environ."""
    log_file = log_path.open("w", buffering=1)
    # A new session makes the command its own process group, so teardown can
    # signal the server and every child it spawned.
    try:
        proc = subprocess.Popen(
            command,
            shell=True,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            text=True,
            errors="replace",
            env={**os.environ, **env} if env else None,
        )
    except Exception:
        log_file.close()
        raise
    assert proc.stdout is not None
    return proc, _LogTee(proc.stdout, log_file, prefix)


def stop_process(proc: subprocess.Popen[str], grace: float = _STOP_GRACE_SECONDS) -> None:
    if proc.poll() is not None:
        return
    _signal_group(proc, signal.SIGTERM)
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        _signal_group(proc, signal.SIGKILL)
        proc.wait(timeout=grace)


def wait_until_ready(
    exp: Experiment,
    server: subprocess.Popen[str],
    server_log: Path,
    deadline: float,
    overall_deadline: float,
    urls: Sequence[str] | None = None,
    abort: Callable[[], str | None] | None = None,
) -> None:
    """Wait until every URL answers 200. `abort` returns a reason to give up."""
    pending = list(urls) if urls else [exp.ready_url]
    last_error = "no response yet"
    while True:
        if server.poll() is not None:
            raise RunError(
                f"server exited before it was ready (status {server.returncode})\n"
                + _tail(server_log)
            )
        reason = abort() if abort else None
        if reason:
            raise RunError(f"{reason} before the servers were ready")
        now = time.monotonic()
        if now >= overall_deadline:
            raise RunError(f"run exceeded timeout of {exp.timeout}\n" + _tail(server_log))
        if now >= deadline:
            raise RunError(
                f"server did not become ready at {pending[0]} within {exp.ready_timeout}"
                f" ({last_error})\n" + _tail(server_log)
            )
        while pending:
            status, last_error = _probe(pending[0])
            if status != 200:
                break
            pending.pop(0)
        if not pending:
            return
        time.sleep(min(_POLL_SECONDS, max(0.0, deadline - time.monotonic())))


def wait_for_exit(proc: subprocess.Popen[str], deadline: float, timeout_label: str) -> None:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise RunError(f"run exceeded timeout of {timeout_label}")
    try:
        proc.wait(timeout=remaining)
    except subprocess.TimeoutExpired as exc:
        raise RunError(f"run exceeded timeout of {timeout_label}") from exc


def _probe(url: str) -> tuple[int | None, str]:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status, ""
    except urllib.error.HTTPError as exc:
        return exc.code, f"HTTP {exc.code}"
    except Exception as exc:  # connection refused and DNS failures are expected while booting
        return None, str(exc)


def _signal_group(proc: subprocess.Popen[str], sig: int) -> None:
    try:
        os.killpg(proc.pid, sig)
    except (ProcessLookupError, PermissionError):
        return


def _tail(path: Path, lines: int = 40) -> str:
    if not path.exists():
        return ""
    content = path.read_text(errors="replace").splitlines()
    if not content:
        return ""
    return "server log (last lines):\n" + "\n".join(content[-lines:])
