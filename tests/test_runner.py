import os
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

from modelsight.config import load_experiment
from modelsight.control import ControlServer, safe_log_path
from modelsight.runner import RunError, execute_cluster_run, start_process, stop_process
from modelsight.targets.local import run as run_local


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _assert_dead(pid: int) -> None:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    raise AssertionError(f"process {pid} is still running")


def _server_script(path: Path, pid_file: Path, port: int) -> None:
    path.write_text(
        "import os\n"
        "from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer\n"
        f"open({str(pid_file)!r}, 'w').write(str(os.getpid()))\n"
        "class Handler(BaseHTTPRequestHandler):\n"
        "    def do_GET(self):\n"
        "        self.send_response(200)\n"
        "        self.end_headers()\n"
        "    def log_message(self, format, *args):\n"
        "        return\n"
        f"ThreadingHTTPServer(('127.0.0.1', {port}), Handler).serve_forever()\n"
    )


def _experiment(tmp_path: Path, text: str):
    path = tmp_path / "exp.yaml"
    path.write_text(text)
    return load_experiment(path)


def test_local_run_saves_outputs_and_stops_the_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    port = _free_port()
    pid_file = tmp_path / "server.pid"
    script = tmp_path / "server.py"
    bench = tmp_path / "bench.py"
    _server_script(script, pid_file, port)
    bench.write_text(
        "import os\nfrom pathlib import Path\nPath('marker.txt').write_text(str(Path.cwd()))\n"
        "Path('env.txt.bench').write_text(os.environ['MS_RANK'] + ' ' + os.environ['MS_NODES']"
        " + ' ' + os.environ['MS_NODE_IPS'] + ' ' + os.environ['MS_RUN_DIR'])\n"
    )
    experiment, text = _experiment(
        tmp_path,
        f"""
target: local
output: {tmp_path / "results"}
timeout: 30s
ready_timeout: 15s
ready_url: http://127.0.0.1:{port}/
server: "{sys.executable} {script}"
bench: "{sys.executable} {bench}"
""",
    )

    run_dir = run_local(experiment, text, "exp-test")

    assert (run_dir / "exp.yaml").read_text() == text
    assert (run_dir / "commands.txt").read_text().startswith("server: ")
    env = (run_dir / "env.txt").read_text()
    assert "ms_version:" in env
    assert "nvidia-smi:" in env
    assert (run_dir / "marker.txt").read_text() == str(run_dir)
    # a single node is rank 0 of 1 at 127.0.0.1
    assert (run_dir / "env.txt.bench").read_text() == f"0 1 127.0.0.1 {run_dir}"
    assert (run_dir / "server.log").exists()
    assert (run_dir / "bench.log").exists()
    _assert_dead(int(pid_file.read_text()))


def test_timeout_stops_the_server_and_its_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    parent_pid = tmp_path / "parent.pid"
    child_pid = tmp_path / "child.pid"
    child = tmp_path / "child.py"
    server = tmp_path / "server.py"
    child.write_text(
        "import os, time\n"
        f"open({str(child_pid)!r}, 'w').write(str(os.getpid()))\n"
        "time.sleep(120)\n"
    )
    server.write_text(
        "import os, subprocess, sys, time\n"
        f"open({str(parent_pid)!r}, 'w').write(str(os.getpid()))\n"
        f"subprocess.Popen([sys.executable, {str(child)!r}])\n"
        "time.sleep(120)\n"
    )
    experiment, text = _experiment(
        tmp_path,
        f"""
target: local
output: results
timeout: 3s
ready_timeout: 30s
ready_url: http://127.0.0.1:9/
server: "{sys.executable} {server}"
bench: "{sys.executable} -c 'import time; time.sleep(30)'"
""",
    )

    with pytest.raises(RunError, match="exceeded timeout"):
        run_local(experiment, text, "exp-timeout")

    for pid_path in (parent_pid, child_pid):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and not pid_path.exists():
            time.sleep(0.05)
        assert pid_path.exists(), f"{pid_path.name} was never written"
        _assert_dead(int(pid_path.read_text()))


def test_bench_failure_still_stops_the_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    port = _free_port()
    pid_file = tmp_path / "server.pid"
    script = tmp_path / "server.py"
    _server_script(script, pid_file, port)
    experiment, text = _experiment(
        tmp_path,
        f"""
target: local
output: results
timeout: 20s
ready_timeout: 10s
ready_url: http://127.0.0.1:{port}/
server: "{sys.executable} {script}"
bench: "{sys.executable} -c 'import sys; sys.exit(3)'"
""",
    )

    with pytest.raises(RunError, match="status 3"):
        run_local(experiment, text, "exp-fail")

    saved = list((tmp_path / "results").rglob("exp.yaml"))
    assert saved, "failed runs should still keep the yaml"
    _assert_dead(int(pid_file.read_text()))


def test_stop_process_is_harmless_when_already_gone(tmp_path: Path) -> None:
    proc, log = start_process(
        f'"{sys.executable}" -c "import sys; sys.exit(0)"',
        tmp_path / "p.log",
        tmp_path,
    )
    proc.wait(timeout=5)
    stop_process(proc)
    log.close()
    assert proc.returncode == 0


def test_report_section_runs_commands_and_keeps_going_after_failures() -> None:
    from modelsight.runner import report_section

    text = report_section(("echo hello", "echo oops >&2; exit 3", "true"))
    assert text.startswith("report:\n$ echo hello\nhello\n")
    assert "$ echo oops >&2; exit 3\noops\n(exit status 3)\n" in text
    assert text.endswith("$ true\n(no output)\n")


def test_report_section_is_empty_without_commands() -> None:
    from modelsight.runner import report_section

    assert report_section(()) == ""


def test_report_section_times_out() -> None:
    from modelsight.runner import report_section

    assert "(timed out after 0.2s)" in report_section(("sleep 5",), timeout=0.2)


def test_env_txt_includes_extra_text_and_report(tmp_path: Path) -> None:
    from modelsight.runner import execute_run

    experiment, text = _experiment(
        tmp_path,
        "target: local\nserver: sleep 30\nbench: 'true'\nready_url: http://127.0.0.1:1/\n"
        "ready_timeout: 1s\nreport: [echo revision-1]\n",
    )
    run_dir = tmp_path / "run"
    with pytest.raises(RunError):  # nothing listens, but env.txt is written first
        execute_run(experiment, run_dir, text, extra_env="image_fingerprint: abc\n")
    env_txt = (run_dir / "env.txt").read_text()
    assert "image_fingerprint: abc\n" in env_txt
    assert "$ echo revision-1\nrevision-1\n" in env_txt


class _Volume:
    def __init__(self, action) -> None:
        self.action = action
        self.commits = 0

    def commit(self) -> None:
        self.commits += 1
        self.action()


def test_a_quick_commit_is_reported_with_its_time(capsys: pytest.CaptureFixture[str]) -> None:
    from modelsight.runner import commit_with_timeout

    volume = _Volume(lambda: None)
    assert commit_with_timeout("results", volume, timeout=5) is True
    assert volume.commits == 1
    assert "committed the results volume in" in capsys.readouterr().out


def test_a_failing_commit_is_a_warning_not_a_crash(capsys: pytest.CaptureFixture[str]) -> None:
    from modelsight.runner import commit_with_timeout

    def boom() -> None:
        raise RuntimeError("volume gone")

    assert commit_with_timeout("cache", _Volume(boom), timeout=5) is False
    assert "could not commit the cache volume: volume gone" in capsys.readouterr().out


def test_a_slow_commit_is_abandoned_so_it_cannot_hold_the_gpus(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from modelsight.runner import commit_with_timeout

    release = threading.Event()
    started = time.monotonic()
    done = commit_with_timeout("cache", _Volume(lambda: release.wait(30)), timeout=0.3)
    elapsed = time.monotonic() - started
    release.set()  # let the stuck worker thread finish
    assert done is False
    assert elapsed < 5, "must return near the timeout, not wait for the commit"
    assert "still running after 0.3s; not waiting any longer" in capsys.readouterr().out


def test_local_target_refuses_a_multi_node_run(tmp_path: Path) -> None:
    experiment, text = _experiment(
        tmp_path, "target: local\nnodes: 2\nserver: s\nbench: b\n"
    )
    with pytest.raises(RunError, match="multi-node runs need the modal target"):
        run_local(experiment, text, "exp-cluster")


# Cluster runs: two ranks on this host, each in a thread, with their own ports.

_CLUSTER_SERVER = """
import os, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
rank = int(os.environ["MS_RANK"])
port = [{p0}, {p1}][rank]
open(os.path.join(os.environ["MS_RANK_DIR"], "pid"), "w").write(str(os.getpid()))
print(f"rank {{rank}} of {{os.environ['MS_NODES']}} at {{os.environ['MS_NODE_IP']}}", flush=True)
if rank == {die_rank}:
    time.sleep(1)
    sys.exit(3)
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
    def log_message(self, *args):
        return
ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
"""

# The bench checks that rank 0 still serves, and reads rank 0's log through
# rank 0's control endpoint, as the H6 probe does.
_CLUSTER_BENCH = """
import os, sys, time, urllib.request
from pathlib import Path
urllib.request.urlopen("http://" + os.environ["MS_NODE0_IP"] + ":{p0}/", timeout=5)
log = urllib.request.urlopen(
    "http://" + os.environ["MS_NODE0_IP"] + ":{c0}/logs/server.log?offset=2", timeout=5
).read().decode()
Path("bench-saw.txt").write_text(f"{{time.monotonic()}}|" + os.environ["MS_RANK"] + "|" + log)
sys.exit({bench_status})
"""


def _cluster(tmp_path: Path, die_rank: int = -1, bench_status: int = 0):
    p0, p1, c0, c1 = (_free_port() for _ in range(4))
    server = tmp_path / "server.py"
    bench = tmp_path / "bench.py"
    server.write_text(_CLUSTER_SERVER.format(p0=p0, p1=p1, die_rank=die_rank))
    bench.write_text(_CLUSTER_BENCH.format(p0=p0, c0=c0, bench_status=bench_status))
    experiment, text = _experiment(
        tmp_path,
        f"""
target: local
nodes: 2
timeout: 40s
ready_timeout: 20s
ready_url: http://127.0.0.1:{p1}/
ready_urls: ["http://{{node0}}:{p0}/"]
server: "{sys.executable} {server}"
bench: "{sys.executable} {bench}"
""",
    )
    run_dir = tmp_path / "run"
    errors: dict[int, BaseException | None] = {}
    finished: dict[int, float] = {}

    def rank(r: int) -> None:
        try:
            execute_cluster_run(experiment, run_dir, text, r, ["127.0.0.1"] * 2,
                                control_ports=[c0, c1], linger_seconds=5)  # fmt: skip
            errors[r] = None
        except BaseException as exc:  # noqa: BLE001 - checked by the test
            errors[r] = exc
        finished[r] = time.monotonic()

    threads = [threading.Thread(target=rank, args=(r,)) for r in (0, 1)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    return run_dir, errors, finished


def test_cluster_rank_0_serves_until_the_bench_rank_is_done(tmp_path: Path) -> None:
    run_dir, errors, finished = _cluster(tmp_path)

    assert errors == {0: None, 1: None}
    # the bench ran on rank 1, in the run dir, while rank 0 still served
    bench_at, rank, log = (run_dir / "bench-saw.txt").read_text().split("|", 2)
    assert rank == "1"
    assert log == "nk 0 of 2 at 127.0.0.1\n"  # rank 0's log from byte 2
    assert finished[0] > float(bench_at)  # rank 0 stopped only after the bench
    for r in (0, 1):
        assert (run_dir / f"rank-{r}" / "server.log").read_text().startswith(f"rank {r} of 2")
        assert (run_dir / f"rank-{r}" / "env.txt").read_text().startswith(f"rank: {r} of 2")
        _assert_dead(int((run_dir / f"rank-{r}" / "pid").read_text()))
    assert (run_dir / "bench.log").exists() and (run_dir / "exp.yaml").exists()


def test_a_server_that_dies_on_rank_0_fails_both_ranks(tmp_path: Path) -> None:
    _, errors, _ = _cluster(tmp_path, die_rank=0)

    assert isinstance(errors[0], RunError) and "server exited early" in str(errors[0])
    assert isinstance(errors[1], RunError) and "rank 0 failed" in str(errors[1])


def test_a_failed_bench_stops_rank_0(tmp_path: Path) -> None:
    run_dir, errors, _ = _cluster(tmp_path, bench_status=3)

    assert isinstance(errors[1], RunError) and "status 3" in str(errors[1])
    assert isinstance(errors[0], RunError) and "bench rank failed" in str(errors[0])
    _assert_dead(int((run_dir / "rank-0" / "pid").read_text()))


def test_a_rank_that_never_hears_from_the_bench_rank_gives_up(tmp_path: Path) -> None:
    experiment, text = _experiment(
        tmp_path, "target: local\nnodes: 2\nserver: sleep 30\nbench: 'true'\n"
    )
    with pytest.raises(RunError, match="lost the bench rank"):
        execute_cluster_run(experiment, tmp_path / "run", text, 0, ["127.0.0.1"] * 2,
                            control_ports=[_free_port(), _free_port()],
                            peer_lost_seconds=1, linger_seconds=1)  # fmt: skip


def test_control_endpoint_serves_state_and_logs_from_an_offset_only_inside_its_dir(
    tmp_path: Path,
) -> None:
    import json
    import urllib.error
    import urllib.request

    rank_dir = tmp_path / "rank-0"
    rank_dir.mkdir()
    (rank_dir / "server-A.log").write_text("0123456789")
    (tmp_path / "secret.txt").write_text("no")
    control = ControlServer(0, rank_dir, port=0, host="127.0.0.1").start()
    base = f"http://127.0.0.1:{control.port}"

    def get(path: str) -> tuple[int, bytes]:
        try:
            with urllib.request.urlopen(base + path, timeout=5) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as exc:
            return exc.code, b""

    try:
        control.set("serving")
        assert json.loads(get("/status")[1]) == {"rank": 0, "state": "serving"}
        assert get("/logs/server-A.log") == (200, b"0123456789")
        assert get("/logs/server-A.log?offset=7") == (200, b"789")
        assert get("/logs/server-A.log?offset=99") == (200, b"")
        assert get("/logs/missing.log")[0] == 404
        for bad in ("../secret.txt", "..%2Fsecret.txt", "%2E%2E", "a/b"):
            assert get(f"/logs/{bad}")[0] in (400, 404), bad
    finally:
        control.stop()
    assert safe_log_path(rank_dir, "..") is None
    assert safe_log_path(rank_dir, "../secret.txt") is None
    assert safe_log_path(rank_dir, "server-A.log") == (rank_dir / "server-A.log").resolve()
