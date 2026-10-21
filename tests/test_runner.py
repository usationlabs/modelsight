import os
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

from modelsight.config import load_experiment
from modelsight.runner import RunError, start_process, stop_process
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
    bench.write_text("from pathlib import Path\nPath('marker.txt').write_text(str(Path.cwd()))\n")
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
