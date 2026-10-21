"""revenue.py prices AIPerf results; it is a script, so load it by path."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "deepinfer" / "revenue.py"
spec = importlib.util.spec_from_file_location("revenue", SCRIPT)
revenue = importlib.util.module_from_spec(spec)
sys.modules["revenue"] = revenue  # dataclasses looks the module up by name
spec.loader.exec_module(revenue)


def _result(
    path: Path,
    *,
    duration=100.0,
    isl=1_000_000,
    osl=100_000,
    usage_in=None,
    usage_out=None,
    concurrency=64,
    complete=True,
) -> Path:
    """A minimal profile_export_aiperf.json with the fields revenue.py reads."""
    path.mkdir(parents=True, exist_ok=True)
    data = {
        "benchmark_duration": {"unit": "sec", "avg": duration},
        "request_count": {"unit": "requests", "avg": 50.0},
        "request_error_rate": {"unit": "%", "avg": 0.0},
        "total_isl": {"unit": "tokens", "avg": isl},
        "total_osl": {"unit": "tokens", "avg": osl},
        "input_config": {
            "phases": [
                {"name": "warmup", "kind": "warmup", "concurrency": concurrency},
                {"name": "profiling", "kind": "profiling", "concurrency": concurrency},
            ]
        },
        "is_complete": complete,
        "was_cancelled": not complete,
    }
    if usage_in is not None:
        data["total_usage_prompt_tokens"] = {"avg": usage_in}
        data["total_usage_completion_tokens"] = {"avg": usage_out}
    (path / "profile_export_aiperf.json").write_text(json.dumps(data))
    return path / "profile_export_aiperf.json"


def test_revenue_is_tokens_times_price_per_hour(tmp_path: Path) -> None:
    run = revenue.load_run(_result(tmp_path / "c64"), "aiperf")
    # 1M input at $0.15/M + 100k output at $0.60/M = $0.21 in 100s = $7.56/hour.
    assert run.revenue_per_hour(0.0) == pytest.approx(7.56)
    assert run.revenue_per_hour(0.0) / 8 == pytest.approx(0.945)


def test_cache_hit_rate_reprices_input_tokens(tmp_path: Path) -> None:
    run = revenue.load_run(_result(tmp_path / "c64"), "aiperf")
    # 90% of input at $0.003/M, 10% at $0.15/M, output unchanged.
    expected = (1.0 * (0.1 * 0.15 + 0.9 * 0.003) + 0.1 * 0.60) / (100 / 3600)
    assert run.revenue_per_hour(0.9) == pytest.approx(expected)
    assert run.revenue_per_hour(1.0) < run.revenue_per_hour(0.5) < run.revenue_per_hour(0.0)


def test_throughput_is_per_second_and_concurrency_comes_from_the_profiling_phase(
    tmp_path: Path,
) -> None:
    run = revenue.load_run(_result(tmp_path / "c", concurrency=128), "aiperf")
    assert run.input_tps == pytest.approx(10_000)
    assert run.output_tps == pytest.approx(1_000)
    assert run.concurrency == 128


def test_usage_tokens_are_used_only_when_asked_for(tmp_path: Path) -> None:
    path = _result(tmp_path / "c", usage_in=2_000_000, usage_out=200_000)
    assert revenue.load_run(path, "aiperf").input_tokens == 1_000_000
    assert revenue.load_run(path, "usage").input_tokens == 2_000_000


def test_usage_mode_without_server_usage_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no token usage"):
        revenue.load_run(_result(tmp_path / "c"), "usage")


def test_a_directory_of_runs_finds_one_result_per_run_and_skips_warmup_phases(
    tmp_path: Path,
) -> None:
    for name in ("c16", "c32"):
        _result(tmp_path / name)
        _result(tmp_path / name / "phases" / "profiling")
        _result(tmp_path / name / "phases" / "warmup")
    found = revenue.find_results([str(tmp_path)])
    assert sorted(p.parent.name for p in found) == ["c16", "c32"]


def test_a_run_with_only_phase_exports_uses_its_profiling_phase(tmp_path: Path) -> None:
    _result(tmp_path / "c8" / "phases" / "profiling")
    _result(tmp_path / "c8" / "phases" / "warmup")
    found = revenue.find_results([str(tmp_path)])
    assert [p.parent.name for p in found] == ["profiling"]


def test_main_prints_a_row_per_run_and_writes_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _result(tmp_path / "c16", concurrency=16)
    _result(tmp_path / "c32", concurrency=32, osl=200_000)
    out = tmp_path / "summary.json"
    code = revenue.main(["--gpus", "8", str(tmp_path), "--json", str(out)])
    text = capsys.readouterr().out
    assert code == 0
    assert "$38" in text and "off-peak" in text
    assert text.index("  16 ") < text.index("  32 ")
    summary = json.loads(out.read_text())
    assert summary["gpus"] == 8 and len(summary["runs"]) == 2
    assert summary["runs"][0]["usd_per_gpu_hour"]["0.0"] == pytest.approx(0.945)


def test_main_reports_when_nothing_is_found(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert revenue.main(["--gpus", "8", str(tmp_path)]) == 1
    assert "no profile_export_aiperf.json" in capsys.readouterr().err


def test_incomplete_runs_are_flagged(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _result(tmp_path / "c", complete=False)
    assert revenue.main(["--gpus", "8", str(tmp_path)]) == 0
    assert "incomplete or cancelled" in capsys.readouterr().out


@pytest.mark.parametrize("args", [["--gpus", "0"], ["--gpus", "8", "--hit-rates", "1.5"], ["--gpus", "8", "--hit-rates", "x"]])
def test_bad_arguments_exit_with_usage_errors(tmp_path: Path, args: list[str]) -> None:
    with pytest.raises(SystemExit):
        revenue.main([str(tmp_path), *args])
