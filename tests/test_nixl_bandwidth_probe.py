import itertools
from pathlib import Path

from modelsight.config import load_experiment
from modelsight.probes import nixl_bandwidth as bw

CUDA_IPC = ["| remote memory read by ucp_get*(multi) into cuda from cuda |",
            "| 0..inf | zero-copy | cuda_ipc/cuda |"]  # fmt: skip


def result(rows, path=CUDA_IPC, exit_code=0, data_ok=True):
    return {"nixl_read": {"rows": rows, "read_path": path, "data_ok": data_ok,
                          "initiator_exit": exit_code, "target_exit": 0}}  # fmt: skip


def test_chunks_cover_the_same_bytes_at_every_count():
    for count in (1, 36, 431):
        pieces = bw.chunks(1000, 36 * 431, count, 0)
        assert len(pieces) == count
        assert pieces[0][0] == 1000
        assert all(b[0] == a[0] + a[1] for a, b in itertools.pairwise(pieces))
        assert sum(p[1] for p in pieces) == 36 * 431


def test_gbps():
    assert bw.gbps(4_000_000_000, 2.0) == 2.0
    assert bw.gbps(1, 0.0) == 0.0


def test_verdict_passes_on_cuda_ipc_reads():
    assert bw.verdict(result([{"descriptors": 1, "gbps_median": 200.0}])) == "PASS"


def test_verdict_fails_on_another_transport_or_missing_table():
    rows = [{"descriptors": 1, "gbps_median": 1.0}]
    assert bw.verdict(
        result(rows, path=["| 0..inf | zero-copy | tcp/lo |"])
    ).startswith("FAIL")
    assert bw.verdict(result(rows, path=[])).startswith("UNCONFIRMED")


def test_verdict_fails_on_errors_and_bad_data():
    assert bw.verdict(result([{"descriptors": 36, "error": "x"}])).endswith("[36]")
    assert bw.verdict(result([], exit_code=1)).startswith("FAIL")
    rows = [{"descriptors": 1, "gbps_median": 1.0}]
    assert "do not match" in bw.verdict(result(rows, data_ok=False))


def test_gpu_link_example_is_a_valid_experiment():
    path = Path(__file__).parent.parent / "examples" / "kv-transfer" / "gpu-link.yaml"

    experiment, _ = load_experiment(path)

    assert experiment.dockerfile.endswith("Dockerfile.h4")
    assert "nixl_bandwidth" in experiment.bench
