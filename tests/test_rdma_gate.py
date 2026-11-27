import pytest

from modelsight.probes import rdma_gate as gate

# H5 run 2: reads between two GPUs on one host, with control traffic on sysv.
H5_CUDA_IPC = """\
[1] [h:9:0]   | active message by ucp_am_send* from host memory |
[1] [h:9:0]   | 0..8176 | short | sysv/memory |
[1] [h:9:0]   | remote memory read by ucp_get*(multi) into cuda/GPU0 from cuda/dev[2] |
[1] [h:9:0]   | Range | Description | Config |
[1] [h:9:0]   | 0..inf | zero-copy   | cuda_ipc/cuda |
"""

RC_MLX5 = """\
[1] [h:9:0]   | ucp_context_0 inter-node cfg#1 | remote memory read by ucp_get*(multi) into cuda/GPU0 from cuda/dev[0] |
[1] [h:9:0]   | 0..8184 | short | rc_mlx5/mlx5_0:1 |
[1] [h:9:0]   | 8185..inf | zero-copy | rc_mlx5/mlx5_0:1 + rc_mlx5/mlx5_1:1 |
"""

TCP = RC_MLX5.replace("rc_mlx5/mlx5_0:1 + rc_mlx5/mlx5_1:1", "tcp/eth0").replace(
    "rc_mlx5/mlx5_0:1", "tcp/eth0"
)

IB_WRITE_BW = """\
---------------------------------------------------------------------------------------
                    RDMA_Write BW Test
 Device          : mlx5_0
 Connection type : RC
---------------------------------------------------------------------------------------
 #bytes     #iterations    BW peak[Gb/sec]    BW average[Gb/sec]   MsgRate[Mpps]
 65536      5000             {peak}             {avg}               0.741738
---------------------------------------------------------------------------------------
"""

PERFTEST = """\
+--------------+--------------+------------------------------+---------------------+-----------------------+
|    Stats     |  Iterations  | typical  | average | overall |  average |  overall |  average  |  overall  |
+--------------+--------------+----------+---------+---------+----------+----------+-----------+-----------+
Final:          100000         90.00     90.00     90.00     {bw}    {bw}     11111      11111
"""


def test_cuda_ipc_table_fails_even_with_control_traffic_on_another_transport():
    result = gate.classify_transports(H5_CUDA_IPC)

    assert not result.ok
    assert result.transports == ["cuda_ipc"]  # sysv belongs to the control table
    assert "cuda_ipc" in result.reason


def test_rdma_table_passes_and_lists_every_lane_once():
    result = gate.classify_transports(RC_MLX5)

    assert result.ok
    assert result.transports == ["rc_mlx5"]


@pytest.mark.parametrize("name", ["dc_mlx5", "ud_verbs", "rc_gda"])
def test_other_rdma_transports_pass(name):
    assert gate.classify_transports(RC_MLX5.replace("rc_mlx5", name)).ok


def test_tcp_table_fails():
    result = gate.classify_transports(TCP)

    assert not result.ok
    assert result.reason == "not RDMA: tcp"


def test_one_bad_lane_among_rdma_ones_fails():
    mixed = RC_MLX5 + "[1] [h:9:0]   | 0..inf | zero-copy | posix/memory |\n"

    assert gate.classify_transports(mixed).transports == ["rc_mlx5", "posix"]
    assert not gate.classify_transports(mixed).ok


def test_text_without_a_table_fails():
    assert not gate.classify_transports("INFO vLLM started\n").ok


def test_ib_write_bw_counts_as_rdma_and_reports_gbps():
    text = IB_WRITE_BW.format(peak="389.32", avg="388.87")

    assert gate.classify_transports(text).ok
    assert gate.bandwidth_gbps(text) == pytest.approx(388.87)


def test_ib_write_bw_in_mb_per_second_is_converted():
    text = IB_WRITE_BW.format(peak="1", avg="40000").replace("Gb/sec", "MB/sec")

    assert gate.bandwidth_gbps(text) == pytest.approx(40000 * 2**20 * 8 / 1e9)


def test_ucx_perftest_bandwidth_is_converted_from_mib_per_s():
    text = PERFTEST.format(bw="40000.00")

    assert gate.bandwidth_gbps(text) == pytest.approx(335.5, abs=0.1)
    assert gate.bandwidth_gbps("nothing here") is None


def test_gate_applies_the_floor_per_log_and_ignores_logs_without_transports():
    fast = IB_WRITE_BW.format(peak="390", avg="388")
    slow = IB_WRITE_BW.format(peak="30", avg="28")
    texts = {"bench.log": "nothing to judge", "ib.log": fast, "ucx.log": RC_MLX5}

    assert gate.gate(texts).ok
    assert not gate.gate({**texts, "slow.log": slow}).ok
    assert gate.gate({"slow.log": slow}, min_gbps=20).ok


def test_gate_requires_bandwidth_only_when_asked():
    assert gate.gate({"b.log": RC_MLX5}).ok
    result = gate.gate({"b.log": RC_MLX5}, require_bandwidth=True)

    assert not result.ok
    assert "no bandwidth" in result.reason


def test_cli_prints_one_line_and_exits_by_verdict(tmp_path, capsys):
    good = tmp_path / "good.log"
    good.write_text(RC_MLX5 + IB_WRITE_BW.format(peak="390", avg="388"))
    bad = tmp_path / "bad.log"
    bad.write_text(H5_CUDA_IPC)

    assert gate.main([str(good), "--require-bandwidth"]) == 0
    assert capsys.readouterr().out.startswith("PASS: ")
    assert gate.main([str(good), str(bad)]) == 1
    out = capsys.readouterr().out
    assert out.startswith("FAIL: ") and "bad.log" in out and out.count("\n") == 1
