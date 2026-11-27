"""Did the bytes travel over RDMA? Reads UCX_PROTO_INFO tables and benchmark output.

H6 only means something if node A's KV reaches node B over the RDMA fabric. UCX
picks the transport itself and falls back to tcp or shared memory without
saying so, so the logs are checked:

  - UCX_PROTO_INFO tables: the transports in the data tables (remote memory
    read/write, tagged messages, rendezvous). Control traffic takes other
    transports and is ignored when a data table is present.
  - ucx_perftest output: the Final line's bandwidth (MiB/s).
  - ib_write_bw output: the BW average column, which is verbs by definition.

classify_transports() judges the transports of one text. The bandwidth floor is
applied by gate() / the command line:

    python -m modelsight.probes.rdma_gate <logfile...> [--min-gbps N] [--require-bandwidth]

Exit 0 and one PASS line, or exit 1 and one FAIL line.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# mlx5 and verbs transports, plus the GPU-direct-async ones newer UCX lists.
RDMA = frozenset({
    "rc_mlx5", "dc_mlx5", "ud_mlx5", "rc_verbs", "ud_verbs",
    "rc_gda", "dc_gda", "rc", "dc", "ud",
})  # fmt: skip
# Not RDMA: they work between processes on one host or over the IP network.
NOT_RDMA = frozenset({"tcp", "sysv", "posix", "cuda_ipc", "cma", "knem"})
IB_WRITE_BW = "ib_write_bw"

# `rc_mlx5/mlx5_0:1`, `cuda_ipc/cuda`, `tcp/eth0`: transport / device.
_TRANSPORT = re.compile(r"\b([a-z][a-z0-9_]*)/[\w:.\[\]-]+")
_DATA_TABLE = re.compile(r"remote memory|tagged message|rendezvous")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_MIB_BITS = 2**20 * 8 / 1e9  # MiB/s -> Gb/s


@dataclass
class GateResult:
    ok: bool
    transports: list[str] = field(default_factory=list)
    reason: str = ""


def _ucx_tables(text: str) -> list[tuple[str, list[str]]]:
    """(header line, rows) of each UCX_PROTO_INFO table in the text."""
    tables: list[tuple[str, list[str]]] = []
    for line in text.splitlines():
        if "|" not in line:
            continue
        cell = line[line.index("|") :]
        if "ucp_" in line:
            tables.append((cell, []))
        elif tables and ".." in cell:
            tables[-1][1].append(cell)
    return tables


def _ucx_transports(text: str) -> list[str]:
    tables = _ucx_tables(text)
    data = [t for t in tables if _DATA_TABLE.search(t[0])] or tables
    found: list[str] = []
    for _, rows in data:
        for row in rows:
            found += [m.group(1) for m in _TRANSPORT.finditer(row)]
    return found


def _ib_write_bw_seen(text: str) -> bool:
    return bool(re.search(r"^\s*#bytes\s+#iterations\s+BW", text, re.M))


def classify_transports(text: str) -> GateResult:
    """RDMA transports only: ok is False on tcp, sysv, posix, cuda_ipc, cma, knem.

    Mixed is a fail: one forbidden transport in the data tables means some bytes
    did not take the fabric. Text with no table or benchmark output is a fail too.
    """
    found = _ucx_transports(text)
    if _ib_write_bw_seen(text):
        found.append(IB_WRITE_BW)
    transports = list(dict.fromkeys(found))
    if not transports:
        return GateResult(False, [], "no UCX protocol table or ib_write_bw result in the text")
    bad = [t for t in transports if t in NOT_RDMA]
    if bad:
        return GateResult(False, transports, f"not RDMA: {', '.join(bad)}")
    good = [t for t in transports if t in RDMA or t == IB_WRITE_BW]
    if not good:
        return GateResult(False, transports, f"no RDMA transport among {', '.join(transports)}")
    return GateResult(True, transports, f"RDMA: {', '.join(good)}")


def bandwidth_gbps(text: str) -> float | None:
    """Best bandwidth in Gb/s that ib_write_bw or ucx_perftest printed, if any."""
    found: list[float] = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        # ib_write_bw: `#bytes #iterations BW peak[Gb/sec] BW average[Gb/sec] ...`
        if re.match(r"\s*#bytes\s+#iterations\s+BW", line):
            cols = re.findall(r"BW (\w+)\[([^\]]+)\]", line)
            col = next((k for k, (kind, _) in enumerate(cols) if kind == "average"), None)
            if col is None:
                continue
            gbit = cols[col][1] == "Gb/sec"
            for row in lines[i + 1 :]:
                nums = _NUMBER.findall(row)
                if not nums or row.lstrip().startswith("-"):
                    break
                if len(nums) > col + 2:
                    value = float(nums[col + 2])
                    found.append(value if gbit else value * _MIB_BITS)
        # ucx_perftest: `Final: iterations typical average overall bw_avg bw_overall mr_avg mr_overall`
        elif "Final:" in line:
            nums = _NUMBER.findall(line.split("Final:", 1)[1].replace("|", " "))
            if len(nums) >= 6:
                found.append(float(nums[5]) * _MIB_BITS)
    return max(found) if found else None


def gate(
    texts: dict[str, str], min_gbps: float = 50.0, require_bandwidth: bool = False
) -> GateResult:
    """Judge several logs together. A bad transport in any of them fails."""
    transports: list[str] = []
    reasons: list[str] = []
    ok = True
    judged = 0
    for name, text in texts.items():
        result = classify_transports(text)
        if not result.transports:
            continue  # a log with nothing to judge, such as a bench log
        judged += 1
        transports += [t for t in result.transports if t not in transports]
        if not result.ok:
            ok = False
            reasons.append(f"{name}: {result.reason}")
    if not judged:
        return GateResult(False, [], "no UCX protocol table or ib_write_bw result in any log")
    rates = {n: bandwidth_gbps(t) for n, t in texts.items()}
    rates = {n: r for n, r in rates.items() if r is not None}
    for name, rate in rates.items():
        if rate < min_gbps:
            ok = False
            reasons.append(f"{name}: {rate:.1f} Gb/s is below {min_gbps:g}")
    if require_bandwidth and not rates:
        ok = False
        reasons.append("no bandwidth in any log")
    if ok:
        best = f", best {max(rates.values()):.1f} Gb/s" if rates else ""
        reasons.append(f"transports {', '.join(transports)}{best}")
    return GateResult(ok, transports, "; ".join(reasons))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--min-gbps", type=float, default=50.0)
    parser.add_argument("--require-bandwidth", action="store_true")
    args = parser.parse_args(argv)
    texts = {p.name: p.read_text(errors="replace") for p in args.logs}
    result = gate(texts, args.min_gbps, args.require_bandwidth)
    print(f"{'PASS' if result.ok else 'FAIL'}: {result.reason}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
