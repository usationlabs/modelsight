"""GPU-to-GPU bandwidth: which UCX transport NIXL uses, and how fast it reads.

Two measurements on a machine with at least two GPUs:

  peer_copy  torch copies a buffer from GPU 0 to GPU 1 in one process. This is
             the link itself (NVLink or PCIe), with no NIXL or UCX in the way.
  nixl_read  a target process on GPU 0 registers a buffer with NIXL; an
             initiator process on GPU 1 READs it, as vLLM's consumer reads a
             producer's KV blocks. The same bytes are split into 1 up to
             thousands of descriptors, because vLLM's reads are many small
             blocks, not one large one.

The initiator runs with UCX_PROTO_INFO=y, so UCX prints the protocol it picks
for remote reads (cuda_ipc means a direct GPU-to-GPU copy over NVLink or PCIe).
That table, `nvidia-smi nvlink --status` and `ucx_info -d` are saved with the
results.

Writes results.json to the current directory, which is the run folder. Exits
with status 1 if a measurement fails or UCX did not pick cuda_ipc.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from multiprocessing.connection import Client, Listener
from pathlib import Path

from modelsight.probes.kv_transfer import read_path

GIB = 1 << 30
DEFAULT_SIZE_GIB = 4.0
# 36 is what vLLM used per pull in H4 (about 3.9 GB of Qwen3-8B KV); larger
# counts show how far a fragmented read falls.
DEFAULT_DESCRIPTORS = "1,36,431,4096,16384"
DEFAULT_REPEATS = 5
AUTH = b"modelsight-nixl-bandwidth"
XFER_TIMEOUT_SECONDS = 120.0
HANDSHAKE_TIMEOUT_SECONDS = 120.0


# Measurements -----------------------------------------------------------------


def gbps(nbytes: int, seconds: float) -> float:
    return nbytes / seconds / 1e9 if seconds > 0 else 0.0


def chunks(base: int, size: int, count: int, device: int) -> list[tuple[int, int, int]]:
    """Split [base, base+size) into `count` contiguous (addr, len, device) pieces.

    The pieces cover the same bytes whatever the count, so only the number of
    descriptors changes between rows.
    """
    step = size // count
    return [(base + i * step, step, device) for i in range(count)]


def peer_copy(size: int, repeats: int) -> dict:
    """GPU 0 to GPU 1 with torch, in one process."""
    import torch

    if torch.cuda.device_count() < 2:
        return {"error": f"need 2 GPUs, found {torch.cuda.device_count()}"}
    src = torch.empty(size, dtype=torch.uint8, device="cuda:0")
    dst = torch.empty(size, dtype=torch.uint8, device="cuda:1")
    times = []
    for i in range(repeats + 1):  # the first copy warms up the peer mapping
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)
        started = time.perf_counter()
        dst.copy_(src, non_blocking=True)
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)
        if i:
            times.append(time.perf_counter() - started)
    del src, dst
    torch.cuda.empty_cache()  # leave the GPUs to the NIXL processes
    return {
        "bytes": size,
        "peer_access": bool(torch.cuda.can_device_access_peer(0, 1)),
        "seconds": times,
        "gbps_median": gbps(size, statistics.median(times)),
        "gbps_max": gbps(size, min(times)),
    }


def nixl_agent(name: str):
    """A NIXL agent with only the UCX backend, in the error mode Modal needs."""
    from nixl._api import nixl_agent as agent_cls
    from nixl._api import nixl_agent_config

    agent = agent_cls(name, nixl_agent_config(backends=[]))
    # NIXL's default "peer" mode needs tcp or RDMA, which UCX lacks on Modal.
    agent.create_backend("UCX", {"ucx_error_handling_mode": "none"})
    return agent


def run_target(port: int, size: int) -> int:
    """Hold a registered GPU buffer until the initiator is done with it."""
    import torch

    buf = torch.ones(size, dtype=torch.uint8, device="cuda:0")
    agent = nixl_agent("target")
    agent.register_memory(agent.get_reg_descs([(buf.data_ptr(), size, 0, "")], "VRAM"))
    with Listener(("127.0.0.1", port), authkey=AUTH) as listener:
        print(f"target ready on port {port}", flush=True)
        with listener.accept() as conn:
            conn.send({"metadata": agent.get_agent_metadata(),
                       "addr": buf.data_ptr(), "size": size})  # fmt: skip
            conn.recv()  # "done"
    return 0


def read_once(agent, local: list, remote: list, remote_name: str) -> float:
    """One NIXL READ of `remote` into `local`; seconds from post to DONE."""
    local_descs = agent.get_xfer_descs(local, "VRAM")
    remote_descs = agent.get_xfer_descs(remote, "VRAM")
    handle = agent.initialize_xfer("READ", local_descs, remote_descs, remote_name)
    started = time.perf_counter()
    state = agent.transfer(handle)
    deadline = started + XFER_TIMEOUT_SECONDS
    while state not in ("DONE", "ERR"):
        if time.perf_counter() > deadline:
            raise TimeoutError(f"READ of {len(local)} descriptors did not finish")
        state = agent.check_xfer_state(handle)
    elapsed = time.perf_counter() - started
    agent.release_xfer_handle(handle)
    if state == "ERR":
        raise RuntimeError(f"READ of {len(local)} descriptors failed")
    return elapsed


def run_initiator(port: int, size: int, counts: list[int], repeats: int) -> int:
    """READ the target's buffer at each descriptor count; one JSON line per row."""
    import torch

    buf = torch.zeros(size, dtype=torch.uint8, device="cuda:0")
    agent = nixl_agent("initiator")
    agent.register_memory(agent.get_reg_descs([(buf.data_ptr(), size, 0, "")], "VRAM"))
    deadline = time.monotonic() + HANDSHAKE_TIMEOUT_SECONDS
    while True:
        try:
            conn = Client(("127.0.0.1", port), authkey=AUTH)
            break
        except OSError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.2)
    with conn:
        info = conn.recv()
        remote_name = agent.add_remote_agent(info["metadata"])
        for count in counts:
            local = chunks(buf.data_ptr(), size, count, 0)
            remote = chunks(info["addr"], size, count, 0)
            moved = local[0][1] * count
            row: dict = {"descriptors": count, "bytes": moved}
            try:
                times = [read_once(agent, local, remote, remote_name)
                         for _ in range(repeats + 1)][1:]  # fmt: skip
                row.update(seconds=times,
                           gbps_median=gbps(moved, statistics.median(times)),
                           gbps_max=gbps(moved, min(times)))  # fmt: skip
            except (RuntimeError, TimeoutError) as exc:
                row["error"] = repr(exc)
            print("ROW " + json.dumps(row), flush=True)
        torch.cuda.synchronize()
        row_check = bool((buf[:1024] == 1).all().item())
        print("ROW " + json.dumps({"data_ok": row_check}), flush=True)
        conn.send("done")
    return 0


# Orchestration ----------------------------------------------------------------


def shell(command: str) -> str:
    try:
        done = subprocess.run(command, shell=True, capture_output=True, text=True, check=False,
                              timeout=60)  # fmt: skip
        return (done.stdout + done.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return repr(exc)


def role_env(gpu: int) -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["UCX_PROTO_INFO"] = "y"
    return env


def nixl_read(
    args: argparse.Namespace, size: int, counts: list[int], logdir: Path
) -> dict:
    """Run target (GPU 0) and initiator (GPU 1) as separate processes."""
    base = [sys.executable, "-u", "-m", "modelsight.probes.nixl_bandwidth",
            "--port", str(args.port), "--size-gib", str(args.size_gib)]  # fmt: skip
    target_log = logdir / "nixl-target.log"
    initiator_log = logdir / "nixl-initiator.log"
    with target_log.open("w") as tout, initiator_log.open("w") as iout:
        target = subprocess.Popen(base + ["--role", "target"], env=role_env(0),
                                  stdout=tout, stderr=subprocess.STDOUT)  # fmt: skip
        initiator = subprocess.Popen(
            base + ["--role", "initiator", "--descriptors", ",".join(map(str, counts)),
                    "--repeats", str(args.repeats)],
            env=role_env(1), stdout=iout, stderr=subprocess.STDOUT)  # fmt: skip
        try:
            initiator.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            initiator.kill()
        try:
            target.wait(timeout=30)
        except subprocess.TimeoutExpired:
            target.kill()
    text = initiator_log.read_text(errors="replace")
    rows = [
        json.loads(line[4:]) for line in text.splitlines() if line.startswith("ROW ")
    ]
    data_ok = next((r["data_ok"] for r in rows if "data_ok" in r), None)
    return {
        "rows": [r for r in rows if "descriptors" in r],
        "data_ok": data_ok,
        "read_path": read_path(text),
        "initiator_exit": initiator.returncode,
        "target_exit": target.returncode,
    }


def verdict(result: dict) -> str:
    nixl = result["nixl_read"]
    if nixl["initiator_exit"] != 0 or not nixl["rows"]:
        return "FAIL: the NIXL initiator did not finish; see nixl-initiator.log"
    failed = [r["descriptors"] for r in nixl["rows"] if "error" in r]
    if failed:
        return f"FAIL: NIXL reads failed at descriptor counts {failed}"
    if nixl["data_ok"] is False:
        return "FAIL: the bytes read do not match the target's buffer"
    if not nixl["read_path"]:
        return "UNCONFIRMED: reads ran, but UCX's read table is not in the log"
    if not any("cuda_ipc" in row for row in nixl["read_path"]):
        return "FAIL: UCX did not select cuda_ipc for reads"
    return "PASS"


def print_report(result: dict) -> None:
    copy = result["peer_copy"]
    if "error" in copy:
        print(f"peer_copy: {copy['error']}")
    else:
        print(f"peer_copy GPU0->GPU1  {copy['bytes'] / GIB:.1f} GiB  "
              f"median {copy['gbps_median']:.1f} GB/s  max {copy['gbps_max']:.1f} GB/s  "
              f"peer_access={copy['peer_access']}")  # fmt: skip
    print("nixl_read  descriptors   MB/desc   median_GB/s   max_GB/s")
    for row in result["nixl_read"]["rows"]:
        per = row["bytes"] / row["descriptors"] / 1e6
        if "error" in row:
            print(f"           {row['descriptors']:>11}  {per:>8.2f}   {row['error']}")
        else:
            print(f"           {row['descriptors']:>11}  {per:>8.2f}   "
                  f"{row['gbps_median']:>11.1f}   {row['gbps_max']:>8.1f}")  # fmt: skip
    path = result["nixl_read"]["read_path"]
    print(
        "UCX read path:"
        + ("".join(f"\n  {row}" for row in path) if path else " (not logged)")
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size-gib", type=float, default=DEFAULT_SIZE_GIB)
    parser.add_argument("--descriptors", default=DEFAULT_DESCRIPTORS)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--port", type=int, default=5700)
    parser.add_argument("--timeout", type=float, default=15 * 60)
    parser.add_argument("--output", type=Path, default=Path("results.json"))
    parser.add_argument(
        "--role", choices=("target", "initiator"), help=argparse.SUPPRESS
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    size = int(args.size_gib * GIB)
    counts = [int(c) for c in args.descriptors.split(",")]
    if args.role == "target":
        return run_target(args.port, size)
    if args.role == "initiator":
        return run_initiator(args.port, size, counts, args.repeats)

    logdir = args.output.parent
    result = {
        "size_bytes": size,
        "nvlink_status": shell("nvidia-smi nvlink --status"),
        "ucx_transports": shell("ucx_info -d | grep -E 'Transport|Device'"),
        "peer_copy": peer_copy(size, args.repeats),
        "nixl_read": nixl_read(args, size, counts, logdir),
    }
    result["verdict"] = verdict(result)
    args.output.write_text(json.dumps(result, indent=2))
    print_report(result)
    print(result["verdict"], flush=True)
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
