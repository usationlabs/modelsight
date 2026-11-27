"""M0: how long does vLLM's OffloadingConnector take to load a prefix from CPU?

SCRATCH, not committed. Run as the bench of examples/kv-transfer/offload_load_latency.yaml,
against one Qwen3-8B engine with OffloadingConnector, whose load jobs are
timestamped by probes/offload_latency_hook/sitecustomize.py.

The design question (design/vllm/peer-pin-cpu-stage-fallback.md, "Job shape: measure
first"): can a peer stage, which copies the CPU-only tail of a prefix into GPU
blocks, be one load job per pin, or does it need one job per chunk? The rule:
if p99 submit-to-report latency of a ~10% of pool tail is under 0.5 s while the
engine is under load, one job per pin; otherwise per chunk.

A CPU-only hit is made without a second engine: a prompt is sent (its KV is
stored to CPU), the GPU prefix cache alone is reset (POST /reset_prefix_cache,
no reset_external), and the same prompt is sent again, so its blocks can only
come from CPU. The resend is the measured load.

Scenarios, for each prefix size:

  idle        nothing else running
  loaded      background traffic, 12 long decodes and 4 loops of unique long
              prompts by default, so the engine is busy and keeps storing to CPU
  concurrent  (largest size only) N CPU-only prompts submitted at once under the
              same background load, which is what several pins at a time look like

The reset fails while any request runs, so the CPU-only prompts of the loaded
scenarios are prepared in batches on the idle engine, then the background starts
and the prompts are sent one by one (or N at once).

Per load job, from the hook: submit-to-report latency, the transfer's own copy
time (start to end event), and bytes. Cross-checks: client TTFT for a CPU hit, a
GPU hit (the resend before the reset) and a recompute (a fresh prompt), and the
change of vllm:kv_offload_load_bytes / load_time against the job log.

Prints a table and the M0 verdict, and writes results.json to the current
directory (the run folder).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from modelsight.probes import kv_transfer as kv

VERDICT_P99_SECONDS = 0.5
# A CPU-hit job moves at least this fraction of the prompt's KV bytes, or the
# hit was not (fully) from CPU.
MIN_JOB_FRACTION = 0.9
MIN_STORE_FRACTION = 0.98
STORE_WAIT_SECONDS = 30.0
RESET_ATTEMPTS = 60
MIN_VERDICT_JOBS = 10
EXPECTED_QWEN3_8B_BYTES_PER_TOKEN = 36 * 2 * 8 * 128 * 2  # layers, K+V, heads, dim, bf16

METRICS = ("vllm:kv_offload_load_bytes", "vllm:kv_offload_load_time", "vllm:kv_offload_store_bytes")


# Client --------------------------------------------------------------------


@dataclass(frozen=True)
class Client:
    url: str
    model: str

    def open(self, body: dict, timeout: float = 600):
        request = urllib.request.Request(
            f"{self.url}/v1/completions",
            data=json.dumps(
                {"model": self.model, "temperature": 0, "stream": True, **body}
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            # The server's reason is in the body; without it a 400 says nothing.
            detail = exc.read().decode(errors="replace")[:2000]
            raise RuntimeError(f"{self.url} returned {exc.code}: {detail}") from exc

    def ttft(self, prompt: list[int]) -> float:
        """Seconds to the first streamed token of a 1-token completion."""
        started = time.perf_counter()
        first = None
        with self.open({"prompt": prompt, "max_tokens": 1}) as response:
            for line in response:
                if first is None and line.startswith(b"data:") and b"[DONE]" not in line:
                    first = time.perf_counter() - started
        if first is None:
            raise RuntimeError("the stream ended without a token")
        return first


def scrape(url: str) -> dict:
    """Prometheus series summed over labels, plus the cache config labels."""
    out: dict = {}
    with urllib.request.urlopen(f"{url}/metrics", timeout=10) as response:
        for line in response.read().decode().splitlines():
            if line.startswith("#") or not line.strip():
                continue
            name = line.split("{")[0].split(" ")[0]
            if name == "vllm:cache_config_info":
                for key in ("block_size", "num_gpu_blocks"):
                    match = re.search(rf'{key}="(\d+)"', line)
                    if match:
                        out[key] = int(match.group(1))
                continue
            try:
                out[name] = out.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
            except ValueError:
                pass
    return out


def counter(metrics: dict, base: str) -> float:
    return metrics.get(f"{base}_total", metrics.get(base, 0.0))


def nudge(client: Client, rng: random.Random) -> None:
    """A tiny request, so an idle engine takes a step and reports finished jobs."""
    try:
        client.ttft(kv.fresh_prompt(rng, 16))
    except (OSError, RuntimeError):
        pass


def wait_stored(client: Client, rng: random.Random, before: float, expected: float | None) -> bool:
    """Wait until the CPU store of the last prompt has been reported.

    With `expected` bytes, until the store counter has grown by that much; without,
    until it has grown and then stood still for a second.
    """
    deadline = time.monotonic() + STORE_WAIT_SECONDS
    last, last_change, last_nudge = before, time.monotonic(), time.monotonic()
    while time.monotonic() < deadline:
        now = counter(scrape(client.url), "vllm:kv_offload_store_bytes")
        if expected is not None and now - before >= MIN_STORE_FRACTION * expected:
            return True
        if now != last:
            last, last_change = now, time.monotonic()
        elif expected is None and now > before and time.monotonic() - last_change > 1.0:
            return True
        if time.monotonic() - max(last_change, last_nudge) > 1.0:
            nudge(client, rng)
            last_nudge = time.monotonic()
        time.sleep(0.1)
    return False


def reset_gpu(client: Client, rng: random.Random) -> None:
    """Reset the GPU prefix cache only. vLLM refuses while any block is held."""
    request = urllib.request.Request(
        f"{client.url}/reset_prefix_cache", data=b"", method="POST"
    )
    for _ in range(RESET_ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                if json.load(response).get("success", True):
                    return
        except (OSError, ValueError):
            pass
        nudge(client, rng)
        time.sleep(0.5)
    raise RuntimeError("the GPU prefix cache would not reset")


# Job log -------------------------------------------------------------------


class JobLog:
    """The hook's one-line-per-load-job file, read incrementally."""

    def __init__(self, path: Path):
        self.path = path
        self.offset = 0
        self.all: list[dict] = []

    def take(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open("rb") as f:
            f.seek(self.offset)
            data = f.read()
        end = data.rfind(b"\n") + 1  # leave a half-written line for the next call
        self.offset += end
        jobs = [json.loads(line) for line in data[:end].splitlines() if line.strip()]
        self.all += jobs
        return jobs


# Background load ------------------------------------------------------------


class Background:
    """Concurrent traffic: long decodes, and loops of unique long prompts."""

    def __init__(self, client: Client, decoders: int, prefillers: int,
                 decode_prompt: int, decode_tokens: int, prefill_prompt: int, seed: int):
        self.client = client
        self.plan = [("decode", i) for i in range(decoders)] + [
            ("prefill", decoders + i) for i in range(prefillers)
        ]
        self.decode_prompt, self.decode_tokens = decode_prompt, decode_tokens
        self.prefill_prompt = prefill_prompt
        self.seed = seed
        self.stop_flag = threading.Event()
        self.threads: list[threading.Thread] = []
        self.errors = 0
        self.completed = 0
        self.samples: list[tuple[float, float, float]] = []  # time, running, waiting
        self._sampler: threading.Thread | None = None

    def _loop(self, kind: str, index: int) -> None:
        rng = random.Random(self.seed * 10007 + index)
        while not self.stop_flag.is_set():
            if kind == "decode":
                body = {"prompt": kv.fresh_prompt(rng, self.decode_prompt),
                        "max_tokens": self.decode_tokens, "ignore_eos": True}
            else:
                body = {"prompt": kv.fresh_prompt(rng, self.prefill_prompt),
                        "max_tokens": 4, "ignore_eos": True}
            try:
                response = self.client.open(body)
                try:
                    for _ in response:
                        if self.stop_flag.is_set():
                            break  # closing the connection aborts the request
                    else:
                        self.completed += 1
                finally:
                    response.close()
            except (OSError, urllib.error.URLError):
                self.errors += 1
                time.sleep(0.5)

    def _sample(self) -> None:
        while not self.stop_flag.is_set():
            try:
                m = scrape(self.client.url)
                self.samples.append((time.monotonic(), m.get("vllm:num_requests_running", 0.0),
                                     m.get("vllm:num_requests_waiting", 0.0)))
            except OSError:
                pass
            time.sleep(0.25)

    def start(self, settle_seconds: float = 2.0, busy_timeout: float = 90.0) -> float:
        """Start the traffic; return how many requests are running once it is busy."""
        self.stop_flag.clear()
        self.samples = []
        self.threads = [threading.Thread(target=self._loop, args=p, daemon=True) for p in self.plan]
        for t in self.threads:
            t.start()
        self._sampler = threading.Thread(target=self._sample, daemon=True)
        self._sampler.start()
        want = 0.75 * len(self.plan)
        deadline = time.monotonic() + busy_timeout
        while time.monotonic() < deadline:
            if self.samples and self.samples[-1][1] >= want:
                break
            time.sleep(0.25)
        else:
            raise RuntimeError(f"the engine never reached {want:.0f} running requests")
        time.sleep(settle_seconds)
        return self.samples[-1][1]

    def stop(self) -> None:
        self.stop_flag.set()
        for t in self.threads:
            t.join(timeout=30)
        if self._sampler:
            self._sampler.join(timeout=5)

    def running(self) -> dict:
        running = [r for _, r, _ in self.samples]
        waiting = [w for _, _, w in self.samples]
        if not running:
            return {}
        return {"running_median": statistics.median(running), "running_min": min(running),
                "running_max": max(running), "waiting_median": statistics.median(waiting)}


# Measurement -----------------------------------------------------------------


@dataclass
class Setup:
    client: Client
    jobs: JobLog
    rng: random.Random
    pool_tokens: int
    block_size: int
    bytes_per_token: float
    args: argparse.Namespace

    def expected_bytes(self, tokens: int) -> float:
        return tokens * self.bytes_per_token


def calibrate(client: Client, rng: random.Random, block_size: int, override: float | None) -> float:
    """KV bytes per token, from what one prompt adds to the store counter."""
    if override:
        return override
    tokens = 128 * block_size + block_size // 2  # 128 full blocks and a partial one
    before = counter(scrape(client.url), "vllm:kv_offload_store_bytes")
    client.ttft(kv.fresh_prompt(rng, tokens))
    if not wait_stored(client, rng, before, None):
        raise RuntimeError("calibration prompt never reached the CPU store")
    stored = counter(scrape(client.url), "vllm:kv_offload_store_bytes") - before
    per_token = stored / (128 * block_size)
    note = ""
    if abs(per_token / EXPECTED_QWEN3_8B_BYTES_PER_TOKEN - 1) > 0.02:
        note = f" (Qwen3-8B bf16 should be {EXPECTED_QWEN3_8B_BYTES_PER_TOKEN})"
    print(f"calibration: stored {stored:.0f} B for 128 blocks -> {per_token:.0f} B/token{note}", flush=True)
    return per_token


def prepare(s: Setup, prompts: list[list[int]], gpu_hit: bool = False) -> list[dict]:
    """Send each prompt once so it is stored to CPU, then reset the GPU cache.

    With `gpu_hit` each is also resent before the reset, to time a GPU hit.
    Returns the recompute (and GPU hit) TTFTs, per prompt.
    """
    out = []
    for prompt in prompts:
        before = counter(scrape(s.client.url), "vllm:kv_offload_store_bytes")
        row = {"recompute_ttft_s": s.client.ttft(prompt)}
        row["stored"] = wait_stored(s.client, s.rng, before, s.expected_bytes(len(prompt) - s.block_size))
        if gpu_hit:
            row["gpu_ttft_s"] = s.client.ttft(prompt)
        out.append(row)
    reset_gpu(s.client, s.rng)
    return out


def collect_jobs(s: Setup, prompt_tokens: int, expect: int) -> tuple[list[dict], int]:
    """The CPU-hit load jobs since the last call: the big ones, and how many others."""
    deadline = time.monotonic() + 2.0
    big: list[dict] = []
    other = 0
    while True:
        for job in s.jobs.take():
            if job.get("success") and (job.get("bytes") or 0) >= MIN_JOB_FRACTION * s.expected_bytes(prompt_tokens - s.block_size):
                big.append(job)
            else:
                other += 1
        if len(big) >= expect or time.monotonic() > deadline:
            return big, other
        time.sleep(0.05)


def job_row(job: dict) -> dict:
    return {k: job[k] for k in ("job", "bytes", "copy_s", "submit_to_seen_s", "submit_to_report_s",
                                "submit_call_s", "expedited")}


def measure_idle(s: Setup, size: int, reps: int) -> list[dict]:
    records = []
    attempts, bad = 0, 0
    while sum(r["valid"] for r in records) < reps and attempts < reps * 3:
        attempts += 1
        prompt = kv.fresh_prompt(s.rng, size)
        s.jobs.take()
        prep = prepare(s, [prompt], gpu_hit=True)[0]
        s.jobs.take()  # drop jobs of the preparation, if any
        before = scrape(s.client.url)
        cpu_ttft = s.client.ttft(prompt)
        jobs, other = collect_jobs(s, size, 1)
        moved = counter(scrape(s.client.url), "vllm:kv_offload_load_bytes") - counter(before, "vllm:kv_offload_load_bytes")
        valid = len(jobs) == 1 and prep["stored"]
        records.append({"size": size, "scenario": "idle", "valid": valid, "cpu_ttft_s": cpu_ttft,
                        "gpu_ttft_s": prep["gpu_ttft_s"], "recompute_ttft_s": prep["recompute_ttft_s"],
                        "jobs": [job_row(j) for j in jobs], "other_jobs": other, "load_bytes_metric": moved})
        report(records[-1])
        bad = 0 if valid else bad + 1
        if bad >= s.args.abort_after_invalid:
            raise RuntimeError(f"{bad} CPU hits in a row did not load from CPU; is the CPU tier surviving the reset?")
    return records


def measure_loaded(s: Setup, size: int, reps: int, per_rep: int, bg: Background) -> list[dict]:
    """per_rep == 1: one CPU hit at a time. More: that many at once."""
    scenario = "loaded" if per_rep == 1 else "concurrent"
    # A batch is CPU-resident prompts prepared on the idle engine, then measured
    # under load. Its size is capped so the background cannot evict them from CPU.
    batch_reps = max(1, min(s.args.max_batch_prompts // per_rep, reps))
    records, attempts, bad = [], 0, 0
    while sum(r["valid"] for r in records) < reps and attempts < reps * 3:
        n = min(batch_reps, reps - sum(r["valid"] for r in records))
        attempts += n
        prompts = [kv.fresh_prompt(s.rng, size) for _ in range(n * per_rep)]
        s.jobs.take()
        prepare(s, prompts)
        s.jobs.take()
        running = bg.start()
        try:
            for i in range(n):
                group = prompts[i * per_rep:(i + 1) * per_rep]
                rec = one_loaded_rep(s, size, scenario, group, running)
                records.append(rec)
                report(rec)
                bad = 0 if rec["valid"] else bad + 1
                if bad >= s.args.abort_after_invalid:
                    raise RuntimeError(f"{bad} CPU hits in a row did not load from CPU")
                time.sleep(1.0)
        finally:
            bg.stop()
        records[-1]["background"] = bg.running()
        print(f"  background: {bg.running()} completed={bg.completed} errors={bg.errors}", flush=True)
    return records


def one_loaded_rep(s: Setup, size: int, scenario: str, group: list[list[int]], running: float) -> dict:
    now = scrape(s.client.url)
    running_at_send = now.get("vllm:num_requests_running", running)
    before = counter(now, "vllm:kv_offload_load_bytes")
    s.jobs.take()
    ttfts: list[float | None] = [None] * len(group)
    errors: list[str] = []
    barrier = threading.Barrier(len(group))

    def send(i: int) -> None:
        try:
            barrier.wait()
            ttfts[i] = s.client.ttft(group[i])
        except (OSError, RuntimeError, threading.BrokenBarrierError) as exc:
            errors.append(repr(exc))

    if len(group) == 1:
        send(0)
    else:
        threads = [threading.Thread(target=send, args=(i,)) for i in range(len(group))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    jobs, other = collect_jobs(s, size, len(group))
    moved = counter(scrape(s.client.url), "vllm:kv_offload_load_bytes") - before
    rec = {"size": size, "scenario": scenario, "valid": len(jobs) == len(group) and not errors,
           "cpu_ttft_s": statistics.median(t for t in ttfts if t is not None) if any(ttfts) else None,
           "cpu_ttfts_s": ttfts, "jobs": [job_row(j) for j in jobs], "other_jobs": other,
           "load_bytes_metric": moved, "running_at_send": running_at_send, "errors": errors}
    if len(group) == 1 and not errors:
        rec["gpu_ttft_s"] = s.client.ttft(group[0])  # its blocks are on GPU now
        rec["recompute_ttft_s"] = s.client.ttft(kv.fresh_prompt(s.rng, size))
        s.jobs.take()  # the recompute does not load, but drop whatever it logged
    return rec


# Reporting --------------------------------------------------------------------


def report(r: dict) -> None:
    jobs = " ".join(
        f"{j['submit_to_report_s'] * 1e3:.0f}ms(copy {1e3 * (j['copy_s'] or 0):.0f})" for j in r["jobs"]
    )
    print(
        f"{r['scenario']} {r['size']}: {'ok ' if r['valid'] else 'BAD'} cpu_ttft="
        f"{(r['cpu_ttft_s'] or 0):.3f}s gpu={r.get('gpu_ttft_s', 0):.3f}s "
        f"recompute={r.get('recompute_ttft_s', 0):.3f}s jobs=[{jobs}] other={r['other_jobs']}"
        + (f" running={r['running_at_send']:.0f}" if "running_at_send" in r else ""),
        flush=True,
    )


def pct(values: list[float], fraction: float) -> float:
    return kv.percentile(values, fraction)


def summarize(records: list[dict], s: Setup) -> list[dict]:
    rows = []
    keys = sorted({(r["scenario"], r["size"]) for r in records},
                  key=lambda k: (("idle", "loaded", "concurrent").index(k[0]), k[1]))
    for scenario, size in keys:
        group = [r for r in records if r["scenario"] == scenario and r["size"] == size and r["valid"]]
        jobs = [j for r in group for j in r["jobs"]]
        latency = [j["submit_to_report_s"] for j in jobs]
        per_pin = 1 if scenario != "concurrent" else len(group[0]["jobs"]) if group else 1
        row = {"size": size, "scenario": scenario, "reps": len(group), "jobs": len(jobs),
               "pins_at_once": per_pin, "pct_pool": 100.0 * size * per_pin / s.pool_tokens,
               "gb": jobs and statistics.median(j["bytes"] for j in jobs) / 1e9 * per_pin or size * s.bytes_per_token / 1e9}
        if latency:
            row.update(p50=statistics.median(latency), p90=pct(latency, 0.9), p99=pct(latency, 0.99),
                       max=max(latency), copy_p50=statistics.median(j["copy_s"] for j in jobs),
                       seen_p50=statistics.median(j["submit_to_seen_s"] for j in jobs if j["submit_to_seen_s"] is not None))
        for name in ("cpu", "gpu", "recompute"):
            values = [r[f"{name}_ttft_s"] for r in group if r.get(f"{name}_ttft_s") is not None]
            row[f"{name}_ttft_p50"] = statistics.median(values) if values else None
        rows.append(row)
    return rows


def fmt(value, scale=1.0, digits=0) -> str:
    return "-" if value is None else f"{value * scale:.{digits}f}"


def print_table(rows: list[dict]) -> None:
    print("\nsize   %pool   GB    scenario    reps jobs | submit-to-report ms: p50   p90   p99   max |"
          " copy p50 | TTFT p50 ms: cpu   gpu  recompute")
    for r in rows:
        print(f"{r['size']:>5} {r['pct_pool']:>6.1f} {r['gb']:>5.2f}  {r['scenario']:<10} {r['reps']:>4} {r['jobs']:>4} |"
              f"{fmt(r.get('p50'), 1e3):>26} {fmt(r.get('p90'), 1e3):>5} {fmt(r.get('p99'), 1e3):>5} {fmt(r.get('max'), 1e3):>5} |"
              f" {fmt(r.get('copy_p50'), 1e3, 1):>8} |"
              f" {fmt(r['cpu_ttft_p50'], 1e3):>18} {fmt(r['gpu_ttft_p50'], 1e3):>5} {fmt(r['recompute_ttft_p50'], 1e3):>8}")
    print("p99 over fewer than 100 jobs is the max.", flush=True)


def verdict(rows: list[dict], size: int) -> str:
    wanted = [r for r in rows if r["size"] == size and r["scenario"] in ("loaded", "concurrent")]
    have = {r["scenario"]: r for r in wanted}
    if set(have) != {"loaded", "concurrent"} or any(r["jobs"] < MIN_VERDICT_JOBS or "p99" not in r for r in have.values()):
        return f"M0 verdict: INCONCLUSIVE (need {MIN_VERDICT_JOBS}+ valid jobs in the loaded and concurrent {size} cases)"
    worst = max(r["p99"] for r in have.values())
    detail = ", ".join(f"{k} p99={v['p99'] * 1e3:.0f} ms" for k, v in have.items())
    shape = "one job per pin" if worst < VERDICT_P99_SECONDS else "per chunk"
    return f"M0 verdict: {shape} ({detail}; rule: p99 < {VERDICT_P99_SECONDS * 1e3:.0f} ms)"


def hook_installed(log: Path) -> bool:
    text = log.read_text(errors="replace") if log.exists() else ""
    return "[LOADLAT] patched CPUOffloadingWorker" in text and "[LOADLAT] patched OffloadingConnectorScheduler" in text


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True)
    p.add_argument("--url", required=True)
    p.add_argument("--server-log", type=Path, required=True)
    p.add_argument("--jobs-file", type=Path, required=True, help="the hook's $OFFLOAD_LATENCY_LOG")
    p.add_argument("--sizes", default="4096,8192,16384,32768")
    p.add_argument("--reps", type=int, default=20, help="valid reps per size and scenario")
    p.add_argument("--concurrent", type=int, default=3, help="pins at once, for the largest size")
    p.add_argument("--max-batch-prompts", type=int, default=6, help="CPU-only prompts prepared per loaded batch")
    p.add_argument("--bg-decoders", type=int, default=12)
    p.add_argument("--bg-prefillers", type=int, default=4)
    p.add_argument("--bg-decode-prompt", type=int, default=1024)
    p.add_argument("--bg-decode-tokens", type=int, default=1024)
    p.add_argument("--bg-prefill-prompt", type=int, default=8192)
    p.add_argument("--kv-bytes-per-token", type=float, default=None)
    p.add_argument("--abort-after-invalid", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=Path("results.json"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sizes = [int(x) for x in args.sizes.split(",")]
    client = Client(args.url, args.model)
    kv.wait_ready(args.url)
    rng = random.Random(args.seed)

    if not hook_installed(args.server_log):
        print("FAIL: the sitecustomize hook did not patch the server (no [LOADLAT] patched lines)", flush=True)
        return 1
    config = scrape(args.url)
    block_size = config.get("block_size", 16)
    pool_tokens = kv.pool_tokens(args.server_log)
    if "num_gpu_blocks" in config and config["num_gpu_blocks"] * block_size != pool_tokens:
        print(f"note: pool {pool_tokens} tokens in the log, {config['num_gpu_blocks']} blocks x {block_size} in metrics", flush=True)
    bytes_per_token = calibrate(client, rng, block_size, args.kv_bytes_per_token)
    s = Setup(client, JobLog(args.jobs_file), rng, pool_tokens, block_size, bytes_per_token, args)
    print(f"GPU pool: {pool_tokens} tokens = {pool_tokens * bytes_per_token / 1e9:.1f} GB "
          f"(block {block_size} tokens, {block_size * bytes_per_token / 1e6:.2f} MB/block)", flush=True)
    for size in sizes:
        print(f"  {size} tokens = {100 * size / pool_tokens:.1f}% of pool, {size * bytes_per_token / 1e9:.2f} GB", flush=True)

    bg = Background(client, args.bg_decoders, args.bg_prefillers, args.bg_decode_prompt,
                    args.bg_decode_tokens, args.bg_prefill_prompt, args.seed)
    records: list[dict] = []
    error = None
    load_before = scrape(args.url)
    try:
        for size in sizes:
            records += measure_idle(s, size, args.reps)
        for size in sizes:
            records += measure_loaded(s, size, args.reps, 1, bg)
        records += measure_loaded(s, sizes[-1], args.reps, args.concurrent, bg)
    except (RuntimeError, OSError) as exc:
        error = repr(exc)
        print(f"FAIL: {error}", flush=True)
    finally:
        bg.stop()

    load_after = scrape(args.url)
    jobs = s.jobs.all
    metrics = {
        "load_bytes": counter(load_after, "vllm:kv_offload_load_bytes") - counter(load_before, "vllm:kv_offload_load_bytes"),
        "load_time_s": counter(load_after, "vllm:kv_offload_load_time") - counter(load_before, "vllm:kv_offload_load_time"),
        "hook_job_bytes": sum(j["bytes"] or 0 for j in jobs),
        "hook_copy_s": sum(j["copy_s"] or 0 for j in jobs),
        "hook_jobs": len(jobs),
    }
    rows = summarize(records, s)
    print_table(rows)
    print(f"cross-check vs /metrics: {json.dumps(metrics)}", flush=True)
    line = verdict(rows, sizes[-1])
    print(line, flush=True)
    args.output.write_text(json.dumps({
        "model": args.model, "pool_tokens": pool_tokens, "block_size": block_size,
        "bytes_per_token": bytes_per_token, "pool_gb": pool_tokens * bytes_per_token / 1e9,
        "table": rows, "verdict": line, "metrics_cross_check": metrics, "records": records,
        "all_jobs": jobs, "error": error, "args": {k: str(v) for k, v in vars(args).items()},
    }, indent=2, default=str))
    for path in (args.server_log, args.jobs_file):
        target = args.output.parent / path.name
        if path.exists() and path.resolve() != target.resolve():
            shutil.copy(path, target)
    return 1 if error else 0


if __name__ == "__main__":
    sys.exit(main())
