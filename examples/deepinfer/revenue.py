#!/usr/bin/env python3
"""Turn AIPerf results into revenue-equivalent dollars per GPU-hour.

DeepSeek reports "US$38 per GPU-hour (at valley pricing)" for its production
cluster (doc/inference_deployment_qa.md in the DeepInfer repo). This prices the
tokens an AIPerf run measured at DeepSeek's published API rates, so a run on
your own hardware can be put next to that number. It is an estimate under the
assumptions printed with it, not a replication: see README.md in this folder.

    python revenue.py --gpus 8 results/<run>/aiperf

Each argument is a directory holding profile_export_aiperf.json, a parent whose
subdirectories do, or the JSON file itself. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

# USD per 1M tokens, off-peak ("valley") rates for deepseek-flash, which is
# served by DeepSeek-V4.1-Flash-1017. Off-peak is half of the peak rate. Read
# from https://api-docs.deepseek.com/quick_start/pricing on 2026-10-21; check
# the page before relying on them.
PRICE_INPUT_CACHE_MISS = 0.15
PRICE_INPUT_CACHE_HIT = 0.003
PRICE_OUTPUT = 0.60

# DeepSeek's reported cluster average, off-peak pricing.
DEEPSEEK_REPORTED_PER_GPU_HOUR = 38.0

# Approximate, read by eye from doc/images/wide_ep_decode_pareto.png in the
# DeepInfer repo: decode-only output tokens/s/GPU at context 131,072 with
# nextn=3. Keys are requests per DP rank, so with 8 DP ranks the total
# concurrency is 8x the key.
DEEPSEEK_CHART_EP8DP8 = {8: 1900, 16: 3700, 32: 7000, 64: 12700, 128: 19200, 256: 24900}
DEEPSEEK_CHART_EP64DP64 = {8: 2600, 16: 5000, 32: 9500, 64: 15700, 128: 22800, 256: 26300}

RESULT_FILE = "profile_export_aiperf.json"


@dataclass
class Run:
    path: str
    concurrency: int | None
    duration_s: float
    requests: int
    error_rate_pct: float
    input_tokens: float
    output_tokens: float
    token_source: str
    usage_input_tokens: float | None
    usage_output_tokens: float | None
    complete: bool

    @property
    def input_tps(self) -> float:
        return self.input_tokens / self.duration_s

    @property
    def output_tps(self) -> float:
        return self.output_tokens / self.duration_s

    def revenue_per_hour(self, hit_rate: float) -> float:
        """Dollars per hour if `hit_rate` of the input tokens were billed as
        cache hits. The run itself computed all of them; this is a billing
        scenario, not a measured cache."""
        per_million = (
            self.input_tokens * ((1 - hit_rate) * PRICE_INPUT_CACHE_MISS + hit_rate * PRICE_INPUT_CACHE_HIT)
            + self.output_tokens * PRICE_OUTPUT
        )
        return per_million / 1e6 / (self.duration_s / 3600)


def find_results(paths: list[str]) -> list[Path]:
    """Result files for the given paths, one per AIPerf run.

    AIPerf writes a top-level summary and, when there is a warmup phase, one
    more per phase under `phases/`. Only the summary is used, so a warmup is
    never counted as a run.
    """
    found: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_file():
            found.append(path)
        elif (path / RESULT_FILE).is_file():
            found.append(path / RESULT_FILE)
        else:
            candidates = sorted(path.rglob(RESULT_FILE))
            run_dirs = {c.parent for c in candidates if "phases" not in c.parent.parts}
            for candidate in candidates:
                if "phases" in candidate.parent.parts:
                    run_dir = candidate.parent.parent.parent
                    if run_dir in run_dirs or candidate.parent.name != "profiling":
                        continue
                found.append(candidate)
    return found


def _avg(data: dict, name: str) -> float | None:
    value = data.get(name)
    if isinstance(value, dict) and isinstance(value.get("avg"), (int, float)):
        return float(value["avg"])
    return None


def load_run(path: Path, token_source: str) -> Run:
    data = json.loads(path.read_text())
    duration = _avg(data, "benchmark_duration")
    if not duration or duration <= 0:
        raise ValueError("no benchmark_duration")
    counted_in, counted_out = _avg(data, "total_isl"), _avg(data, "total_osl")
    usage_in, usage_out = _avg(data, "total_usage_prompt_tokens"), _avg(data, "total_usage_completion_tokens")
    if token_source == "usage":
        if not usage_in or not usage_out:
            raise ValueError("the server reported no token usage; use --tokens aiperf")
        input_tokens, output_tokens = usage_in, usage_out
    else:
        if counted_in is None or counted_out is None:
            raise ValueError("no total_isl / total_osl")
        input_tokens, output_tokens = counted_in, counted_out
    phases = data.get("input_config", {}).get("phases", [])
    profiling = next((p for p in phases if p.get("kind") == "profiling"), {})
    return Run(
        path=str(path.parent),
        concurrency=profiling.get("concurrency"),
        duration_s=duration,
        requests=int(_avg(data, "request_count") or 0),
        error_rate_pct=_avg(data, "request_error_rate") or 0.0,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        token_source=token_source,
        usage_input_tokens=usage_in,
        usage_output_tokens=usage_out,
        complete=bool(data.get("is_complete")) and not data.get("was_cancelled"),
    )


def render(runs: list[Run], gpus: int, hit_rates: list[float]) -> str:
    lines = [
        f"GPUs: {gpus}.  Prices per 1M tokens (off-peak): input miss ${PRICE_INPUT_CACHE_MISS}, "
        f"input hit ${PRICE_INPUT_CACHE_HIT}, output ${PRICE_OUTPUT}.",
        f"Reference: DeepSeek reports ${DEEPSEEK_REPORTED_PER_GPU_HOUR:.0f} per GPU-hour for its production cluster.",
        "",
    ]
    header = ["conc", "reqs", "err%", "in tok/s/GPU", "out tok/s/GPU"] + [
        f"$/GPU-h @{int(h * 100)}% hit" for h in hit_rates
    ]
    rows = [header]
    for run in sorted(runs, key=lambda r: (r.concurrency or 0, r.path)):
        rows.append(
            [
                str(run.concurrency if run.concurrency is not None else "?"),
                str(run.requests),
                f"{run.error_rate_pct:.1f}",
                f"{run.input_tps / gpus:,.0f}",
                f"{run.output_tps / gpus:,.0f}",
            ]
            + [f"{run.revenue_per_hour(h) / gpus:.2f}" for h in hit_rates]
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    for row in rows:
        lines.append("  ".join(cell.rjust(widths[i]) for i, cell in enumerate(row)))
    best = max(runs, key=lambda r: r.revenue_per_hour(0.0))
    lines += [
        "",
        f"Best at 0% hit: ${best.revenue_per_hour(0.0) / gpus:.2f} per GPU-hour "
        f"({best.revenue_per_hour(0.0) / gpus / DEEPSEEK_REPORTED_PER_GPU_HOUR:.0%} of the reported figure).",
    ]
    skewed = [
        r
        for r in runs
        if r.usage_output_tokens and abs(r.output_tokens - r.usage_output_tokens) / r.usage_output_tokens > 0.05
    ]
    if skewed:
        lines.append(
            f"Note: AIPerf's own output-token count differs from the server's usage by >5% in {len(skewed)} run(s); "
            "compare with --tokens usage."
        )
    unfinished = [r.path for r in runs if not r.complete]
    if unfinished:
        lines.append(f"Warning: incomplete or cancelled runs: {', '.join(unfinished)}")
    lines += [
        "",
        "How to read this:",
        "- Concurrency is total across all GPUs. DeepSeek's chart counts requests per DP rank, and its",
        "  EP8DP8 point has 8 ranks, so 64 on the chart is 512 here. This backend serves one DP rank.",
        "- Output-only reference, decode at context 131,072 (read from DeepSeek's chart, approximate):",
        "  EP8DP8 tok/s/GPU at 8/16/32/64/128/256 per rank = "
        + "/".join(f"{v:,}" for v in DEEPSEEK_CHART_EP8DP8.values())
        + "; EP64DP64 = "
        + "/".join(f"{v:,}" for v in DEEPSEEK_CHART_EP64DP64.values())
        + ".",
        f"  At ${PRICE_OUTPUT}/M output that is about ${DEEPSEEK_CHART_EP8DP8[128] * 3600 / 1e6 * PRICE_OUTPUT:.0f} per GPU-hour"
        f" for EP8DP8 at 128 per rank, from output tokens alone.",
        "- A hit-rate column is a billing scenario: this run computed every input token, so it does not",
        "  show what a real KV-cache store would save. DeepSeek's figure also covers prefill GPUs.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="AIPerf result directories or JSON files")
    parser.add_argument("--gpus", type=int, required=True, help="GPUs the run used")
    parser.add_argument("--hit-rates", default="0,0.5,0.9", help="comma-separated cache-hit scenarios (default 0,0.5,0.9)")
    parser.add_argument("--tokens", choices=["aiperf", "usage"], default="aiperf", help="count tokens as AIPerf does, or from the server's usage")
    parser.add_argument("--json", metavar="FILE", help="also write the rows as JSON")
    args = parser.parse_args(argv)
    if args.gpus < 1:
        parser.error("--gpus must be at least 1")
    try:
        hit_rates = [float(x) for x in args.hit_rates.split(",") if x.strip()]
    except ValueError:
        parser.error("--hit-rates must be numbers between 0 and 1")
    if not hit_rates or any(not 0 <= h <= 1 for h in hit_rates):
        parser.error("--hit-rates must be numbers between 0 and 1")

    files = find_results(args.paths)
    if not files:
        print(f"no {RESULT_FILE} found under: {', '.join(args.paths)}", file=sys.stderr)
        return 1
    runs = []
    for file in files:
        try:
            runs.append(load_run(file, args.tokens))
        except (ValueError, json.JSONDecodeError, OSError) as exc:
            print(f"skipping {file}: {exc}", file=sys.stderr)
    if not runs:
        return 1
    print(render(runs, args.gpus, hit_rates))
    if args.json:
        rows = []
        for run in runs:
            row = asdict(run)
            row["input_tokens_per_s_per_gpu"] = run.input_tps / args.gpus
            row["output_tokens_per_s_per_gpu"] = run.output_tps / args.gpus
            row["usd_per_gpu_hour"] = {str(h): run.revenue_per_hour(h) / args.gpus for h in hit_rates}
            rows.append(row)
        Path(args.json).write_text(json.dumps({"gpus": args.gpus, "runs": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
