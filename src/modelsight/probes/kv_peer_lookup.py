"""H4: can B take a prefix from A's cache without A computing anything?

Run as the bench of examples/kv-transfer/h4.yaml, against two vLLM servers built
from the peer lookup-and-pin branch. Prompts are turns of real agent sessions:
each turn's prompt extends the previous one, as an agent's context does.

Cases, each on a fresh pair of prefix caches:

  full     A holds the whole prompt; B pulls it all from A
  partial  A holds the previous turn; B pulls that and computes the new turn
  miss     A holds nothing; B computes everything

Arms per case:

  peer       one request to B with the prompt's block hashes; B asks A, A pins
             what it holds and B pulls it (needs the branch; --mode peer)
  stock      today's flow: A prefills, B pulls, B serves
  recompute  B computes the prompt itself

Checks, which fail the run:

  served       A answered no request for the peer arm: its request counter does
               not move, so nothing was computed there
  lookup       A's log shows the lookup (--lookup-pattern)
  pulled       B's NIXL bytes match the prefix A held, per byte-per-token
               calibration from the stock arm; zero on a miss
  equivalent   B's greedy tokens after a pull match B's own recompute
  no_leaks     A can reset its prefix cache at the end, which vLLM refuses
               while any block is still held

Writes results.json to the current directory.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import statistics
import sys
import time
import urllib.request
from collections.abc import Callable, Iterable
from itertools import pairwise
from pathlib import Path

from modelsight.probes import kv_transfer as kv
from modelsight.probes.kv_transfer import Servers

DEFAULT_DATASET = "nebius/SWE-agent-trajectories"
CASES = ("full", "partial", "miss")
CHECK_TOKENS = 16
# Relative error allowed between pulled bytes and the calibrated expectation.
BYTES_TOLERANCE = 0.1
MIN_EQUIVALENT_FRACTION = 0.95
LOOKUP_PATTERN = r"lookup[_ -]?pin"
REQUEST_METRICS = {"vllm:request_success_total": "served"}
RESET_ATTEMPTS = 10


# Agent sessions -------------------------------------------------------------


def _messages(row: dict) -> list[dict] | None:
    """The chat messages of a trace row, whatever the dataset calls them."""
    for key in ("messages", "trajectory", "conversation", "conversations"):
        value = row.get(key)
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                continue
        if isinstance(value, list) and value and isinstance(value[0], dict):
            messages = []
            for item in value:
                role = item.get("role") or item.get("from")
                content = item.get("content", item.get("text", item.get("value")))
                if role and isinstance(content, str):
                    role = {"human": "user", "gpt": "assistant", "ai": "assistant"}.get(
                        role, role
                    )
                    messages.append({"role": role, "content": content})
            return messages or None
    return None


def load_sessions(source: str, limit: int, seed: int) -> list[list[dict]]:
    """Agent sessions with at least two assistant turns.

    `source` is a JSONL file with one session per line, or a Hugging Face
    dataset name streamed from the hub.
    """
    path = Path(source)
    if path.exists():
        rows: Iterable[dict] = (
            json.loads(line) for line in path.open() if line.strip()
        )
    else:
        from datasets import load_dataset

        rows = load_dataset(source, split="train", streaming=True)
    sessions = []
    for row in rows:
        messages = _messages(row)
        if messages and sum(m["role"] == "assistant" for m in messages) >= 2:
            sessions.append(messages)
        if len(sessions) >= limit * 4:
            break
    random.Random(seed).shuffle(sessions)
    return sessions[:limit]


def turn_prompts(
    render: Callable[[list[dict]], list[int]], messages: list[dict]
) -> list[list[int]]:
    """The prompt the agent sent before each assistant turn, as token ids."""
    return [
        render(messages[:i])
        for i, m in enumerate(messages)
        if m["role"] == "assistant" and i
    ]


def pick_turn(
    prompts: list[list[int]], max_tokens: int
) -> tuple[list[int], list[int]] | None:
    """The last (previous, current) turn pair whose current prompt fits.

    Current usually extends previous. A template that rewrites old turns (one
    that drops past reasoning, say) shares less, and expected_hit counts only
    what is shared, so the pair still tests a partial hit.
    """
    for previous, current in reversed(list(pairwise(prompts))):
        if len(current) <= max_tokens and len(current) > len(previous):
            return previous, current
    return None


def chat_renderer(model: str) -> Callable[[list[dict]], list[int]]:
    """Token ids of a conversation, as the model's chat template renders it."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)

    def render(messages: list[dict]) -> list[int]:
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        return list(tokenizer(text, add_special_tokens=False)["input_ids"])

    return render


# Block hashes, as vLLM computes them ---------------------------------------


def block_hashes(prompt: list[int], block_size: int, hash_algo: str) -> list[str]:
    """Hex prefix-cache hashes of the prompt's full blocks.

    Uses vLLM's own code, so the router side of the RFC is exact. sha256 with
    no PYTHONHASHSEED uses vLLM's fixed seed, the same in every process.
    """
    from vllm.utils.hashing import get_hash_fn_by_name
    from vllm.v1.core import kv_cache_utils

    hash_fn = get_hash_fn_by_name(hash_algo)
    kv_cache_utils.init_none_hash(hash_fn)
    hashes, parent = [], None
    for start in range(0, len(prompt) - block_size + 1, block_size):
        parent = kv_cache_utils.hash_block_tokens(
            hash_fn, parent, prompt[start : start + block_size]
        )
        hashes.append(bytes(parent).hex())
    return hashes


def expected_hit(held: list[int], prompt: list[int], block_size: int) -> int:
    """Tokens of `prompt` B should pull: the full blocks it shares with what A
    holds, short of the last token, which B computes for its logits."""
    shared = 0
    for a, b in zip(held, prompt):
        if a != b:
            break
        shared += 1
    shared = min(shared, len(held), len(prompt) - 1)
    return shared // block_size * block_size


# Requests -------------------------------------------------------------------


def reset_cache(url: str, attempts: int = RESET_ATTEMPTS) -> bool:
    """Reset a server's prefix cache. vLLM refuses while any block is held,
    so retry briefly for transfers that are still finishing."""
    request = urllib.request.Request(
        f"{url}/reset_prefix_cache", data=b"", method="POST"
    )
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                if json.load(response).get("success", True):
                    return True
        except (OSError, ValueError):
            pass
        if attempt + 1 < attempts:
            time.sleep(1)
    return False


def greedy(servers: Servers, url: str, body: dict) -> dict:
    """TTFT and the first CHECK_TOKENS greedy tokens with their logprobs."""
    started = time.perf_counter()
    first, tokens, logprobs = None, [], []
    stream = {**body, "stream": True, "max_tokens": CHECK_TOKENS, "logprobs": 1,
              "ignore_eos": True}  # fmt: skip
    with kv.post(servers, url, stream) as response:
        for line in response:
            if not line.startswith(b"data:") or b"[DONE]" in line:
                continue
            first = first or time.perf_counter() - started
            choice = json.loads(line[5:])["choices"][0]
            lp = choice.get("logprobs") or {}
            tokens.extend(lp.get("tokens") or [choice.get("text", "")])
            logprobs.extend(lp.get("token_logprobs") or [])
    if first is None:
        raise RuntimeError("the stream ended without a token")
    return {"ttft_s": first, "total_s": time.perf_counter() - started,
            "tokens": tokens, "logprobs": logprobs}  # fmt: skip


def peer_params(producer: dict, hashes: list[str], identity: dict | None) -> dict:
    """kv_transfer_params for one request to B that asks A by block hash.

    One request to B, and B calls A on the NIXL side channel. B hashes the
    prompt itself; the hashes are sent as a router would send them. A's
    topology (tp, dcp, pp sizes) lets B heartbeat the pin and release it.
    """
    params = {
        "do_remote_prefill": True,
        "peer_lookup": True,
        "remote_block_hashes": hashes,
        "remote_engine_id": producer["remote_engine_id"],
        "remote_host": producer["remote_host"],
        "remote_port": producer["remote_port"],
    }
    for key in ("tp_size", "dcp_size", "pp_size"):
        if key in producer:
            params[key] = producer[key]
    if identity:
        params["kv_identity"] = identity
    return params


def served(url: str) -> float:
    return kv.metric_totals(url, REQUEST_METRICS)["served"]


def log_size(log: Path) -> int:
    return log.stat().st_size if log.exists() else 0


def log_since(log: Path, offset: int) -> str:
    if not log.exists():
        return ""
    with log.open("rb") as f:
        f.seek(offset)
        return f.read().decode(errors="replace")


# One case -------------------------------------------------------------------


def run_case(
    args: argparse.Namespace,
    servers: Servers,
    case: str,
    previous: list[int],
    current: list[int],
    producer: dict,
) -> dict:
    """All arms of one case for one turn, each on fresh prefix caches."""
    held = {"full": current, "partial": previous, "miss": []}[case]
    record: dict = {
        "case": case,
        "prompt_tokens": len(current),
        "held_tokens": len(held),
        "expected_hit_tokens": expected_hit(held, current, args.block_size),
    }

    def fresh() -> None:
        for url in (servers.producer, servers.consumer):
            if not reset_cache(url):
                raise RuntimeError(f"{url} refused to reset its prefix cache")
        if held:
            kv.timed_stream(servers, servers.producer, {"prompt": held})

    fresh()
    record["recompute"] = greedy(servers, servers.consumer, {"prompt": current})

    if case != "miss":
        fresh()
        params = kv.prefill(servers, current)
        before = kv.metric_totals(servers.consumer, kv.NIXL_METRICS)
        stock = greedy(
            servers, servers.consumer, {"prompt": current, "kv_transfer_params": params}
        )
        stock["transfer"] = kv.metrics_change(servers.consumer, before)
        stock["pulled_tokens"] = expected_hit(current, current, args.block_size)
        record["stock"] = stock

    if args.mode == "peer":
        fresh()
        hashes = block_hashes(current, args.block_size, args.hash_algo)
        time.sleep(kv.METRICS_WAIT_SECONDS)  # let the warm-up's count land
        a_served = served(servers.producer)
        a_log = log_size(args.producer_log)
        before = kv.metric_totals(servers.consumer, kv.NIXL_METRICS)
        body = {"prompt": current,
                "kv_transfer_params": peer_params(producer, hashes, args.identity)}  # fmt: skip
        peer = greedy(servers, servers.consumer, body)
        peer["transfer"] = kv.metrics_change(servers.consumer, before)
        time.sleep(kv.METRICS_WAIT_SECONDS)
        peer["a_served"] = served(servers.producer) - a_served
        peer["lookups_logged"] = len(
            re.findall(
                args.lookup_pattern, log_since(args.producer_log, a_log), re.IGNORECASE
            )
        )
        record["peer"] = peer
    return record


def matches(a: dict, b: dict) -> bool:
    return a["tokens"] == b["tokens"]


def bytes_per_token(records: list[dict]) -> float | None:
    """From the stock arm, where the pulled length is known."""
    rates = [
        r["stock"]["transfer"]["bytes"] / r["stock"]["pulled_tokens"]
        for r in records
        if "stock" in r
        and r["stock"]["transfer"]["bytes"] > 0
        and r["stock"]["pulled_tokens"]
    ]
    return statistics.median(rates) if rates else None


def check(records: list[dict], mode: str, leak_free: bool) -> dict:
    """Pass or fail per check, with the failing records' indexes."""
    failures: dict[str, list[int]] = {
        "served": [], "lookup": [], "pulled": [], "equivalent": []
    }  # fmt: skip
    rate = bytes_per_token(records)
    compared = equal = 0
    for i, r in enumerate(records):
        for arm in ("stock", "peer"):
            if arm in r:
                compared += 1
                equal += matches(r[arm], r["recompute"])
        if mode != "peer":
            continue
        peer = r["peer"]
        if peer["a_served"] != 0:
            failures["served"].append(i)
        if peer["lookups_logged"] == 0:
            failures["lookup"].append(i)
        pulled = peer["transfer"]["bytes"]
        expected = r["expected_hit_tokens"]
        if expected == 0:
            ok = pulled == 0
        elif rate is None:
            ok = pulled > 0
        else:
            ok = abs(pulled - expected * rate) <= BYTES_TOLERANCE * expected * rate
        if not ok:
            failures["pulled"].append(i)
    fraction = equal / compared if compared else 1.0
    result = {name: {"ok": not idx, "records": idx} for name, idx in failures.items()}
    if mode != "peer":
        for name in ("served", "lookup", "pulled"):
            result[name] = {"ok": None, "records": []}
    result["equivalent"] = {"ok": fraction >= MIN_EQUIVALENT_FRACTION,
                            "fraction": fraction, "compared": compared}  # fmt: skip
    result["no_leaks"] = {"ok": leak_free}
    result["bytes_per_token"] = rate
    return result


def latency(records: list[dict]) -> dict:
    """Median B TTFT per case and arm."""
    table: dict = {}
    for case in CASES:
        rows = [r for r in records if r["case"] == case]
        table[case] = {
            arm: statistics.median(r[arm]["ttft_s"] for r in rows if arm in r)
            for arm in ("recompute", "stock", "peer")
            if any(arm in r for r in rows)
        }
    return table


def pull_rates(records: list[dict]) -> dict:
    """Pull throughput per arm, from vLLM's NIXL metrics on B.

    NIXL stops the transfer clock when B polls it, so this is a lower bound on
    the link speed; compare it with probes.nixl_bandwidth.
    """
    table: dict = {}
    for arm in ("stock", "peer"):
        moved = [r[arm]["transfer"] for r in records
                 if arm in r and r[arm]["transfer"].get("xfer_s", 0) > 0]  # fmt: skip
        if moved:
            table[arm] = {
                "gbps_median": statistics.median(
                    m["bytes"] / m["xfer_s"] / 1e9 for m in moved
                ),
                "descriptors_median": statistics.median(
                    m["descriptors"] for m in moved
                ),
                "transfers": len(moved),
            }
    return table


def print_summary(checks: dict, table: dict) -> None:
    print("case     recompute_s  stock_s  peer_s")
    for case, arms in table.items():
        cells = [
            f"{arms[a]:.3f}" if a in arms else "-"
            for a in ("recompute", "stock", "peer")
        ]
        print(f"{case:<8} {cells[0]:>11}  {cells[1]:>7}  {cells[2]:>6}")
    for name, result in checks.items():
        if isinstance(result, dict):
            state = {True: "PASS", False: "FAIL", None: "skip"}[result["ok"]]
            print(f"{state}  {name}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--producer", required=True, help="A's base URL")
    parser.add_argument("--consumer", required=True, help="B's base URL")
    parser.add_argument("--producer-log", type=Path, required=True)
    parser.add_argument("--consumer-log", type=Path, required=True)
    parser.add_argument("--mode", choices=("stock", "peer"), default="peer")
    parser.add_argument("--traces", default=DEFAULT_DATASET,
                        help="JSONL file or Hugging Face dataset of agent sessions")  # fmt: skip
    parser.add_argument("--sessions", type=int, default=20)
    parser.add_argument("--max-prompt-tokens", type=int, default=30000)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--hash-algo", default="sha256")
    parser.add_argument("--identity", type=json.loads, default=None,
                        help="kv_identity JSON to send, if the branch needs it")  # fmt: skip
    parser.add_argument("--lookup-pattern", default=LOOKUP_PATTERN)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results.json"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    servers = Servers(args.model, args.producer, args.consumer)
    kv.wait_ready(servers.producer)
    kv.wait_ready(servers.consumer)

    render = chat_renderer(args.model)
    # A's side-channel address, from one stock prefill. B then pulls it, so A
    # frees those blocks now instead of holding them until the lease expires,
    # which would make the first resets fail.
    address_prompt = kv.fresh_prompt(random.Random(args.seed), 64)
    producer = kv.prefill(servers, address_prompt)
    kv.timed_stream(
        servers,
        servers.consumer,
        {"prompt": address_prompt, "kv_transfer_params": producer},
    )

    records: list[dict] = []
    for n, messages in enumerate(load_sessions(args.traces, args.sessions, args.seed)):
        pair = pick_turn(turn_prompts(render, messages), args.max_prompt_tokens)
        if pair is None:
            continue
        for case in CASES:
            try:
                record = run_case(args, servers, case, *pair, producer)
            except (OSError, RuntimeError, ValueError, KeyError) as exc:
                record = {"case": case, "error": repr(exc)}
            record["session"] = n
            records.append(record)
            print(json.dumps({k: v for k, v in record.items()
                              if k in ("session", "case", "prompt_tokens",
                                       "expected_hit_tokens", "error")}), flush=True)  # fmt: skip

    good = [r for r in records if "error" not in r]
    leak_free = reset_cache(servers.producer)
    checks = check(good, args.mode, leak_free)
    table = latency(good)
    rates = pull_rates(good)
    path = (kv.read_path(args.consumer_log.read_text(errors="replace"))
            if args.consumer_log.exists() else [])  # fmt: skip
    args.output.write_text(json.dumps(
        {"model": args.model, "mode": args.mode, "traces": args.traces,
         "checks": checks, "latency": table, "pull_rates": rates,
         "read_path": path, "records": records}, indent=2))  # fmt: skip
    for log in (args.producer_log, args.consumer_log):
        if log.exists():
            shutil.copy(log, args.output.parent / log.name)

    print_summary(checks, table)
    for arm, rate in rates.items():
        print(
            f"pull {arm}: median {rate['gbps_median']:.2f} GB/s over "
            f"{rate['descriptors_median']:.0f} descriptors ({rate['transfers']} transfers)"
        )
    print(
        "UCX read path:"
        + ("".join(f"\n  {row}" for row in path) if path else " (not logged)")
    )
    errors = len(records) - len(good)
    failed = [k for k, v in checks.items() if isinstance(v, dict) and v["ok"] is False]
    if errors or failed or not good:
        print(f"FAIL: {errors} errored records, failed checks: {failed}", flush=True)
        return 1
    print("PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
