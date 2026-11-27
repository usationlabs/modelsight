"""Peer lookup capability probe, end to end: does B leave an old engine alone?

SCRATCH, not committed. Run as the bench of
examples/kv-transfer/peer_lookup_compat.yaml, against four engines:

  OLD   stock upstream main (no peer lookup): its side-channel listener dies on
        a LOOKUP_PIN frame, so B must never send one
  NEWD  the review-fix build with peer_lookup off: advertises False
  NEWP  the review-fix build with peer_lookup on: advertises True
  CONS  the review-fix build, the consumer (B)

Each producer is warmed with a prompt and B gets that prompt with peer_lookup
and the producer's address, as the router would send it.

Checks, which fail the run:

  old_and_unadvertised   For OLD and NEWD: B's output equals its own recompute,
                         no NIXL bytes moved, B's log says the lookup missed as
                         `peer_unsupported` (not identity_mismatch, which a
                         lookup would have drawn, so none was sent), and the
                         producer's listener still answers GET_META (and its
                         log shows no thread exception or traceback) and its
                         payload does not advertise peer_lookup
  new_hit                For NEWP: B logs a lookup hit, NIXL bytes moved, output
                         equals recompute, the producer logged the lookup, and
                         its GET_META payload advertises peer_lookup
  no_leaks               NEWP can reset its prefix cache at the end (no pin left)

Informational: with --control, a raw LOOKUP_PIN frame is sent to OLD last, and
the check is whether its listener then stops answering GET_META (the hazard the
probe exists for).

Writes results.json to the current directory.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
import time
from pathlib import Path

import msgspec
import zmq

from modelsight.probes import kv_peer_lookup as pl
from modelsight.probes import kv_transfer as kv
from modelsight.probes.kv_transfer import Servers

PROMPT_TOKENS = 1024
GET_META = b"get_meta_msg"
LOOKUP_PIN = b"lookup_pin"


def get_meta(host: str, port: int, timeout_ms: int = 3000) -> dict | None:
    """The handshake payload's top-level fields, or None if nothing answers.

    This is the frame every vLLM version answers, so it also tells whether an
    engine's listener is still alive.
    """
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    try:
        sock.connect(f"tcp://{host}:{port}")
        sock.send(msgspec.msgpack.encode((GET_META, 0, 0)))
        if not sock.poll(timeout_ms):
            return None
        frames = sock.recv_multipart()
        payload = msgspec.msgpack.decode(frames[0])
        if isinstance(payload, dict):
            return {k: v for k, v in payload.items() if k != "agent_metadata_bytes"}
        return {"raw": repr(payload)[:80]}
    finally:
        sock.close(linger=0)
        ctx.destroy(linger=0)


def raw_lookup(host: str, port: int) -> None:
    """A LOOKUP_PIN-shaped 2-tuple frame, as an unprobed consumer would send."""
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    try:
        sock.connect(f"tcp://{host}:{port}")
        sock.send(msgspec.msgpack.encode((LOOKUP_PIN, b"\x80")))
        sock.poll(1500)
    finally:
        sock.close(linger=0)
        ctx.destroy(linger=0)


def attempt(
    args: argparse.Namespace,
    name: str,
    producer_url: str,
    producer_log: Path,
    seed: int,
) -> dict:
    """One lookup attempt from B against `producer_url`, with its evidence."""
    servers = Servers(args.model, producer_url, args.consumer)
    rng = random.Random(seed)
    # The producer's address, from one stock prefill that B pulls straight away,
    # so the producer frees those blocks instead of holding them for the lease.
    address_prompt = kv.fresh_prompt(rng, 64)
    addr = kv.prefill(servers, address_prompt)
    kv.timed_stream(
        servers, args.consumer, {"prompt": address_prompt, "kv_transfer_params": addr}
    )

    prompt = kv.fresh_prompt(rng, args.prompt_tokens)
    for url in (producer_url, args.consumer):
        pl.reset_cache(url)
    kv.timed_stream(servers, producer_url, {"prompt": prompt})  # A now holds it
    reference = pl.greedy(servers, args.consumer, {"prompt": prompt})
    pl.reset_cache(args.consumer)  # no local hit, so B has to look elsewhere

    b_log, a_log = pl.log_size(args.consumer_log), pl.log_size(producer_log)
    before = kv.metric_totals(args.consumer, kv.NIXL_METRICS)
    got = pl.greedy(
        servers,
        args.consumer,
        {"prompt": prompt, "kv_transfer_params": pl.peer_params(addr, None)},
    )
    moved = kv.metrics_change(args.consumer, before)
    time.sleep(kv.METRICS_WAIT_SECONDS)
    b_text = pl.log_since(args.consumer_log, b_log)
    a_text = pl.log_since(producer_log, a_log)
    meta = get_meta(addr["remote_host"], addr["remote_port"])
    return {
        "producer": name,
        "output_equals_recompute": got["tokens"] == reference["tokens"]
        or pl.is_near_tie(got, reference),
        "nixl_bytes": moved["bytes"],
        "unsupported_logged": bool(
            re.search(r"Peer lookup for \S+ missed: peer_unsupported", b_text)
        ),
        "identity_mismatch_logged": "missed: identity_mismatch" in b_text,
        "hit_logged": "Peer lookup_pin hit for" in b_text,
        "producer_logged_lookup": len(
            re.findall(r"lookup[_ -]?pin", a_text, re.IGNORECASE)
        ),
        "producer_errors": len(re.findall(r"Traceback|Error", a_text)),
        # The old listener dies on a LOOKUP_PIN frame with an unhandled thread
        # exception, which Python prints as "Exception in thread".
        "producer_crashed": bool(
            re.search(r"Exception in thread|Traceback \(most recent call last\)", a_text)
        ),
        "get_meta_after": meta,
        "listener_alive": meta is not None,
        "advertises_peer_lookup": bool(meta and meta.get("peer_lookup")),
    }


def judge(records: dict[str, dict], leak_free: bool) -> dict:
    checks: dict = {}
    for name in ("old", "unadvertised"):
        r = records[name]
        ok = (
            r["output_equals_recompute"]
            and r["nixl_bytes"] == 0
            and r["unsupported_logged"]
            and not r["identity_mismatch_logged"]
            and not r["hit_logged"]
            and r["producer_logged_lookup"] == 0
            and r["listener_alive"]
            and not r["advertises_peer_lookup"]
            and not r["producer_crashed"]
        )
        checks[f"{name}_gets_no_lookup_and_keeps_serving"] = {"ok": ok, **r}
    r = records["new"]
    checks["new_hit"] = {
        "ok": r["output_equals_recompute"]
        and r["nixl_bytes"] > 0
        and r["hit_logged"]
        and r["producer_logged_lookup"] > 0
        and r["advertises_peer_lookup"],
        **r,
    }
    checks["no_leaks"] = {"ok": leak_free}
    return checks


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True)
    p.add_argument("--old", required=True)
    p.add_argument("--unadvertised", required=True)
    p.add_argument("--new", required=True)
    p.add_argument("--consumer", required=True)
    p.add_argument("--old-log", type=Path, required=True)
    p.add_argument("--unadvertised-log", type=Path, required=True)
    p.add_argument("--new-log", type=Path, required=True)
    p.add_argument("--consumer-log", type=Path, required=True)
    p.add_argument("--prompt-tokens", type=int, default=PROMPT_TOKENS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--control", action="store_true")
    p.add_argument("--output", type=Path, default=Path("results.json"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for url in (args.old, args.unadvertised, args.new, args.consumer):
        kv.wait_ready(url)

    records: dict[str, dict] = {}
    plan = (
        ("old", args.old, args.old_log),
        ("unadvertised", args.unadvertised, args.unadvertised_log),
        ("new", args.new, args.new_log),
    )
    for n, (name, url, log) in enumerate(plan):
        try:
            records[name] = attempt(args, name, url, log, args.seed + n)
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            records[name] = {"producer": name, "error": repr(exc)}
        print(json.dumps(records[name], default=str), flush=True)
    if any("error" in r for r in records.values()):
        args.output.write_text(json.dumps({"records": records}, indent=2, default=str))
        print("FAIL: an attempt errored", flush=True)
        return 1

    leak_free = pl.reset_cache(args.new)
    checks = judge(records, leak_free)

    control = None
    if args.control:
        host = "127.0.0.1"

        addr = Servers(args.model, args.old, args.consumer)
        side_port = kv.prefill(addr, kv.fresh_prompt(random.Random(99), 64))["remote_port"]
        before = get_meta(host, side_port)
        raw_lookup(host, side_port)
        time.sleep(1)
        after = get_meta(host, side_port)
        control = {
            "answered_before": before is not None,
            "answered_after_raw_lookup": after is not None,
            "hazard_reproduced": before is not None and after is None,
        }
        print("control (informational):", json.dumps(control), flush=True)

    args.output.write_text(
        json.dumps({"model": args.model, "checks": checks, "control": control}, indent=2, default=str)
    )
    for log in (args.old_log, args.unadvertised_log, args.new_log, args.consumer_log):
        if log.exists():
            shutil.copy(log, args.output.parent / log.name)
    failed = [k for k, v in checks.items() if not v["ok"]]
    for k, v in checks.items():
        print(f"{'ok  ' if v['ok'] else 'FAIL'} {k}")
    if failed:
        print(f"FAIL: {failed}", flush=True)
        return 1
    print("PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
