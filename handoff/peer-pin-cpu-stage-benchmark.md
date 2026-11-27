# Peer stage benchmark (H6): CPU staging under load, over RDMA

Status: plan, 2026-11-27. Measures the feature in [peer-pin-cpu-stage-fallback.md](peer-pin-cpu-stage-fallback.md). Extends H5 (`modelsight/examples/kv-transfer/h5.yaml`, probe `kv_peer_load.py`, run 2 = `h5-20261109-130046`).

## Questions

1. When A holds part of a prefix only on CPU, does staging beat what B does today, which is to take the GPU part and compute the rest? Answer it per prefix size and per A load level.
2. Where is the break-even, and does the skip rule pick the right side of it?
3. How often does the deadline trim, and what does a trim cost B?
4. What does staging for peers cost A's own requests?

## Setup

- **Two nodes, RDMA.** A on node 0 and B on node 1, one GPU each, so NIXL reads go over InfiniBand/RoCE (UCX `rc`/`dc`), not `cuda_ipc`. H5 and every earlier run read over NVLink inside one host, so their pull times do not carry over.
- **Hardware:** Modal `@modal.experimental.clustered(size=2, rdma=True)`. Modal documents RDMA clusters for H100-class GPUs, so this is likely 2 × H100 rather than A100. That needs confirming. M0's step times were measured on A100, so H6 records its own.
- **Model:** Qwen/Qwen3-8B, as in H5.
- **Engines:** both run `MultiConnector(NixlConnector, OffloadingConnector)` with `peer_lookup` on. A also runs `peer_stage`, with the GPU pool at the default size (realistic eviction) and a CPU tier big enough that nothing is evicted from CPU during a level.
- **Gate:** the run fails at startup unless `UCX_PROTO_INFO` shows B's reads into GPU taking an RDMA transport. H5 printed this table (`read_path`) and it said `cuda_ipc`.
- **Build:** feat/nixl-peer-stage, packed over upstream 3332daf8fe like `peer_stage_e2e.yaml`.

## What is varied

| Variable | Values | Why |
|---|---|---|
| A load (AIPerf agent sessions) | 0, 8, 16, 32, 64, 128 (H5's levels) | M0: staging time is about two A steps, so A's step time drives everything |
| Prefix location on A | `gpu`, `cpu`, `cpu_tail` (+ H5's `partial` and `miss`) | the feature only acts on CPU blocks |
| Prefix size | H5's 10 sessions (3.6k to 29k tokens) | spans the 10k to 12k break-even M0 suggests |
| Staging | off (`peer`) / on (`peer_stage`) | the comparison itself |

Fixed:
- B is idle (as in H5);
- `reply_budget_s` = 2 s, `peer_stage_deadline` = 1.5 s, `peer_stage_max_fraction` = 0.10.

The skip rule is off during the sweep (`peer_stage_recompute_tokens_per_s` = 0), so every CPU case is staged and its true cost measured. Each record carries `est_stage_s`, and the rule is scored offline against the measured outcome (Part A). One confirming level runs with the rule on (Part B).

### Prefix locations

| Case | A's state before the request | Built by |
|---|---|---|
| `gpu` (= H5 `full`) | whole prefix on GPU | warm it |
| `partial` | GPU, minus a short tail | H5's method |
| `cpu` | whole prefix only on CPU | warm it, then GPU-only `/reset_prefix_cache` (`reset_connector=false`); the CPU tier survives |
| `cpu_tail` | GPU head, CPU-only tail | `cpu`, then warm the first half again, which loads the head back onto GPU |
| `miss` | nowhere | fresh `cache_salt` |

A cannot reset under load. Each level therefore builds every case's state while idle, starts AIPerf, waits until A runs about min(N, 16) requests (H5's gate), and then measures. Background traffic can still evict a prepared GPU head. So each record is classified by what A actually pinned and staged (from A's `Peer stage submitted` and `Peer lookup_pin pinned` lines), not by the case it was meant to be. Drifted records are reported, as H5 reports `evicted`.

## Arms

| Arm | What happens | Meaning |
|---|---|---|
| `recompute` | B computes the whole prompt | upper bound |
| `stock` | A prefills, B pulls (H5). With offloading, A loads its own CPU hit | today's P/D path |
| `peer` | lookup with staging off: A pins the GPU part, B pulls it and computes the rest | **the baseline staging must beat** |
| `peer_stage` | lookup with staging on | the feature |
| `on_a` | A serves the request itself (queued behind its load) | what the router avoids by picking B |

Every arm uses its own `cache_salt`, as in H5, so arms don't share cache state.

## Parts

- **A. Sweep (main table):** all levels × cases × arms, 10 sessions, skip rule off. About 6 × 5 × 5 × 10 = 1,500 requests. H5 run 2 took about 6 min per level.
- **B. Skip rule check:** one busy level (32) with the rule on at a rate from Part A. Reports whether skipped requests would have lost and staged ones won.
- **C. Impact on A:** at levels 16 and 64, B sends a steady stream of `cpu` lookups (one every 0.5 s) for 3 min. It is run twice, staging on and off, and compares AIPerf's TTFT and ITL on A, A's lowest free-block count, and preemptions.
- **D. Deadline:** at level 64, `cpu` lookups with `peer_stage_deadline` at 0.5, 1.0 and 1.5 s. It reports the trim rate and B's TTFT on trimmed requests (the cost of the GPU part plus recompute after the wait).

## Result format

### Records (`records.jsonl`, one line per request)

H5's record (`level`, `case`, `prompt_tokens`, `held_tokens`, `expected_hit_tokens`, `session`, and per arm `ttft_s`, `total_s`, `tokens`, `logprobs`, `a_load`, `transfer`, `request_id`, `a_served`), with these additions:

| Field | Where | Meaning |
|---|---|---|
| `location_actual` | record | `gpu` / `cpu` / `cpu_tail` / `partial` / `miss`, from A's log |
| `gpu_tokens`, `cpu_tokens` | record | the split A actually had |
| `a_step_ms` | record | A's median step interval over the 10 s before the request |
| `outcome` | `peer_stage` arm | `staged`, `trimmed_deadline`, `trimmed_failed`, `skipped`, `no_cpu` or `miss` |
| `est_stage_s`, `stage_s` | `peer_stage` arm | A's estimate and the measured submit-to-done time |
| `reply_s` | `peer`, `peer_stage` | B sending the lookup to B getting the reply |
| `fence_wait_s` | `peer`, `peer_stage` | from A's log: pin to fence passed |
| `kept_tokens` | `peer_stage` | tokens in the reply (after any trim) |
| `pull_s`, `b_compute_s` | `peer`, `peer_stage`, `stock` | NIXL read time, then B's compute of the rest |
| `peer_held_peak` | level | peak of pinned plus draining blocks on A |

### Tables (printed, and in `results.json`)

1. **A reply time:** rows = `cpu_tokens` bucket × level. Columns: `reply_s` p50/p99, `stage_s` p50/p99, `fence_wait_s` p50, outcome counts.
2. **B TTFT (headline):** rows = `location_actual` × prefix-size bucket × level. Columns: TTFT p50/p99 of `recompute`, `stock`, `peer`, `peer_stage`, `on_a`; then `peer_stage` win rate against `peer` and against the best other arm, and its median saving in seconds.
3. **Impact on A (Part C):** rows = level × staging on/off. Columns: A's TTFT p50/p99, ITL p50/p99, lowest free blocks, preemptions.
4. **Deadline (Part D):** rows = deadline. Columns: trim rate, B TTFT p50 for staged and for trimmed requests.

### Verdict lines

- `break-even: peer_stage beats peer above N cpu tokens at A step = X ms` (one line per level).
- `skip rule (offline): right on K of M; would skip W winners, stage L losers`.
- `skip rule (live, level 32): right on K of M`.
- `deadline 1.5 s: trims P% at level 64; trimmed TTFT +Y s vs peer`.
- `impact on A: TTFT p99 +Z% with staging at level 64`.
- PASS/FAIL checks: H5's (`served`, `lookup`, `pulled`, `equivalent`, `no_leaks`), plus `rdma_path`, `staged_equivalent` (`peer_stage` output equals recompute), and `no_draining_left` after each level.

## Reused from H5 run 2

- **Harness:** the sessions, the levels, the AIPerf start/stop, the settle gate, per-arm salts, the checks and the table layout. The probe becomes `kv_peer_stage_load.py`, built from `kv_peer_load.py`.
- **Reference numbers, NVLink and an older build, not mixed into H6's tables:** `peer` stays at 0.23 to 0.29 s from level 16 to 128 (`full`), `recompute` about 2.1 s, and `stock` grows from 0.13 s to 35 s. H6's `gpu` rows should repeat that shape. If the pull is clearly slower over RDMA, that is the transport, and Table 2 shows it.
- **The ~0.25 s `peer` time under load** is an early check on the fence-plus-reply margin (0.2 s). Part A measures it properly.

## Work items

1. **modelsight multi-node target:** run a yaml on `clustered(size=2, rdma=True)`. Rank 0 runs A, rank 1 runs B and the probe. A's side-channel host is its cluster address, and both servers' logs come back tagged as today.
2. **Image with RDMA userspace:** rdma-core and a UCX with IB verbs (check that the NIXL wheel's UCX has them), with `UCX_TLS` left to choose `rc`/`dc`. The `rdma_path` gate catches a silent TCP fallback.
3. **Probe:** H5's probe plus the cases, arms, record fields, Parts B–D and the verdict lines above.
4. **Smoke run** at levels 0 and 32 with 2 sessions before the full sweep.

## Open

1. Whether Modal RDMA clusters offer A100. If not, H100 changes step times, so M0 should rerun on the same hardware, in Part A's idle level.
2. Part C's lookup rate (one every 0.5 s) is a guess. Use the router's peer-fetch rate from `peer_fetch_active` if it is higher.
