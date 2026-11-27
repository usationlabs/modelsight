# H6 handoff plan: peer-stage benchmark on two nodes over RDMA

For: a cloud agent working in `usationlabs/modelsight`. Self-contained: everything you need is here or in the files it names.

## 0. Context

vLLM is gaining "peer stage" (branch `feat/nixl-peer-stage` of `usationlabs/vllm`, stacked on vllm-project/vllm#68430):
- **The existing feature:** engine B asks engine A over NIXL's side channel to pin a prefix by block hash (`LOOKUP_PIN`). A pins its cached GPU run, replies with block ids, and B reads them over NIXL.
- **What peer stage adds:** when part of the prefix sits only in A's CPU offload tier, A copies it into fresh GPU blocks before it replies.
  - A trims back to the GPU part when the copy misses a deadline or fails.
  - A skips the copy when it estimates B would compute those tokens faster.

H6 measures whether that pays off under realistic load, with A and B on two nodes and NIXL reads over RDMA.

- **Design:** `handoff/peer-pin-cpu-stage-fallback.md`.
- **Benchmark design:** `handoff/peer-pin-cpu-stage-benchmark.md`.

Read both fully. This plan turns the benchmark design into work packages.

H6 extends H5:
- `examples/kv-transfer/h5.yaml`;
- `src/modelsight/probes/kv_peer_load.py`;
- its tests, `tests/test_kv_peer_load_probe.py`.

Read all three before writing code. They define the harness: sessions, AIPerf load levels, the settle gate, per-arm `cache_salt`, the checks, and the table layout.

## 1. Starting state

- **Branch:** `exp/h6-peer-stage-bench`, stacked on `exp/prefix-kv-transfer` (PR usationlabs/modelsight#8). Your PR targets `exp/prefix-kv-transfer`.
- **Already on the branch, from earlier local work (scratch, single node):**
  - `examples/kv-transfer/offload_load_latency.yaml` + `src/modelsight/probes/kv_offload_latency.py` + `offload_latency_hook/`: the M0 copy-latency probe.
  - `examples/kv-transfer/peer_stage_e2e.yaml`, `peer_stage_unit_tests.yaml`, `Dockerfile.peer_stage`, `src/modelsight/probes/kv_peer_stage.py`, `peer_stage_hook/`: peer-stage correctness on one host with 2 GPUs. **Reuse its log parsers and engine config.**
  - `examples/kv-transfer/prepare_local_vllm.sh`: packs the vLLM worktree files that differ from a base commit into a tarball the Dockerfile lays over a precompiled install. Usage: `prepare_local_vllm.sh <worktree> <base_sha> <archive>`. Base for peer stage: `3332daf8fe613c0e3d7c06f42b12e95eb450aa1e`. The tarball (`local_vllm_peer_stage.tar.gz`) is built on the user's machine and gitignored. You cannot build it, so H6 yamls must expect it, like `peer_stage_e2e.yaml` does.
  - `handoff/`: this plan and the two design docs. **Delete `handoff/` in your last commit before the PR is marked ready.**

## 2. Rules

- **No Modal or GPU runs.** You have no Modal credentials. Verify with unit tests, `uv run ms run -f <yaml> --dry-run`, and fakes. For every run you prepare, put the exact command in its yaml header (prepare, build, run), as the existing yamls do. The user runs them.
- **Tests:** no duplicated tests. Extend an existing test wherever one covers the area (parametrize, add asserts), and add a new test only where nothing covers it. Run `uv run pytest -q` before each commit. Everything that passed before must still pass, and note any pre-existing failure. One known one: `test_example_yaml_loads` reads an unrelated example.
- **Style:** match the repo. Short, plain comments. Yaml headers explain the experiment in a few sentences. Log lines are tagged (`[A]`, `[B]`, `[bench]`). Follow ruff, if the repo uses it (check `pyproject.toml`).
- **Commits:** small, one work package (or part of one) each, conventional prefixes as in the history (`feat(kv-transfer): …`, `fix(...)`, `test(...)`). Use the repository's configured author. Add no `Co-Authored-By` lines and no AI attribution in commits or the PR.
- **PR:** open it as a draft against `exp/prefix-kv-transfer` once WP1 passes its tests, and push as you go. The description says what each WP adds and lists the run commands. It stays a draft. Do not merge.
- **Don't touch:** the H5 files' behaviour (`h5.yaml`, `kv_peer_load.py` must still run as before), the single-node path of every existing yaml, or `main`.
- **When blocked** on something only a real run can answer (for example a Modal API detail you cannot confirm from the installed `modal` package), implement the most likely variant behind one small function. Mark it in the PR description under "Needs a real run to confirm", and continue.

## 3. Work packages

### WP1: multi-node experiments in modelsight

**Goal:** one yaml runs on N Modal containers with RDMA between them. Every rank runs the `server` script, and one rank also runs `bench`.

**Config** (`src/modelsight/config.py`, `Experiment`, `_KNOWN_FIELDS`, `to_mapping`, `experiment_from_mapping`):
- `nodes: int = 1`. Over 1 makes the run clustered. Validate 1 to 8.
- `rdma: bool = False`. Only valid with `nodes > 1`.
- `bench_rank: int | None = None`. Defaults to `nodes - 1`. Must satisfy `0 <= bench_rank < nodes`.
- `ready_url` may contain `{node0}` … `{nodeN}` placeholders. Add `ready_urls: tuple[str, ...] = ()` so the bench rank can wait on several servers, for example A on node 0 and B locally.
- Unknown-field rejection stays. Single-node yamls must load exactly as before; extend `tests/test_config.py`.

**Runtime environment** (set for both `server` and `bench` on every rank):
- `MS_RANK` (0-based) and `MS_NODES`.
- `MS_NODE_IPS`: comma-separated, index = rank.
- `MS_NODE_IP`: this rank's IP, and `MS_NODE0_IP` … `MS_NODE{N-1}_IP`.
- On a single node: `MS_RANK=0`, `MS_NODES=1`, `MS_NODE_IPS=127.0.0.1`. Existing yamls then see the variables and nothing else changes.

**Runner** (`src/modelsight/runner.py`):
- **New `execute_cluster_run(exp, run_dir, config_text, rank, ips, extra_env)`.** Every rank starts `server` with the env above, writing `run_dir/rank-<r>/server.log`. The tee keeps the existing `[server]` prefix.
- **Bench rank:**
  1. Wait for its `ready_url` and every `ready_urls` entry, with placeholders resolved, within `ready_timeout`.
  2. Run `bench`.
  3. Mark the run done on the control endpoint below.
  4. Stop its server.
- **Non-bench ranks:** keep the server running until the bench rank says done, or the bench rank has been unreachable for 120 s, or `timeout`. Then stop the server.
- **Control endpoint:** a tiny stdlib HTTP server (`modelsight/control.py`) on each rank, port `MS_CONTROL_PORT` (default 8399), on all interfaces.
  - `GET /status` returns `{"rank": r, "state": "starting|serving|running_bench|done|failed"}`.
  - `GET /logs/<name>?offset=<n>` returns the bytes of `run_dir/rank-<r>/<name>` from `offset`. That is how the probe on node 1 tails A's log on node 0.
  - Allow only names matching `^[A-Za-z0-9._-]+$` inside that rank's dir, with no path traversal. Add a test for this.
- **A server that exits early fails the run on its rank,** and the bench rank sees `failed` and stops, like `wait -n` in today's scripts.
- **Single-node runs keep using `execute_run`,** unchanged.
- **Tests:** `tests/test_runner.py` runs a 2-rank cluster on localhost. Use two threads or processes with `ips=["127.0.0.1","127.0.0.1"]` and distinct control ports, and fake `server`/`bench` shell commands. Assert:
  - the order: rank 0 serves until the bench is done, then exits;
  - log files per rank;
  - failure propagation in both directions;
  - the log endpoint with offsets.

**Modal target** (`src/modelsight/targets/modal.py`):
- When `exp.nodes > 1`, wrap the function with `modal.experimental.clustered(size=exp.nodes, rdma=exp.rdma)`. That signature is confirmed in modal 1.6.1: `clustered(size, broadcast=True, rdma=False, fabric_size=None)`.
- Inside the function, `modal.experimental.get_cluster_info()` gives the rank and the container IPs. Check the installed package for the exact attribute names (`rank`, `container_ips` or `container_ipv4_ips`) and wrap the access in one helper.
- Spawning: check from the installed package how a clustered function is called (once per cluster with `broadcast=True`). `detach`, `_record`, `attach` and `_download` must work for cluster runs, with results for all ranks in one run dir on the results volume. Each rank commits the volume on exit.
- GPU per container stays `exp.gpu`.
- **Tests:** extend `tests/test_modal_image.py` (or the test that covers `run`) with a fake `modal` module. Check that the decorator and arguments are chosen for `nodes: 2, rdma: true`, and that single-node runs are untouched.
- **Local target** (`targets/local.py`): `nodes > 1` raises a clear `RunError` ("multi-node runs need the modal target"). Add a test.

**CLI:** `ms run --dry-run` prints nodes, rdma and the bench rank. Extend `tests/test_cli.py`.

**Done when** the WP1 tests pass and existing yamls dry-run unchanged.

### WP2: RDMA image and the transport gate

**Image** (`examples/kv-transfer/Dockerfile.h6`): start from `Dockerfile.peer_stage` (the vLLM tarball over the precompiled base `3332daf8fe`), then:
- apt install `rdma-core ibverbs-providers libibverbs1 ibverbs-utils perftest infiniband-diags`;
- at build time, print `ucx_info -v` and `ucx_info -d | grep -E "Transport|Device"`, and `ibv_devinfo || true`, so the build log shows whether the UCX that NIXL loads has `rc`/`dc` (mlx5 or verbs). Find the UCX inside the `nixl` wheel: check `python -c "import nixl, os; print(os.path.dirname(nixl.__file__))"` and its bundled libs.
- If the wheel's UCX lacks IB transports, add a commented, ready-to-enable stage that builds UCX with verbs and points NIXL at it. Note this in the PR under "Needs a real run".

**Smoke yaml** (`examples/kv-transfer/rdma_smoke.yaml`): `nodes: 2`, `rdma: true`, `gpu: <see below>`, no vLLM.
- Rank 0 runs `ib_write_bw -d <dev> --report_gbits` as server, and `ucx_perftest` as server with `UCX_PROTO_INFO=y`.
- Rank 1 runs the clients against `MS_NODE0_IP` and prints the bandwidth and the protocol table.
- A small checker, `modelsight/probes/rdma_gate.py`, parses the output. It exits 0 only if a transport matching `rc_mlx5|dc_mlx5|rc_verbs|ud_mlx5|ud_verbs` was used, with bandwidth above a floor (`--min-gbps`, default 50). It must fail on `tcp`, `sysv`, `posix` or `cuda_ipc`.
- Test the parser with fixture text in `tests/test_rdma_gate.py`, including H5's real `cuda_ipc` table (in the H5 results, the `read_path` field quotes the UCX lines).

**GPU type:** Modal documents RDMA clusters for H100-class GPUs, and that A100 is supported is not confirmed.
- Default `gpu: H100` in H6 yamls, with a header note that M0 (the copy-latency baseline) was measured on A100.
- Put "A100 RDMA availability" under "Needs a real run".

**vLLM read-path gate:** reuse H5's `read_path` extraction (search `kv_peer_load.py` / `h4` helpers for how it reads `UCX_PROTO_INFO` lines from B's log). The H6 probe fails the run (`rdma_path` check) unless B's KV reads use an RDMA transport by the rules above.

**Done when** the parser tests pass, the smoke yaml dry-runs, and the Dockerfile builds locally as far as possible without a GPU (`docker build` if available; otherwise state it was not built).

### WP3: H6 engines (`examples/kv-transfer/h6.yaml`)

- **Shape:** `nodes: 2`, `rdma: true`, `gpu: H100`, `dockerfile: Dockerfile.h6`, `bench_rank: 1`, `secrets: [HF_TOKEN]`, `cache.prefetch: [Qwen/Qwen3-8B]`, a timeout of 4 h.
- **`server` script:** branch on `MS_RANK`:
  - **rank 0 → A:** `vllm serve Qwen/Qwen3-8B` with the H5 flags (`--max-model-len 32768 --gpu-memory-utilization 0.9 --prefix-caching-hash-algo sha256 --no-enable-extensible-kv-cache --max-num-seqs 16`), plus:
    - `VLLM_NIXL_SIDE_CHANNEL_HOST=$MS_NODE_IP` and `VLLM_NIXL_SIDE_CHANNEL_PORT=5600`;
    - `--port 8100`;
    - `VLLM_SERVER_DEV_MODE=1`;
    - `UCX_PROTO_INFO=y`;
    - `--kv-transfer-config` = MultiConnector with children **NixlConnector first, then OffloadingConnector** (order matters: a reversed order costs a step). The NIXL child gets `kv_connector_extra_config` `{"peer_lookup": true, "peer_stage": true, "peer_stage_deadline": ${PEER_STAGE_DEADLINE:-1.5}, "peer_stage_recompute_tokens_per_s": ${PEER_STAGE_RATE:-0}, "peer_stage_max_fraction": 0.10}`. The Offloading child gets a CPU tier big enough for every session prefix (`cpu_bytes_to_use`; copy the key names from `peer_stage_e2e.yaml`).
    - The logging config from `peer_stage_e2e.yaml`: DEBUG for the peer-pin, peer-stage, pull-scheduler and offloading-scheduler loggers.
  - **rank 1 → B:** the same model and flags, `--port 8200`, `VLLM_NIXL_SIDE_CHANNEL_HOST=$MS_NODE_IP`, port 5601, MultiConnector (NIXL `peer_lookup: true`, Offloading), and DEBUG logging.
  - Logs go to `$MS_RUN_DIR/rank-$MS_RANK/server-A.log` or `server-B.log` (the runner provides the run dir; add `MS_RUN_DIR` to the env in WP1 if needed) and are piped through `sed -u "s/^/[A] /"`.
- **Readiness:** `ready_url: http://localhost:8200/health` and `ready_urls: ["http://{node0}:8100/health"]`.
- **AIPerf** runs from the bench rank against `http://$MS_NODE0_IP:8100`, as in H5.
- **Variants:** sibling yamls that differ only in env or flags. Keep them tiny by sharing the server script: put it in `examples/kv-transfer/h6_server.sh` and call it with env.
  - `h6_smoke.yaml`: levels `0,32`, 2 sessions, all arms.
  - `h6_skip.yaml`: Part B, `PEER_STAGE_RATE` set to the rate the user fills in after Part A.
  - `h6_deadline.yaml`: Part D, run once per `PEER_STAGE_DEADLINE` in `0.5 1.0 1.5`.
- **Discovery:** B learns A's address from A's `GET /kv_event_sources` `peer_lookup` record (engine id, side-channel host, port). Assert in the probe that the host is A's node IP, not `localhost`. If the record carries `localhost`, fail with a message naming `VLLM_NIXL_SIDE_CHANNEL_HOST`.

**Done when** every H6 yaml dry-runs and `tests/test_config.py` loads them.

### WP4: H6 probe (`src/modelsight/probes/kv_peer_stage_load.py`)

Build it from `kv_peer_load.py`. Import from it, or factor shared code into a helper both use, without changing H5's behaviour. Add `tests/test_kv_peer_stage_load_probe.py` in the style of `tests/test_kv_peer_load_probe.py`, against a mock HTTP server and fixture logs.

**CLI:** H5's arguments, plus:
- `--producer-log` / `--consumer-log` accept a path or a `http://host:8399/logs/<name>` URL. Add a `LogSource` abstraction with `read_from(offset)` for both.
- `--part sweep|skip|deadline|impact`, `--cases gpu,partial,cpu,cpu_tail,miss` and `--arms recompute,stock,peer,peer_stage,on_a`.
- `--impact-interval 0.5` and `--impact-seconds 180`.

**Cases.** Build each level's case states while A is idle, before starting AIPerf for that level, as H5 does for its own cases:
- `gpu` and `partial`: as H5's `full` and `partial`.
- `cpu`:
  1. Send the prompt with that arm's `cache_salt` and `max_tokens=1`.
  2. Wait until A's offload store has the blocks. Poll A's `/metrics` `vllm:kv_offload_store_bytes` (or `_total`) until it stops growing for 2 s, as `kv_offload_latency.py` does.
  3. Then `POST http://A/reset_prefix_cache` with `reset_external=false`, which clears only the GPU prefix cache.

  After the reset, every `cpu` and `cpu_tail` state for the level has been prepared. The reset is all-or-nothing, so prepare all CPU states first, reset once, then warm the GPU states.
- `cpu_tail`: like `cpu`. After the reset, send the first half of the prompt (cut at a block boundary). That loads the head back onto GPU from CPU and leaves the tail CPU-only.
- `miss`: a fresh salt.
- **Drift:** background load can evict a prepared GPU head before measurement. Classify every record by what A actually did, from A's log lines for that request's pin (`location_actual`, `gpu_tokens`, `cpu_tokens`), not by the intended case. Count drift per case, as H5 counts `evicted`.

**Arms.** Each uses its own `cache_salt`, as in H5.
- `recompute`, `stock`: as in H5.
- `peer`: lookup with staging opted out per request. Send `kv_transfer_params` with `"peer_stage": false` in addition to H5's peer params. B then sends `reply_budget_s = 0`, and A skips staging and replies with its GPU part. This per-request opt-out is being added on the vLLM side. If a run shows staging happening on the `peer` arm, the `peer_arm_unstaged` check must fail.
- `peer_stage`: the same lookup with staging allowed (no opt-out).
- `on_a`: the request sent straight to A (`http://$MS_NODE0_IP:8100`), so its TTFT includes A's queue.

**A's log lines to parse.** Reuse the parsers in `kv_peer_stage.py`:
- `Peer stage submitted: pin=%s staged=%d in_place=%d deadline_s=%.3f`
- `Peer stage done: pin=%s ok=%s seconds=%.3f`
- `Peer stage trimmed: pin=%s reason=%s kept_blocks=%d`
- `Peer stage skipped: pin=%s reason=%s est_s=%.3f`
- `Peer lookup_pin pinned %d of %d blocks as %s`

B's lines: `Peer lookup_pin hit for %s: %d tokens under pin %s` and `Peer lookup for %s missed: %s`. Copy the exact regexes from `kv_peer_stage.py` and keep them in one place.

**Record:** H5's record plus the fields in the benchmark design's "Result format → Records" table:
- per record: `location_actual`, `gpu_tokens`, `cpu_tokens`, `a_step_ms`;
- `peer_stage` arm: `outcome`, `est_stage_s`, `stage_s`;
- `peer` and `peer_stage` arms: `reply_s`, `fence_wait_s`;
- `peer_stage` arm: `kept_tokens`;
- `peer`, `peer_stage` and `stock` arms: `pull_s`, `b_compute_s`;
- per level: `peer_held_peak`.

Where to get each:
- `reply_s`: B's log, lookup submit to reply. Use the timestamps on B's debug lines. If B logs no submit line, use the request start as an upper bound and name the field `reply_upper_s`.
- `fence_wait_s`: A's log, pin to ship.
- `a_step_ms`: the median gap between A's engine-step log lines over the 10 s before the request. If A logs nothing per step at INFO, use A's `/metrics` iteration counters. Pick what exists and say which.
- `pull_s`: H5's `transfer.xfer_s`.
- `peer_held_peak`: sampled from A's metrics or logs. If no metric exposes pinned plus draining blocks, record `null`, list it under "Needs a real run", and do not invent one.

Write `records.jsonl` (one line per request, written as you go) and `results.json` (summaries, checks, verdicts).

**Tables** (printed and in `results.json`), exactly as the benchmark design describes:
1. A reply time;
2. B TTFT, the headline table;
3. Impact on A, from Part C;
4. Deadline, from Part D.

**Verdicts:**
- `break-even: …` per level, computed from Table 2 rows where `location_actual` is `cpu` or `cpu_tail`.
- Offline skip-rule scoring: `est_stage_s` against measured `peer_stage` vs `peer` TTFT, using the rate in `--score-rate` (default 10000 tok/s). Report "right on K of M; would skip W winners, stage L losers".
- `deadline …` and `impact on A …`, for Parts D and C.

**Parts:**
- `sweep`: all levels × cases × arms.
- `skip`: one level (`--levels 32`), with A started with the rule on (`h6_skip.yaml`). It scores live decisions (`skipped` vs `staged`) against what the `peer` arm and an unskipped estimate say.
- `deadline`: level 64, the `cpu` case, `peer_stage` and `peer` arms only. Run once per `h6_deadline.yaml` run, then merge with a tiny `--merge results-*.json` mode that builds Table 4.
- `impact`: at levels 16 and 64, two 3-minute windows.
  - In each window, B sends `cpu`-case lookups every `--impact-interval` s, with `peer_stage: false` in window 1 and staging allowed in window 2.
  - AIPerf's per-request TTFT and ITL for A in each window come from the AIPerf export, as in H5.
  - Also record A's lowest free-block count and preemptions per window, from metrics.

**Checks** (PASS/FAIL, as in H5):
- H5's `served`, `lookup`, `pulled`, `equivalent` (with near-ties scored as H5 does) and `no_leaks`;
- `rdma_path`;
- `staged_equivalent`: `peer_stage` output equals `recompute`, on every record where `outcome=staged`;
- `peer_arm_unstaged`;
- `no_draining_left`: after each level, A's prefix-cache reset succeeds once the level drains.

**Done when** the probe tests pass:
- table and verdict maths on fixture records;
- parsing of every log line above;
- `LogSource` over HTTP with offsets against the WP1 control server;
- the case-preparation order (all CPU states, one reset, then GPU states), checked with a mock A that records calls;
- drift classification.

All yamls dry-run.

### WP5: PR wrap-up

- The PR description:
  - one paragraph per WP;
  - the run order for the user, each command with its expected wall time:
    1. `prepare_local_vllm.sh` on their machine;
    2. `ms build -f h6.yaml`;
    3. `rdma_smoke.yaml`;
    4. `h6_smoke.yaml`;
    5. `h6.yaml`;
    6. `h6_skip.yaml`;
    7. three runs of `h6_deadline.yaml`;
  - "Needs a real run to confirm" (every assumption you could not check);
  - the test counts before and after.
- Delete `handoff/`.
- Leave the PR as a draft.

## 4. Order and checkpoints

1. WP1, then commit, push and open the draft PR.
2. WP2.
3. WP3 and WP4. WP4 can start while WP3's yamls are finished.
4. WP5.

After each WP, run the whole test suite and push. If one WP takes much longer than the others, push what passes and note what is left in the PR description rather than stopping.

## 5. Facts you may need

- **Modal clustered API** (modal 1.6.1 installed in this repo's env): `modal.experimental.clustered(size: int, broadcast: bool = True, rdma: bool = False, fabric_size: int | None = None)`, `modal.experimental.get_cluster_info()` and `modal.experimental.ClusterInfo`.
- **vLLM endpoints used:**
  - `POST /reset_prefix_cache?reset_external=false` (dev mode, GPU only; fails while requests run);
  - `GET /kv_event_sources` (the `peer_lookup` record);
  - `GET /metrics`.
- **vLLM env:** `VLLM_NIXL_SIDE_CHANNEL_HOST` (default `localhost`) and `VLLM_NIXL_SIDE_CHANNEL_PORT`.
- **H5 run 2 reference** (NVLink, an older build): `peer` at 0.23 to 0.29 s from level 16 to 128, `recompute` about 2.1 s, `stock` from 0.13 to 35 s. H6's `gpu` rows should repeat the shape. Don't hardcode these anywhere.
- **M0 reference** (A100, single node): copy about 24 GB/s; a staged load takes about two engine steps on a busy A (0.7 to 1.1 s), and 0.03 to 0.2 s idle.
