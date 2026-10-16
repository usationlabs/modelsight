# modelsight (`ms`)

A small tool for running reproducible inference-server benchmarks. A YAML file describes an experiment. `ms` runs it, either on a local GPU machine or on [Modal](https://modal.com), and saves the logs, the exact commands and the benchmark output in one folder. The examples use [SGLang](https://github.com/sgl-project/sglang) as the server and [AIPerf](https://github.com/ai-dynamo/aiperf) as the load generator.

It is a thin wrapper, not a benchmarking framework. `server` and `bench` are plain command strings, copied straight from the engine's and the benchmark tool's docs. `ms` never parses or validates their flags, and it only needs the server to answer on `ready_url`, so another engine such as vLLM should work by changing `image`, `server` and `ready_url`. Only SGLang has been tried so far.

## Install

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo-url> && cd modelsight
uv sync
```

On a GPU box you can also install it as a tool: `uv tool install git+<repo-url>`.

AIPerf is not a dependency of `ms`. For `target: local`, `aiperf` must be on the PATH (`uv tool install aiperf`). `ms run` stops before launching anything if it can't find it, and `--dry-run` reports whether it was found. For `target: modal`, AIPerf is installed into its own environment in the container image, so it can't clash with the engine's dependencies.

## Usage

```bash
uv run ms run -f examples/qwen27b-fp8.yaml            # run the experiment
uv run ms run -f examples/qwen27b-fp8.yaml --dry-run  # print what would run, launch nothing
```

## The experiment file

```yaml
target: modal                   # modal | local
gpu: B200                       # modal only, e.g. B200, H100:2
image: lmsysorg/sglang:v0.5.21  # modal only, the container to run in
server: >-
  python -m sglang.launch_server
  --model-path Qwen/Qwen3.8-27B-FP8
  --tp-size 1
  --port 30000
bench: >-
  aiperf profile
  --model Qwen/Qwen3.8-27B-FP8
  --url http://localhost:30000
  --endpoint-type chat
  --streaming
  --concurrency 32
  --request-count 320
  --ui none
```

Optional fields:

| Field | Default | Meaning |
| --- | --- | --- |
| `ready_url` | `http://localhost:30000/health` | What `ms` polls to know the server is up. Change it if you change the port. |
| `ready_timeout` | `20m` | How long to wait for model download and load. |
| `timeout` | `2h` | Hard cap on the whole run, so a hang cannot burn credits. |
| `output` | `results/` | Where run folders are written. |

Durations accept forms like `90s`, `20m`, `2h` and `1h30m`.

**Request count.** AIPerf sends only 10 requests unless you set `--request-count`. That is fine for a smoke test, but with a concurrency above 10 the configured concurrency is never reached. Set it (a common rule of thumb is about 10x the concurrency) when you want numbers to compare.

## What a run does

1. Start the `server` command, locally or inside a Modal container.
2. Poll `ready_url` until it responds, or fail after `ready_timeout`.
3. Run the `bench` command with the run folder as its working directory.
4. Save the outputs and stop the server, even on failure, Ctrl-C or timeout.

Server and bench output is streamed to your terminal with `[server]` and `[bench]` prefixes, and also written to log files.

## What gets saved

```
results/<name>-<YYYYmmdd-HHMMSS>/
  exp.yaml          # the exact YAML used
  commands.txt      # the exact server and bench commands
  env.txt           # nvidia-smi output, image tag, ms and AIPerf versions
  server.log        # server stdout and stderr
  bench.log         # bench stdout and stderr
  artifacts/        # AIPerf's own output, untouched
```

## Targets

- **local** runs the commands as shell processes on this machine. The server runs in its own process group, so teardown stops it and everything it spawned.
- **modal** runs the same commands in one container with the GPU you ask for. The server and bench share the container, so AIPerf talks to `localhost`. Results are written to a Modal Volume (`modelsight-results`) and copied back when the run ends. If your laptop disconnects, the container still stops at `timeout`, and you can fetch results with `modal volume get modelsight-results <run-name> <dir>`.

Modal runs bill GPU time. Model weights are downloaded with `HF_TOKEN` from a Modal secret named `huggingface`:

```bash
modal secret create huggingface HF_TOKEN=$HF_TOKEN
```

Locally, `HF_TOKEN` is read from your environment or a `.env` file in the current directory. Secrets never go in the YAML.
