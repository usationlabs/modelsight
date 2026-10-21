# DeepInfer on Modal

DeepSeek's [DeepInfer](https://github.com/deepseek-ai/DeepInfer) served through Dynamo on 8 B200 GPUs. The image is the `Dockerfile` in this folder, built by Modal and cached.

| File | What it is |
| --- | --- |
| `Dockerfile` | CUDA 13.2.1 on Ubuntu 24.04, DeepInfer at a pinned commit, Dynamo's Python bindings from a branch. |
| `b200x8-1m.yaml` | Validation: chat, a cancelled stream, a ~1M-token prompt, a light AIPerf run. |
| `b200x8-throughput.yaml` | Throughput: an AIPerf sweep over concurrency 16 to 128. |

Run from this folder, with `HF_TOKEN` exported (or in a `.env` here):

```bash
ms build -f b200x8-1m.yaml     # builds the image and downloads the weights, no GPU billed
ms run   -f b200x8-1m.yaml     # the validation run
ms run   -f b200x8-throughput.yaml
```

Both experiments share one image, so the second does not rebuild it.

## The throughput sweep

`b200x8-throughput.yaml` runs one AIPerf sweep (`--concurrency 16,32,64,128`, 300 seconds at each level, 8,192 input and 1,024 output tokens per request). AIPerf writes one `concurrency_N` folder per level under `aiperf/` in the results, and its own `aggregate` summary. Edit `ISL`, `OSL`, `CONCURRENCIES` and `DURATION` at the top of the bench script to change the load.

Concurrency here is the total across all 8 GPUs, and this backend serves a single attention DP rank. For a rough point of comparison, DeepSeek's decode-only chart (`doc/images/wide_ep_decode_pareto.png` in the DeepInfer repo; context 131,072; values read by eye) shows EP8DP8 at about 12,700, 19,200 and 24,900 output tokens/s/GPU with 64, 128 and 256 requests per rank. That chart uses 8 DP ranks, so it is not the same configuration, and it measures decode alone rather than a full serving stack.

## Not yet verified

- Neither experiment has run on Modal or a GPU.
- The bench scripts were run against a fake local OpenAI server, not against DeepInfer.
- AIPerf is installed unpinned in the image. The flags used here were checked against AIPerf 0.13.0, so a later release could change them.
