# DeepInfer on Modal

DeepSeek's [DeepInfer](https://github.com/deepseek-ai/DeepInfer) served through Dynamo on 8 B200 GPUs. The image is the `Dockerfile` in this folder, built by Modal and cached.

| File | What it is |
| --- | --- |
| `Dockerfile` | CUDA 13.2.1 on Ubuntu 24.04, DeepInfer at a pinned commit, Dynamo's Python bindings from a branch. |
| `b200x8-1m.yaml` | Validation: chat, a cancelled stream, a ~1M-token prompt, a light AIPerf run. |
| `b200x8-throughput.yaml` | Throughput: an AIPerf concurrency sweep, to be priced at DeepSeek's API rates. |
| `revenue.py` | Turns the sweep's AIPerf results into dollars per GPU-hour. Standard library only. |

Run from this folder, with `HF_TOKEN` exported (or in a `.env` here):

```bash
ms build -f b200x8-1m.yaml     # builds the image and downloads the weights, no GPU billed
ms run   -f b200x8-1m.yaml     # the validation run
ms run   -f b200x8-throughput.yaml
python revenue.py --gpus 8 results/<run folder>/aiperf
```

Both experiments share one image, so the second does not rebuild it.

## About the US$38 per GPU-hour

DeepSeek's Q&A (`doc/inference_deployment_qa.md` in the DeepInfer repo) says one production cluster serves 155 trillion tokens a day with "overall revenue of US$38 per GPU-hour (at valley pricing)". "Valley" is the off-peak rate on [DeepSeek's pricing page](https://api-docs.deepseek.com/quick_start/pricing) (read 2026-10-21; check it before relying on it). Per 1M tokens for `deepseek-flash`, which is served by DeepSeek-V4.1-Flash-1017:

| | Off-peak | Peak |
| --- | ---: | ---: |
| Input, cache hit | $0.003 | $0.006 |
| Input, cache miss | $0.15 | $0.30 |
| Output | $0.60 | $1.20 |

**This setup cannot reproduce that number.** The cluster behind it differs in ways that matter:

- It is prefill/decode disaggregated, with decode at EP64 to EP128 and DP64 to DP128 and prefill on separate GPUs. Here one aggregated deployment does both, on 8 GPUs with one attention DP rank.
- Most of its input tokens are probably cache hits at $0.003 per million, served from a KV-cache store with days of retention. DeepInfer's README says that store, disaggregation and DP load balancing are not in the open-source release.
- Its traffic mix is real production traffic, and the figure averages prefill and decode GPUs.

For scale, DeepSeek's own decode-only chart (`doc/images/wide_ep_decode_pareto.png`, context 131,072, approximate values read by eye) shows EP8DP8 at about 12,700, 19,200 and 24,900 output tokens/s/GPU with 64, 128 and 256 requests per rank. At $0.60 per million output tokens that is roughly $27, $42 and $54 per GPU-hour from output alone. That chart uses 8 DP ranks, which this backend does not support.

**What the sweep measures instead:** output and input tokens per second per GPU through the whole serving stack at 16 to 128 concurrent requests. `revenue.py` prices them at the off-peak rates above. The hit-rate columns are billing scenarios: the run still computed every input token, so they show what the same traffic would earn if that share were billed as cache hits, not a measured cache.

## Reading the result

```text
conc  reqs  err%  in tok/s/GPU  out tok/s/GPU  $/GPU-h @0% hit  ...
```

- Compare the output tokens/s/GPU column with DeepSeek's chart and the `$/GPU-h` column with $38. Expect a large gap.
- `--tokens usage` counts tokens from the server's usage instead of AIPerf's own tokenizer count. AIPerf's count is the default because Dynamo's streaming responses may not include usage.
- Concurrency here is the total across all GPUs. The chart's is per DP rank.
- Edit `ISL`, `OSL`, `CONCURRENCIES` and `DURATION` at the top of the bench script to change the load.

## Not yet verified

- Neither experiment has run on Modal or a GPU.
- `revenue.py` and the sweep loop were tested against a fake local OpenAI server, not against DeepInfer.
- AIPerf is installed unpinned in the image. The flags used here were checked against AIPerf 0.13.0, so a later release could change them.
- Prices are copied from DeepSeek's page and can change. Edit the constants at the top of `revenue.py`.
