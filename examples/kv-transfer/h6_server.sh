#!/usr/bin/env bash
# The engine of an H6 node, shared by h6*.yaml. Every rank runs it; MS_RANK picks
# the role: rank 0 is A (the producer, with peer_stage on), rank 1 is B (the
# consumer). The image carries it as /opt/h6_server.sh (Dockerfile.h6 COPYs it).
#
# Knobs, from the environment (the yamls set them):
#   PEER_STAGE_DEADLINE  A's peer_stage_deadline in seconds (default 1.5)
#   PEER_STAGE_RATE      A's peer_stage_recompute_tokens_per_s; 0 turns the skip
#                        rule's recompute half off (default 0)
#   CPU_GB               A's CPU offload tier in GiB (default 64)
#   CPU_GB_B             B's (default 32)
#
# The runner gives MS_RANK, MS_NODE_IP and MS_RUN_DIR. Each server logs untagged to
# $MS_RUN_DIR/rank-$MS_RANK/server-<A|B>.log, which the probe reads (A's through the
# control endpoint on port 8399), and tagged [A] or [B] on the run's output.
set -u

export PYTHONPATH=/opt/nixl-error-mode
export PYTHONHASHSEED=0
export VLLM_SERVER_DEV_MODE=1
# UCX_PROTO_INFO prints the protocol UCX picks for each operation, so B's log
# shows the transport its KV reads take: the probe's rdma_path check reads it.
export UCX_RCACHE_MAX_UNRELEASED=1024 UCX_PROTO_INFO=y

RANK=${MS_RANK:?MS_RANK is not set: run this through ms}
LOG_DIR=$MS_RUN_DIR/rank-$RANK
mkdir -p "$LOG_DIR"

# DEBUG for the modules the probe reads: A's pin server (the `Peer stage ...` and
# `Peer lookup_pin pinned` lines), B's scheduler and client (`Peer lookup_pin hit`,
# `Peer lookup ... missed`), the stage source and the offloading scheduler. The
# rest of vLLM stays at INFO: under load the engine core would log every step.
V1=vllm.distributed.kv_transfer.kv_connector.v1
cat > /tmp/vllm-logging.json <<JSON
{
  "version": 1,
  "disable_existing_loggers": false,
  "formatters": {"vllm": {"()": "vllm.logging_utils.NewLineFormatter",
    "datefmt": "%m-%d %H:%M:%S",
    "format": "(%(vllm_process_name)s pid=%(process)d) %(levelname)s %(asctime)s.%(msecs)03d [%(fileinfo)s:%(lineno)d] %(message)s"}},
  "handlers": {"vllm": {"class": "logging.StreamHandler", "formatter": "vllm",
    "level": "DEBUG", "stream": "ext://sys.stdout"}},
  "loggers": {
    "vllm": {"handlers": ["vllm"], "level": "INFO", "propagate": false},
    "$V1.nixl.peer_pins": {"level": "DEBUG"},
    "$V1.nixl.peer_lookup_client": {"level": "DEBUG"},
    "$V1.nixl.pull_scheduler": {"level": "DEBUG"},
    "$V1.peer_stage": {"level": "DEBUG"},
    "$V1.offloading.scheduler": {"level": "DEBUG"}
  }
}
JSON
export VLLM_LOGGING_CONFIG_PATH=/tmp/vllm-logging.json

GIB=1073741824

# serve NAME PORT SIDE_CHANNEL_PORT CPU_BYTES NIXL_EXTRA_CONFIG
# MultiConnector lists NixlConnector first: with Offloading first, a staged copy
# goes out one step later. --max-num-seqs 16 is kv_peer_load.A_MAX_SEQS: sessions
# beyond it queue on A. --block-size 16 is the probe's --block-size (vLLM picks 64
# on H100, which overflows prompts sized in 16-token blocks). --no-enable-extensible-kv-cache keeps KV exportable to NIXL.
# The side channel host is this node's IP, so B on the other node can dial it.
serve() {
  local cfg="{\"kv_connector\":\"MultiConnector\",\"kv_role\":\"kv_both\",\"kv_load_failure_policy\":\"fail\",\"kv_connector_extra_config\":{\"connectors\":[{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":$5},{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":$4}}]}}"
  VLLM_NIXL_SIDE_CHANNEL_HOST=$MS_NODE_IP VLLM_NIXL_SIDE_CHANNEL_PORT=$3 \
    stdbuf -oL vllm serve Qwen/Qwen3-8B \
      --port "$2" \
      --max-model-len 32768 \
      --gpu-memory-utilization 0.9 \
      --block-size 16 \
      --prefix-caching-hash-algo sha256 \
      --no-enable-extensible-kv-cache \
      --max-num-seqs 16 \
      --kv-transfer-config "$cfg" \
      2>&1 | tee "$LOG_DIR/server-$1.log" | sed -u "s/^/[$1] /"
}

STAGE="{\"peer_lookup\":true,\"peer_stage\":true,\"peer_stage_deadline\":${PEER_STAGE_DEADLINE:-1.5},\"peer_stage_recompute_tokens_per_s\":${PEER_STAGE_RATE:-0},\"peer_stage_max_fraction\":0.10}"
if [ "${MS_NODES:-1}" = 1 ]; then
  # One node, two GPUs (as H5): A on GPU 0, B on GPU 1. The run stops if
  # either exits.
  CUDA_VISIBLE_DEVICES=0 serve A 8100 5600 $(( ${CPU_GB:-64} * GIB )) "$STAGE" &
  CUDA_VISIBLE_DEVICES=1 serve B 8200 5601 $(( ${CPU_GB_B:-32} * GIB )) '{"peer_lookup":true}' &
  wait -n
elif [ "$RANK" = 0 ]; then
  serve A 8100 5600 $(( ${CPU_GB:-64} * GIB )) "$STAGE"
else
  serve B 8200 5601 $(( ${CPU_GB_B:-32} * GIB )) '{"peer_lookup":true}'
fi
