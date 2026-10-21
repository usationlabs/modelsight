# DeepInfer (DeepSeek's inference framework) served through Dynamo, for Modal.
#
# `ms build -f b200x8-1m.yaml` passes the build args below: it resolves each
# source's branch or tag to a commit SHA so a moved branch rebuilds the layer
# that clones it. To build with plain docker instead:
#   docker build \
#     --build-arg DEEPINFER_REPO=https://github.com/deepseek-ai/DeepInfer \
#     --build-arg DEEPINFER_SHA=<40-character commit> \
#     --build-arg DYNAMO_REPO=https://github.com/usationlabs/dynamo \
#     --build-arg DYNAMO_SHA=<40-character commit> .
#
# Layers go from slowest and most stable to the part that changes most often,
# so Modal's layer cache is reused when only Dynamo's branch moves.
#
# The base matches DeepInfer's pins: its requirements use torch 2.13.0+cu132,
# its README asks for CUDA 13 and GCC 13 or newer, and Ubuntu 24.04 ships GCC 13.
FROM nvidia/cuda:13.2.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
# Package list taken from Dynamo's own container/templates/dev.Dockerfile.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake pkg-config protobuf-compiler libprotobuf-dev \
        clang libclang-dev patchelf git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Python 3.12 in a virtualenv that comes first on PATH. Modal needs python and
# pip on PATH; if it does not accept a virtualenv, set add_python in the YAML.
RUN curl -LsSf https://astral.sh/uv/0.12.5/install.sh | env UV_UNMANAGED_INSTALL=/usr/local/bin sh \
    && uv venv /opt/venv --python 3.12 \
    && uv pip install --python /opt/venv/bin/python pip
# DG_JIT_CACHE_DIR: DeepGEMM and DeepEP compile kernels at runtime. Keep that
# cache on the volume ms mounts at /cache so a second run does not recompile.
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/root/.cargo/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin \
    DEEPINFER_ACCELERATOR=nvidia \
    DG_JIT_CACHE_DIR=/cache/dj

# --- DeepInfer, at a pinned commit ------------------------------------------
ARG DEEPINFER_REPO
ARG DEEPINFER_SHA
RUN test -n "$DEEPINFER_REPO" -a -n "$DEEPINFER_SHA" \
    || { echo "set build args DEEPINFER_REPO and DEEPINFER_SHA" >&2; exit 1; }
RUN git init -q /opt/DeepInfer \
    && git -C /opt/DeepInfer remote add origin "$DEEPINFER_REPO" \
    && git -C /opt/DeepInfer fetch -q --depth 1 origin "$DEEPINFER_SHA" \
    && git -C /opt/DeepInfer checkout -q FETCH_HEAD
WORKDIR /opt/DeepInfer
# Slow dependency layer first; build.sh then builds the vendored kernels.
RUN pip install -r requirements-nvidia.txt

# DeepSelect's and FlashMLA's setup.py detect the platform by running `lspci`
# as soon as they are imported, and fail if it is missing or exits non-zero.
# A build machine has no GPU, so it reports CPU_ONLY and the explicit CUDA
# target that DeepInfer's build_vendor.sh passes takes over. The wrapper keeps
# lspci from failing where /sys/bus/pci is unavailable (sandboxed builders).
# It sits after the dependency layer so changing it does not redo that layer.
RUN apt-get update && apt-get install -y --no-install-recommends pciutils \
    && rm -rf /var/lib/apt/lists/* \
    && printf '#!/bin/sh\n/usr/bin/lspci "$@" 2>/dev/null\nexit 0\n' > /usr/local/bin/lspci \
    && chmod +x /usr/local/bin/lspci

# DeepInfer's build.sh installs the requirements (done above), patches NCCL,
# builds the vendored kernels, then builds DeepInfer. Those are separate layers
# here, so a failure in a later one does not repeat the long kernel compiles.
RUN pip install --no-deps --force-reinstall nvidia-nccl-cu13==2.32.3

# FlashMLA fails its build if any kernel of its own spills registers. With
# CUDA 13.2's nvcc, four instantiations of the sm100 sparse decode kernel
# (flash_fwd_splitkv_mla_fp8_sparse_kernel, head64) spill a few bytes: a 16-byte
# stack frame, 20 bytes of spill stores and 28 of loads. That is a performance
# guard, not a correctness one, and FlashMLA provides this switch to skip it. It
# is set only for this step. Compare kernel speed if throughput looks low.
RUN FLASH_MLA_SKIP_REG_SPILL_CHECK=1 bash ./scripts/build_vendor.sh
RUN bash ./scripts/build.sh build

# --- Dynamo's Python bindings (Rust) ----------------------------------------
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal

# Last on purpose: this is the part that changes when the branch moves.
ARG DYNAMO_REPO
ARG DYNAMO_SHA
RUN test -n "$DYNAMO_REPO" -a -n "$DYNAMO_SHA" \
    || { echo "set build args DYNAMO_REPO and DYNAMO_SHA" >&2; exit 1; }
RUN git init -q /opt/dynamo \
    && git -C /opt/dynamo remote add origin "$DYNAMO_REPO" \
    && git -C /opt/dynamo fetch -q --depth 1 origin "$DYNAMO_SHA" \
    && git -C /opt/dynamo checkout -q FETCH_HEAD
WORKDIR /opt/dynamo
RUN uv pip install 'maturin[patchelf]' \
    && (cd lib/bindings/python && maturin develop --uv) \
    && uv pip install -e lib/gpu_memory_service -e .
WORKDIR /
