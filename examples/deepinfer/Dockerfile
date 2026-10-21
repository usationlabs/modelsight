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
#
# DeepEP and DeepGEMM compile against DeepJIT, whose exception.hpp includes
# <elfutils/libdwfl.h> (libdw-dev) and loads libdw.so.1 at runtime to print
# stack traces, so the same package serves the build and the running image.
RUN apt-get update && apt-get install -y --no-install-recommends pciutils libdw-dev \
    && rm -rf /var/lib/apt/lists/* \
    && printf '#!/bin/sh\n/usr/bin/lspci "$@" 2>/dev/null\nexit 0\n' > /usr/local/bin/lspci \
    && chmod +x /usr/local/bin/lspci

# DeepInfer's build.sh installs the requirements (done above), patches NCCL,
# builds the vendored kernels with scripts/build_vendor.sh, then builds DeepInfer.
# They are separate layers here, one per vendored package with the same commands
# build_vendor.sh runs, so a failure in one does not repeat the others' long
# compiles.
RUN pip install --no-deps --force-reinstall nvidia-nccl-cu13==2.32.3

ENV WHEEL_DIR=/opt/vendor-wheels
RUN mkdir -p "$WHEEL_DIR" \
    && git config --global url."https://github.com/".insteadOf git@github.com: \
    && git submodule update --init

RUN cd vendor/DeepSelect && git submodule update --init --recursive \
    && DEEP_SELECT_BUILD_TARGET_PLATFORM=CUDA python setup.py bdist_wheel --dist-dir "$WHEEL_DIR"

# FlashMLA fails its build if any kernel of its own spills registers. With
# CUDA 13.2's nvcc, four instantiations of the sm100 sparse decode kernel
# (flash_fwd_splitkv_mla_fp8_sparse_kernel, head64) spill a few bytes: a 16-byte
# stack frame, 20 bytes of spill stores and 28 of loads. That is a performance
# guard, not a correctness one, and FlashMLA provides this switch to skip it. It
# is set only for this step. Compare kernel speed if throughput looks low.
RUN cd vendor/FlashMLA && git submodule update --init --recursive \
    && FLASH_MLA_BUILD_TARGET_PLATFORM=CUDA FLASH_MLA_SKIP_REG_SPILL_CHECK=1 \
       python setup.py bdist_wheel --dist-dir "$WHEEL_DIR"

RUN cd vendor/DeepEP && git submodule update --init --recursive \
    && python setup.py bdist_wheel --dist-dir "$WHEEL_DIR"

RUN cd vendor/DeepGEMM && git submodule update --init --recursive \
    && DG_FORCE_BUILD=1 python setup.py bdist_wheel --dist-dir "$WHEEL_DIR"

RUN python -m pip install --no-deps --no-index --find-links "$WHEEL_DIR" \
    deep_select flash_mla deep_ep deep_gemm
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

# DeepInfer is built in place and never installed: build.sh puts the compiled
# extension under deepinfer/_so/ inside the source tree, and DeepInfer's own
# scripts add the repository root to sys.path. Do the same for Dynamo's worker.
# Last on purpose, so it does not invalidate any of the slow layers above.
ENV PYTHONPATH=/opt/DeepInfer
# Fail the build, not a GPU run, if DeepInfer is not importable or has no
# compiled extension. It imports only what loads without a GPU driver.
RUN python -c "import glob, os, deepinfer, deep_cluster; \
so = [p for p in glob.glob(os.path.join(os.path.dirname(deepinfer.__file__), '_so', '*', '*.so')) if os.path.exists(p)]; \
assert so, 'no compiled DeepInfer extension under deepinfer/_so'; print(len(so), 'extension(s):', so)"
