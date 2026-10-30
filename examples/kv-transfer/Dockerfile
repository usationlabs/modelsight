# vLLM with its pinned NIXL, plus the error mode NIXL needs on Modal. The base is
# the DeepInfer example's, and its compiler is what Triton builds GPU kernels with.
FROM nvidia/cuda:13.2.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Python 3.12 in a virtualenv that comes first on PATH. Modal needs python and
# pip on PATH.
RUN curl -LsSf https://astral.sh/uv/0.12.5/install.sh | env UV_UNMANAGED_INSTALL=/usr/local/bin sh \
    && uv venv /opt/venv --python 3.12 \
    && uv pip install --python /opt/venv/bin/python pip
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin

RUN uv pip install --python /opt/venv/bin/python vllm==0.32.0 nixl==1.5.0

COPY sitecustomize.py /opt/nixl-error-mode/sitecustomize.py
