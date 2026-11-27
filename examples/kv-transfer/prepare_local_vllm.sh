#!/usr/bin/env bash
# Copy the changed files of a local vLLM checkout into ./local_vllm, the build
# context of Dockerfile.peer_lookup_compat. Scratch tooling: not committed.
#
# What lands there is everything that differs from upstream/main in the checkout
# (committed or not): the vllm/ Python files go over a precompiled upstream-main
# install, and the whole tests/ tree comes along for the unit-test run. The
# review-fix branch changes Python only (no csrc), so the main wheel's kernels
# still match.
#
#   examples/kv-transfer/prepare_local_vllm.sh [/path/to/vllm] [base-sha] [archive-name]
#
# With no arguments: the review-fix worktree against upstream main 9df94b9977,
# into local_vllm.tar.gz (Dockerfile.peer_lookup_compat). The peer-stage branch:
#
#   examples/kv-transfer/prepare_local_vllm.sh /Users/bashorun/usation/vllm-peer-stage \
#       3332daf8fe613c0e3d7c06f42b12e95eb450aa1e local_vllm_peer_stage.tar.gz
#
# (Dockerfile.peer_stage; its base is the upstream commit the branch last merged,
# whose precompiled wheel the image installs.) The archive name differs per
# worktree so one build context holds both without clobbering.
#
# The base must equal the `sources.VLLM.ref` of the experiment files and the
# VLLM_SHA build arg; the Dockerfile fails the build otherwise.
set -euo pipefail
SRC="${1:-/Users/bashorun/usation/vllm-review-fixes}"
# The merge base the image installs (VLLM_SHA in the experiment files).
BASE="${2:-9df94b99774f1a0e17716f26dbb4eac50cc66f7a}"
ARCHIVE="${3:-local_vllm.tar.gz}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/local_vllm"

rm -rf "$OUT"
mkdir -p "$OUT/vllm" "$OUT/tests"
# Files in the working tree that differ from the base (minus deletions), plus new
# untracked ones.
{
    git -C "$SRC" diff --name-only --diff-filter=d "$BASE" -- vllm
    git -C "$SRC" ls-files --others --exclude-standard -- vllm
} | sort -u | while read -r f; do
    mkdir -p "$OUT/$(dirname "$f")"
    cp "$SRC/$f" "$OUT/$f"
done
if git -C "$SRC" diff --name-only "$BASE" -- csrc CMakeLists.txt cmake requirements pyproject.toml | grep -q .; then
    echo "WARNING: the checkout changes csrc/CMake/requirements vs $BASE; the precompiled wheel or its deps may not match" >&2
fi
rsync -a --exclude '__pycache__' "$SRC/tests/" "$OUT/tests/"
echo "base:     $(git -C "$SRC" rev-parse "$BASE")"
echo "head:     $(git -C "$SRC" rev-parse HEAD)"
echo "overlay:  $(find "$OUT/vllm" -type f | wc -l) vllm files"
git -C "$SRC" rev-parse "$BASE" > "$OUT/BASE_SHA"
git -C "$SRC" rev-parse HEAD > "$OUT/HEAD_SHA"
# One archive instead of ~3000 small files: Modal uploads the build context file
# by file, and the tests/ tree alone took over an hour.
COPYFILE_DISABLE=1 tar --no-xattrs -C "$OUT" -czf "$HERE/$ARCHIVE" vllm tests BASE_SHA HEAD_SHA
rm -rf "$OUT"
echo "archive:  $(du -h "$HERE/$ARCHIVE" | cut -f1) $ARCHIVE"
