"""Plan the Modal image: resolve git refs, list the layers ms adds, fingerprint.

The image comes from either a registry `image` or a Dockerfile. Nothing here
talks to Modal, so it can be tested and dry-run without an account.

Modal caches an image layer by its content: if the base and the instructions up
to a layer are unchanged, that layer is reused. `sources` give a Dockerfile git
repositories at a moving ref; each ref is resolved to a commit SHA first and
passed as a build arg, so a branch that moved changes the layer that uses it,
and only that layer and the ones after it. The fingerprint hashes the same
inputs, so a run can say whether it will reuse the cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from modelsight.config import Experiment

# Pinned so image builds stay reproducible. Bump deliberately.
UV_VERSION = "0.12.5"
_FULL_SHA = re.compile(r"[0-9a-f]{40}")
_SHORT_SHA = re.compile(r"[0-9a-f]{7,39}")
_LS_REMOTE_TIMEOUT_SECONDS = 60


class BuildError(Exception):
    """The image cannot be planned, for example a ref that does not exist."""


@dataclass(frozen=True)
class ResolvedSource:
    name: str
    repo: str
    ref: str
    sha: str

    def build_args(self) -> tuple[tuple[str, str], ...]:
        return (
            (f"{self.name}_REPO", self.repo),
            (f"{self.name}_REF", self.ref),
            (f"{self.name}_SHA", self.sha),
        )


Resolver = Callable[[str, str], str]


@dataclass(frozen=True)
class Layer:
    """One Modal image method call added after the base: a command or env vars."""

    kind: str  # "run" or "env"
    command: str = ""
    env: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Prepared:
    build_args: tuple[tuple[str, str], ...]
    sources: tuple[ResolvedSource, ...]
    layers: tuple[Layer, ...]
    fingerprint: str

    def info_text(self, exp: Experiment) -> str:
        """Lines for env.txt: what the image was built from."""
        lines = [f"image_fingerprint: {self.fingerprint}"]
        if exp.dockerfile:
            lines.append(f"dockerfile: {Path(exp.dockerfile).name}")
        for source in self.sources:
            lines.append(
                f"source {source.name}: {source.repo} {source.ref} -> {source.sha}"
            )
        if exp.secrets:
            lines.append(f"secrets (names only): {', '.join(exp.secrets)}")
        return "\n".join(lines) + "\n"


def install_commands() -> list[str]:
    """Shell commands that put a pinned uv and AIPerf on the image PATH."""
    installer = f"https://astral.sh/uv/{UV_VERSION}/install.sh"
    return [
        "DEBIAN_FRONTEND=noninteractive apt-get update && "
        "DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends curl ca-certificates",
        f"curl -LsSf {installer} | env UV_UNMANAGED_INSTALL=/usr/local/bin sh",
        "uv python install 3.12",
        "uv venv /opt/aiperf --python 3.12",
        "uv pip install --python /opt/aiperf/bin/python aiperf",
        "ln -sf /opt/aiperf/bin/aiperf /usr/local/bin/aiperf",
    ]


def resolve_ref(
    repo: str, ref: str, run: Callable[..., subprocess.CompletedProcess] = subprocess.run
) -> str:
    """Return the commit SHA that `ref` (a branch, tag or full SHA) points to."""
    if _FULL_SHA.fullmatch(ref):
        return ref
    patterns = [f"refs/heads/{ref}", f"refs/tags/{ref}^{{}}", f"refs/tags/{ref}"]
    try:
        result = run(
            ["git", "ls-remote", repo, *patterns],
            capture_output=True,
            text=True,
            timeout=_LS_REMOTE_TIMEOUT_SECONDS,
            check=False,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
    except FileNotFoundError as exc:
        raise BuildError("git is required to resolve build refs") from exc
    except subprocess.TimeoutExpired as exc:
        raise BuildError(f"timed out reading {repo}") from exc
    if result.returncode != 0:
        detail = (result.stderr or "").strip() or f"exit status {result.returncode}"
        raise BuildError(f"could not read {repo}: {detail}")
    found: dict[str, str] = {}
    for line in result.stdout.splitlines():
        sha, _, name = line.partition("\t")
        found[name.strip()] = sha.strip()
    for name in patterns:
        if name in found:
            return found[name]
    hint = (
        "; a short SHA cannot be resolved remotely, use the full 40-character SHA"
        if _SHORT_SHA.fullmatch(ref)
        else ""
    )
    raise BuildError(f"{ref!r} is not a branch or tag of {repo}{hint}")


def resolve_sources(
    exp: Experiment, resolver: Resolver | None = None
) -> tuple[ResolvedSource, ...]:
    resolver = resolver or resolve_ref
    return tuple(
        ResolvedSource(source.name, source.repo, source.ref, resolver(source.repo, source.ref))
        for source in exp.sources
    )


def plan_layers(exp: Experiment) -> tuple[Layer, ...]:
    """Layers ms adds after the base: its own tooling, then the cache env."""
    layers = [Layer("run", command) for command in install_commands()]
    if exp.cache:
        layers.append(Layer("env", env=(("HF_HOME", f"{exp.cache.mount}/huggingface"),)))
    return tuple(layers)


def fingerprint(
    exp: Experiment,
    dockerfile_text: str | None,
    build_args: tuple[tuple[str, str], ...],
    layers: tuple[Layer, ...],
) -> str:
    """A hash of what determines the image. Files COPY'd from the build context
    are not included, only the Dockerfile text."""
    payload = json.dumps(
        {
            "image": exp.image,
            "dockerfile": dockerfile_text,
            "build_args": sorted(build_args),
            "add_python": exp.add_python,
            "layers": [[layer.kind, layer.command, layer.env] for layer in layers],
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def prepare(exp: Experiment, resolver: Resolver | None = None) -> Prepared:
    sources = resolve_sources(exp, resolver)
    dockerfile_text = None
    if exp.dockerfile:
        try:
            dockerfile_text = Path(exp.dockerfile).read_text()
        except OSError as exc:
            raise BuildError(f"could not read dockerfile {exp.dockerfile}: {exc}") from exc
    build_args = exp.build_args + tuple(arg for source in sources for arg in source.build_args())
    layers = plan_layers(exp)
    return Prepared(
        build_args, sources, layers, fingerprint(exp, dockerfile_text, build_args, layers)
    )


def missing_env(exp: Experiment, environ: Mapping[str, str] | None = None) -> list[str]:
    """Names of local variables the run needs that are unset or empty."""
    env = os.environ if environ is None else environ
    return sorted({name for name in exp.secrets if not env.get(name)})
