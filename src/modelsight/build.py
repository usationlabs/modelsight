"""Plan the Modal image: resolve git refs, list the layers, fingerprint them.

Nothing here talks to Modal, so it can be tested and dry-run without an
account. Modal caches an image layer by its content: if the base image and the
commands up to a layer are unchanged, that layer is reused. The fingerprint is
a hash of the same inputs, so a run can say whether it will reuse the cache.
Git refs are resolved to commit SHAs first, so a branch that moved changes the
layer that clones it, and only that layer and the ones after it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from modelsight.config import Experiment, GitSource

# Pinned so image builds stay reproducible. Bump deliberately.
UV_VERSION = "0.12.5"
_FULL_SHA = re.compile(r"[0-9a-f]{40}")
_SHORT_SHA = re.compile(r"[0-9a-f]{7,39}")
_LS_REMOTE_TIMEOUT_SECONDS = 60


class BuildError(Exception):
    """The image cannot be planned, for example a ref that does not exist."""


@dataclass(frozen=True)
class ResolvedGit:
    repo: str
    path: str
    ref: str
    sha: str


Step = str | ResolvedGit
Resolver = Callable[[str, str], str]


@dataclass(frozen=True)
class Layer:
    """One Modal image method call: a shell command, or a set of env vars."""

    kind: str  # "run" or "env"
    command: str = ""
    env: tuple[tuple[str, str], ...] = ()
    # Names of local environment variables made available to this command only.
    secrets: tuple[str, ...] = ()
    gpu: str | None = None


@dataclass(frozen=True)
class Prepared:
    steps: tuple[Step, ...]
    layers: tuple[Layer, ...]
    fingerprint: str

    def info_text(self, exp: Experiment) -> str:
        """Lines for env.txt: what the image was built from."""
        lines = [f"image_fingerprint: {self.fingerprint}"]
        for step in self.steps:
            if isinstance(step, ResolvedGit):
                lines.append(
                    f"source {step.path}: {step.repo} {step.ref} -> {step.sha}"
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


def resolve_steps(
    steps: tuple[str | GitSource, ...], resolver: Resolver | None = None
) -> tuple[Step, ...]:
    resolver = resolver or resolve_ref
    resolved: list[Step] = []
    seen: dict[tuple[str, str], str] = {}
    for step in steps:
        if isinstance(step, str):
            resolved.append(step)
            continue
        key = (step.repo, step.ref)
        if key not in seen:
            seen[key] = resolver(step.repo, step.ref)
        resolved.append(ResolvedGit(step.repo, step.path, step.ref, seen[key]))
    return tuple(resolved)


def step_command(step: Step) -> str:
    """The shell command for a step. A git step fetches exactly one commit."""
    if isinstance(step, str):
        return step
    path = shlex.quote(step.path)
    return " && ".join(
        [
            f"git init -q {path}",
            f"git -C {path} remote add origin {shlex.quote(step.repo)}",
            f"git -C {path} fetch -q --depth 1 origin {step.sha}",
            f"git -C {path} checkout -q FETCH_HEAD",
        ]
    )


def plan_layers(exp: Experiment, steps: tuple[Step, ...]) -> tuple[Layer, ...]:
    """Layers in build order: ms tooling first, then the experiment's own."""
    layers = [Layer("run", command) for command in install_commands()]
    if exp.build:
        if exp.build.env:
            layers.append(Layer("env", env=exp.build.env))
        for step in steps:
            command = step_command(step)
            used = tuple(
                name
                for name in exp.build.secrets
                if re.search(rf"\${{?{re.escape(name)}\b", command)
            )
            layers.append(Layer("run", command, secrets=used, gpu=exp.build.gpu))
    if exp.cache:
        layers.append(Layer("env", env=(("HF_HOME", f"{exp.cache.mount}/huggingface"),)))
    return tuple(layers)


def fingerprint(image: str | None, layers: tuple[Layer, ...]) -> str:
    payload = json.dumps(
        {
            "image": image,
            "layers": [
                [layer.kind, layer.command, layer.env, layer.secrets, layer.gpu]
                for layer in layers
            ],
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def prepare(exp: Experiment, resolver: Resolver | None = None) -> Prepared:
    steps = (
        resolve_steps(exp.build.steps, resolver or resolve_ref) if exp.build else ()
    )
    layers = plan_layers(exp, steps)
    return Prepared(steps, layers, fingerprint(exp.image, layers))


def missing_env(
    exp: Experiment, prepared: Prepared, environ: Mapping[str, str] | None = None
) -> list[str]:
    """Names of local variables the run needs that are unset or empty."""
    env = os.environ if environ is None else environ
    needed = list(exp.secrets)
    for layer in prepared.layers:
        needed.extend(layer.secrets)
    return sorted({name for name in needed if not env.get(name)})
