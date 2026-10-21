"""Load and check an experiment YAML file.

Engine and benchmark flags are not validated. `server` and `bench` are command
strings, copied through as the user wrote them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

TARGETS = ("local", "modal")
DEFAULT_READY_URL = "http://localhost:30000/health"
DEFAULT_READY_TIMEOUT = "20m"
DEFAULT_TIMEOUT = "2h"
DEFAULT_OUTPUT = "results/"
DEFAULT_GIT_REF = "main"
DEFAULT_CACHE_VOLUME = "modelsight-cache"
DEFAULT_CACHE_MOUNT = "/cache"

_KNOWN_FIELDS = {
    "target",
    "gpu",
    "image",
    "server",
    "bench",
    "ready_url",
    "ready_timeout",
    "timeout",
    "output",
    "dockerfile",
    "build_args",
    "sources",
    "add_python",
    "cache",
    "secrets",
    "report",
}
_CACHE_FIELDS = {"prefetch", "volume", "mount"}
_SOURCE_FIELDS = {"repo", "ref"}
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SOURCE_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
_DURATION = re.compile(r"^(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?$")


class ConfigError(Exception):
    """The experiment file is missing something or has a bad value."""


@dataclass(frozen=True)
class Source:
    """A git repository whose ref is resolved to a commit SHA before the build.

    The Dockerfile receives build args `<NAME>_REPO`, `<NAME>_REF` and
    `<NAME>_SHA`, so a branch that moved changes the layer that uses it.
    """

    name: str
    repo: str
    ref: str = DEFAULT_GIT_REF


@dataclass(frozen=True)
class Cache:
    """A Modal volume that keeps downloaded model weights between runs."""

    prefetch: tuple[str, ...] = ()
    volume: str = DEFAULT_CACHE_VOLUME
    mount: str = DEFAULT_CACHE_MOUNT


@dataclass(frozen=True)
class Experiment:
    target: str
    server: str
    bench: str
    gpu: str | None
    image: str | None
    ready_url: str
    ready_timeout: str
    timeout: str
    output: str
    # A Dockerfile (absolute once loaded) builds the Modal image instead of `image`.
    dockerfile: str | None = None
    build_args: tuple[tuple[str, str], ...] = ()
    sources: tuple[Source, ...] = ()
    # Modal adds this Python when the image has none on PATH.
    add_python: str | None = None
    cache: Cache | None = None
    # Names of local environment variables forwarded to the container.
    secrets: tuple[str, ...] = ()
    # Shell commands whose output is appended to env.txt.
    report: tuple[str, ...] = ()

    @property
    def ready_timeout_seconds(self) -> float:
        return parse_duration(self.ready_timeout)

    @property
    def timeout_seconds(self) -> float:
        return parse_duration(self.timeout)

    def to_mapping(self) -> dict[str, Any]:
        """The experiment as the YAML mapping it was loaded from.

        `experiment_from_mapping(exp.to_mapping()) == exp`, which lets a Modal
        container rebuild the experiment without a YAML parser.
        """
        data: dict[str, Any] = {
            "target": self.target,
            "server": self.server,
            "bench": self.bench,
            "ready_url": self.ready_url,
            "ready_timeout": self.ready_timeout,
            "timeout": self.timeout,
            "output": self.output,
        }
        if self.gpu:
            data["gpu"] = self.gpu
        if self.image:
            data["image"] = self.image
        if self.secrets:
            data["secrets"] = list(self.secrets)
        if self.report:
            data["report"] = list(self.report)
        if self.cache:
            data["cache"] = {
                "prefetch": list(self.cache.prefetch),
                "volume": self.cache.volume,
                "mount": self.cache.mount,
            }
        if self.dockerfile:
            data["dockerfile"] = self.dockerfile
        if self.build_args:
            data["build_args"] = dict(self.build_args)
        if self.sources:
            data["sources"] = {
                source.name: {"repo": source.repo, "ref": source.ref}
                for source in self.sources
            }
        if self.add_python:
            data["add_python"] = self.add_python
        return data


def parse_duration(value: str) -> float:
    """Parse `90`, `90s`, `20m`, `2h`, or a combination like `1h30m` into seconds."""
    text = value.strip()
    if text.isdigit():
        seconds = int(text)
    else:
        match = _DURATION.fullmatch(text)
        if match is None or not any(match.groups()):
            raise ConfigError(
                f"invalid duration {value!r}; use seconds, or a value like 90s, 20m, 2h"
            )
        hours, minutes, secs = (int(part or 0) for part in match.groups())
        seconds = hours * 3600 + minutes * 60 + secs
    if seconds <= 0:
        raise ConfigError(f"duration must be greater than zero, got {value!r}")
    return float(seconds)


def load_experiment(path: Path) -> tuple[Experiment, str]:
    """Return the experiment and the exact file text, so a run can save it unchanged."""
    import yaml

    try:
        text = path.read_text()
    except OSError as exc:
        raise ConfigError(f"could not read {path}: {exc}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping of experiment fields")
    experiment = experiment_from_mapping(data)
    if experiment.dockerfile:
        dockerfile = Path(experiment.dockerfile)
        if not dockerfile.is_absolute():
            dockerfile = (path.parent / dockerfile).resolve()
        if not dockerfile.is_file():
            raise ConfigError(f"dockerfile {experiment.dockerfile!r} not found at {dockerfile}")
        experiment = replace(experiment, dockerfile=str(dockerfile))
    return experiment, text


def experiment_from_mapping(data: dict) -> Experiment:
    unknown = sorted(set(data) - _KNOWN_FIELDS)
    if unknown:
        raise ConfigError(f"unknown field {unknown[0]!r}")

    target = _require_string(data, "target")
    if target not in TARGETS:
        raise ConfigError(f"target must be one of: {', '.join(TARGETS)}")

    gpu = _optional_string(data, "gpu")
    image = _optional_string(data, "image")
    dockerfile = _optional_string(data, "dockerfile")
    if target == "modal":
        if not gpu:
            raise ConfigError("gpu is required when target is modal, for example B200")
        if image and dockerfile:
            raise ConfigError("set image or dockerfile, not both")
        if not image and not dockerfile:
            raise ConfigError("image or dockerfile is required when target is modal")

    build_args = _parse_build_args(data.get("build_args"))
    sources = _parse_sources(data.get("sources"))
    add_python = _optional_string(data, "add_python")
    cache = _parse_cache(data.get("cache"))
    secrets = _env_names(data.get("secrets"), "secrets")
    if target == "local":
        for name, value in (
            ("dockerfile", dockerfile),
            ("build_args", build_args),
            ("sources", sources),
            ("add_python", add_python),
            ("cache", cache),
            ("secrets", secrets),
        ):
            if value:
                raise ConfigError(f"{name} is only supported when target is modal")
    clashes = {
        f"{source.name}_{suffix}"
        for source in sources
        for suffix in ("REPO", "REF", "SHA")
    } & {name for name, _ in build_args}
    if clashes:
        raise ConfigError(
            f"build_args {sorted(clashes)[0]!r} is set by sources; remove it from build_args"
        )
    if sources and not dockerfile:
        raise ConfigError("sources are passed to a Dockerfile, so dockerfile is required")

    ready_url = _optional_string(data, "ready_url") or DEFAULT_READY_URL
    if not ready_url.startswith(("http://", "https://")):
        raise ConfigError("ready_url must start with http:// or https://")
    ready_timeout = _optional_string(data, "ready_timeout") or DEFAULT_READY_TIMEOUT
    timeout = _optional_string(data, "timeout") or DEFAULT_TIMEOUT
    parse_duration(ready_timeout)
    parse_duration(timeout)

    return Experiment(
        target=target,
        server=_require_string(data, "server"),
        bench=_require_string(data, "bench"),
        gpu=gpu,
        image=image,
        ready_url=ready_url,
        ready_timeout=ready_timeout,
        timeout=timeout,
        output=_optional_string(data, "output") or DEFAULT_OUTPUT,
        dockerfile=dockerfile,
        build_args=build_args,
        sources=sources,
        add_python=add_python,
        cache=cache,
        secrets=secrets,
        report=_string_list(data.get("report"), "report"),
    )


def _require_string(data: dict, field: str) -> str:
    value = _optional_string(data, field)
    if not value:
        raise ConfigError(f"{field} is required and must be a non-empty string")
    return value


def _optional_string(data: dict, field: str) -> str | None:
    if field not in data or data[field] is None:
        return None
    value = data[field]
    if not isinstance(value, str):
        raise ConfigError(f"{field} must be a string")
    value = value.strip()
    return value or None


def _string_list(value: Any, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ConfigError(f"{field} must be a list of strings")
    items = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ConfigError(f"{field} must contain only non-empty strings")
        items.append(item.strip())
    return tuple(items)


def _env_names(value: Any, field: str) -> tuple[str, ...]:
    names = _string_list(value, field)
    for name in names:
        if not _ENV_NAME.fullmatch(name):
            raise ConfigError(
                f"{field} entries are environment variable names, got {name!r}"
            )
    return names


def _mapping(value: Any, field: str, allowed: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{field} must be a mapping")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ConfigError(f"unknown field '{field}.{unknown[0]}'")
    return value


def _parse_build_args(value: Any) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    if not isinstance(value, dict):
        raise ConfigError("build_args must be a mapping of names to strings")
    args = []
    for name, item in value.items():
        if not isinstance(name, str) or not _ENV_NAME.fullmatch(name):
            raise ConfigError(f"build_args name {name!r} is not a valid variable name")
        if not isinstance(item, (str, int, float)) or isinstance(item, bool):
            raise ConfigError(f"build_args.{name} must be a string")
        args.append((name, str(item)))
    return tuple(args)


def _parse_sources(value: Any) -> tuple[Source, ...]:
    if value is None:
        return ()
    if not isinstance(value, dict):
        raise ConfigError("sources must be a mapping of names to {repo, ref}")
    sources = []
    for name, item in value.items():
        if not isinstance(name, str) or not _SOURCE_NAME.fullmatch(name):
            raise ConfigError(
                f"sources name {name!r} must be upper case letters, digits and underscores"
            )
        data = _mapping(item, f"sources.{name}", _SOURCE_FIELDS)
        repo = _require_string(data, "repo")
        if not repo.startswith("https://"):
            raise ConfigError(f"sources.{name}.repo must be an https:// URL")
        sources.append(
            Source(name, repo, _optional_string(data, "ref") or DEFAULT_GIT_REF)
        )
    return tuple(sources)


def _parse_cache(value: Any) -> Cache | None:
    if value is None:
        return None
    data = _mapping(value, "cache", _CACHE_FIELDS)
    mount = _optional_string(data, "mount") or DEFAULT_CACHE_MOUNT
    if not mount.startswith("/"):
        raise ConfigError("cache.mount must be an absolute path")
    return Cache(
        prefetch=_string_list(data.get("prefetch"), "cache.prefetch"),
        volume=_optional_string(data, "volume") or DEFAULT_CACHE_VOLUME,
        mount=mount,
    )
