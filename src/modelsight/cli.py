"""Command line: `ms run -f exp.yaml`."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import typer

from modelsight import __version__
from modelsight import build as image_plan
from modelsight.build import BuildError, Prepared, ResolvedGit
from modelsight.config import ConfigError, Experiment, load_experiment
from modelsight.runner import RunError, aiperf_status
from modelsight.targets import local, modal

app = typer.Typer(
    add_completion=False,
    help="Run a YAML experiment on a local GPU machine or on Modal.",
)


def run_name_for(config_path: Path, now: datetime | None = None) -> str:
    moment = now or datetime.now()
    return f"{config_path.stem}-{moment.strftime('%Y%m%d-%H%M%S')}"


def load_env_file(path: Path) -> None:
    """Load KEY=VALUE lines into the environment. Existing variables are left as they are."""
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        key, separator, value = line.partition("=")
        if not separator:
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key and value and key not in os.environ:
            os.environ[key] = value


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    load_env_file(Path.cwd() / ".env")
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit(1)


@app.command()
def run(
    config: Path = typer.Option(
        ...,
        "--config",
        "-f",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Experiment YAML file.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print what would run, without launching anything.",
    ),
) -> None:
    """Start the server, wait until it responds, run the bench, save results, stop."""
    try:
        experiment, config_text = load_experiment(config)
    except ConfigError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc

    name = run_name_for(config)
    prepared = _prepare(experiment)
    if dry_run:
        typer.echo(dry_run_report(experiment, config, name, prepared))
        return

    aiperf = aiperf_status(experiment)
    if aiperf.startswith("not found"):
        typer.echo(f"error: aiperf {aiperf}", err=True)
        raise typer.Exit(1)
    _require_env(experiment, prepared)

    try:
        if prepared is None:
            run_dir = local.run(experiment, config_text, name)
        else:
            run_dir = modal.run(experiment, config_text, name, prepared)
    except RunError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    except KeyboardInterrupt:
        typer.echo("interrupted; server was shut down", err=True)
        raise typer.Exit(130) from None
    typer.echo(f"results: {run_dir}")


def _prepare(exp: Experiment) -> Prepared | None:
    """Plan the Modal image, resolving git refs. None for a local run."""
    if exp.target != "modal":
        return None
    try:
        return image_plan.prepare(exp)
    except BuildError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc


def _require_env(exp: Experiment, prepared: Prepared | None) -> None:
    if prepared is None:
        return
    missing = image_plan.missing_env(exp, prepared)
    if missing:
        typer.echo(
            "error: these environment variables are needed but not set "
            f"(export them or put them in .env): {', '.join(missing)}",
            err=True,
        )
        raise typer.Exit(1)


def dry_run_report(
    exp: Experiment, config: Path, name: str, prepared: Prepared | None = None
) -> str:
    output = Path(exp.output)
    if not output.is_absolute():
        output = Path.cwd() / output
    lines = [
        f"modelsight {__version__} dry run",
        f"config: {config}",
        f"target: {exp.target}",
        f"gpu: {exp.gpu or '-'}",
        f"image: {exp.image or '-'}",
        f"output: {output / name}/",
        f"ready_url: {exp.ready_url}",
        f"ready_timeout: {exp.ready_timeout}",
        f"timeout: {exp.timeout}",
        f"server: {exp.server}",
        f"bench: {exp.bench}",
        f"aiperf: {aiperf_status(exp)}",
    ]
    if prepared is not None:
        lines.extend(_image_lines(exp, prepared))
    return "\n".join(lines)


def _image_lines(exp: Experiment, prepared: Prepared) -> list[str]:
    lines = [f"image_fingerprint: {prepared.fingerprint}"]
    if exp.build:
        lines.append(f"build: {len(prepared.steps)} steps")
        for number, step in enumerate(prepared.steps, start=1):
            if isinstance(step, ResolvedGit):
                lines.append(
                    f"  {number}. git {step.repo} {step.ref} -> {step.sha} at {step.path}"
                )
            else:
                first = step.strip().splitlines()[0]
                lines.append(f"  {number}. {first[:100]}{'...' if len(first) > 100 else ''}")
    if exp.cache:
        models = ", ".join(exp.cache.prefetch) or "-"
        lines.append(
            f"cache: volume={exp.cache.volume} mount={exp.cache.mount} prefetch={models}"
        )
    if exp.secrets:
        missing = set(image_plan.missing_env(exp, prepared))
        shown = ", ".join(
            f"{name} ({'missing' if name in missing else 'set'})" for name in exp.secrets
        )
        lines.append(f"secrets: {shown}")
    if exp.report:
        lines.append(f"report: {len(exp.report)} commands")
    return lines


@app.command()
def build(
    config: Path = typer.Option(
        ...,
        "--config",
        "-f",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Experiment YAML file.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the image plan and resolved refs, without building anything.",
    ),
) -> None:
    """Build the Modal image and download cache.prefetch models, without a GPU."""
    try:
        experiment, _ = load_experiment(config)
    except ConfigError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    if experiment.target != "modal":
        typer.echo("error: ms build only applies to target: modal", err=True)
        raise typer.Exit(1)
    if not experiment.build and not (experiment.cache and experiment.cache.prefetch):
        typer.echo(
            "error: nothing to build; add a build section or cache.prefetch", err=True
        )
        raise typer.Exit(1)

    prepared = _prepare(experiment)
    assert prepared is not None
    if dry_run:
        typer.echo(dry_run_report(experiment, config, run_name_for(config), prepared))
        return
    _require_env(experiment, prepared)
    try:
        modal.build(experiment, prepared)
    except RunError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    except KeyboardInterrupt:
        typer.echo("interrupted", err=True)
        raise typer.Exit(130) from None


@app.command()
def version() -> None:
    """Print the modelsight version."""
    typer.echo(__version__)
