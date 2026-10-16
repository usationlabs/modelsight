"""Command line: `ms run -f exp.yaml`."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import typer

from modelsight import __version__
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
    if dry_run:
        typer.echo(dry_run_report(experiment, config, name))
        return

    aiperf = aiperf_status(experiment)
    if aiperf.startswith("not found"):
        typer.echo(f"error: aiperf {aiperf}", err=True)
        raise typer.Exit(1)

    target = local if experiment.target == "local" else modal
    try:
        run_dir = target.run(experiment, config_text, name)
    except RunError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    except KeyboardInterrupt:
        typer.echo("interrupted; server was shut down", err=True)
        raise typer.Exit(130) from None
    typer.echo(f"results: {run_dir}")


def dry_run_report(exp: Experiment, config: Path, name: str) -> str:
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
    return "\n".join(lines)


@app.command()
def version() -> None:
    """Print the modelsight version."""
    typer.echo(__version__)
