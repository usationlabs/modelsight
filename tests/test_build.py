import subprocess
from pathlib import Path

import pytest

from modelsight.build import (
    BuildError,
    install_commands,
    missing_env,
    plan_layers,
    prepare,
    resolve_ref,
    resolve_sources,
)
from modelsight.config import Cache, Experiment, Source

SHA = "a" * 40
OTHER = "b" * 40


def _experiment(**overrides) -> Experiment:
    values = dict(
        target="modal",
        server="serve",
        bench="bench",
        gpu="B200",
        image="base:1",
        ready_url="http://localhost:1/",
        ready_timeout="1m",
        timeout="1h",
        output="results/",
    )
    values.update(overrides)
    return Experiment(**values)


def _ls_remote(stdout: str, returncode: int = 0, stderr: str = ""):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, returncode, stdout, stderr)

    run.calls = calls
    return run


def test_full_sha_is_returned_without_asking_git() -> None:
    run = _ls_remote("")
    assert resolve_ref("https://x/r", SHA, run) == SHA
    assert run.calls == []


def test_branch_resolves_to_its_head() -> None:
    run = _ls_remote(f"{SHA}\trefs/heads/feat/deepinfer\n")
    assert resolve_ref("https://x/r", "feat/deepinfer", run) == SHA
    assert run.calls[0][:3] == ["git", "ls-remote", "https://x/r"]


def test_annotated_tag_resolves_to_the_commit_it_points_at() -> None:
    run = _ls_remote(f"{OTHER}\trefs/tags/v1\n{SHA}\trefs/tags/v1^{{}}\n")
    assert resolve_ref("https://x/r", "v1", run) == SHA


def test_branch_wins_over_a_tag_with_the_same_name() -> None:
    run = _ls_remote(f"{OTHER}\trefs/tags/v1\n{SHA}\trefs/heads/v1\n")
    assert resolve_ref("https://x/r", "v1", run) == SHA


def test_unknown_ref_is_an_error() -> None:
    with pytest.raises(BuildError, match="not a branch or tag"):
        resolve_ref("https://x/r", "nope", _ls_remote(""))


def test_short_sha_gets_a_hint() -> None:
    with pytest.raises(BuildError, match="full 40-character SHA"):
        resolve_ref("https://x/r", "abc1234", _ls_remote(""))


def test_git_failure_is_reported() -> None:
    run = _ls_remote("", returncode=128, stderr="fatal: repository not found")
    with pytest.raises(BuildError, match="repository not found"):
        resolve_ref("https://x/r", "main", run)


def test_sources_become_repo_ref_and_sha_build_args() -> None:
    exp = _experiment(
        dockerfile="/x/Dockerfile",
        image=None,
        build_args=(("JOBS", "8"),),
        sources=(Source("DYNAMO", "https://x/dynamo", "feat/deepinfer"),),
    )
    asked = []

    def resolver(repo: str, ref: str) -> str:
        asked.append((repo, ref))
        return SHA

    resolved = resolve_sources(exp, resolver)
    assert asked == [("https://x/dynamo", "feat/deepinfer")]
    assert resolved[0].build_args() == (
        ("DYNAMO_REPO", "https://x/dynamo"),
        ("DYNAMO_REF", "feat/deepinfer"),
        ("DYNAMO_SHA", SHA),
    )


def _dockerfile_experiment(tmp_path: Path, text: str = "FROM scratch\n", **overrides) -> Experiment:
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(text)
    return _experiment(image=None, dockerfile=str(dockerfile), **overrides)


def test_prepare_passes_user_args_then_source_args(tmp_path: Path) -> None:
    exp = _dockerfile_experiment(
        tmp_path,
        build_args=(("JOBS", "8"),),
        sources=(Source("A", "https://x/a", "main"),),
    )
    prepared = prepare(exp, lambda repo, ref: SHA)
    assert prepared.build_args == (
        ("JOBS", "8"),
        ("A_REPO", "https://x/a"),
        ("A_REF", "main"),
        ("A_SHA", SHA),
    )


def test_layers_are_ms_tooling_then_the_cache_env() -> None:
    layers = plan_layers(_experiment(cache=Cache(mount="/w")))
    assert [layer.command for layer in layers[:-1]] == install_commands()
    assert layers[-1].env == (("HF_HOME", "/w/huggingface"),)
    assert len(plan_layers(_experiment())) == len(install_commands())


def test_fingerprint_is_stable_and_tracks_inputs(tmp_path: Path) -> None:
    base = prepare(_dockerfile_experiment(tmp_path)).fingerprint
    assert len(base) == 12
    assert prepare(_dockerfile_experiment(tmp_path)).fingerprint == base
    assert prepare(_dockerfile_experiment(tmp_path, "FROM scratch\nRUN true\n")).fingerprint != base
    assert prepare(_dockerfile_experiment(tmp_path, build_args=(("A", "1"),))).fingerprint != base
    assert prepare(_dockerfile_experiment(tmp_path, add_python="3.12")).fingerprint != base
    assert prepare(_dockerfile_experiment(tmp_path, cache=Cache())).fingerprint != base
    assert prepare(_experiment()).fingerprint != prepare(_experiment(image="base:2")).fingerprint


def test_moving_a_branch_changes_the_fingerprint(tmp_path: Path) -> None:
    exp = _dockerfile_experiment(tmp_path, sources=(Source("A", "https://x/a", "main"),))
    before = prepare(exp, lambda repo, ref: SHA)
    after = prepare(exp, lambda repo, ref: OTHER)
    same = prepare(exp, lambda repo, ref: SHA)
    assert before.fingerprint == same.fingerprint
    assert before.fingerprint != after.fingerprint


def test_unreadable_dockerfile_is_a_build_error(tmp_path: Path) -> None:
    exp = _experiment(image=None, dockerfile=str(tmp_path / "missing"))
    with pytest.raises(BuildError, match="could not read dockerfile"):
        prepare(exp)


def test_info_text_lists_sources_and_secret_names_only(tmp_path: Path) -> None:
    exp = _dockerfile_experiment(
        tmp_path,
        secrets=("HF_TOKEN",),
        sources=(Source("A", "https://x/a", "dev"),),
    )
    text = prepare(exp, lambda repo, ref: SHA).info_text(exp)
    assert "image_fingerprint:" in text
    assert "dockerfile: Dockerfile" in text
    assert f"source A: https://x/a dev -> {SHA}" in text
    assert "secrets (names only): HF_TOKEN" in text


def test_missing_env_reports_names_never_values() -> None:
    exp = _experiment(secrets=("HF_TOKEN", "OTHER"))
    assert missing_env(exp, {"HF_TOKEN": "x", "OTHER": ""}) == ["OTHER"]
    assert missing_env(exp, {"HF_TOKEN": "x", "OTHER": "y"}) == []
