import subprocess

import pytest

from modelsight.build import (
    BuildError,
    Layer,
    ResolvedGit,
    fingerprint,
    install_commands,
    missing_env,
    plan_layers,
    prepare,
    resolve_ref,
    resolve_steps,
    step_command,
)
from modelsight.config import Build, Cache, Experiment, GitSource

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


def test_resolve_steps_asks_once_per_repo_and_ref() -> None:
    asked = []

    def resolver(repo: str, ref: str) -> str:
        asked.append((repo, ref))
        return SHA

    steps = (
        "echo hi",
        GitSource("https://x/r", "/opt/a", "main"),
        GitSource("https://x/r", "/opt/b", "main"),
        GitSource("https://x/r", "/opt/c", "dev"),
    )
    resolved = resolve_steps(steps, resolver)
    assert asked == [("https://x/r", "main"), ("https://x/r", "dev")]
    assert resolved[0] == "echo hi"
    assert resolved[1] == ResolvedGit("https://x/r", "/opt/a", "main", SHA)


def test_git_step_fetches_exactly_one_commit() -> None:
    command = step_command(ResolvedGit("https://x/r", "/opt/my dir", "main", SHA))
    assert "git init -q '/opt/my dir'" in command
    assert f"fetch -q --depth 1 origin {SHA}" in command
    assert command.endswith("checkout -q FETCH_HEAD")


def test_layers_put_tooling_first_and_cache_env_last() -> None:
    exp = _experiment(
        build=Build(steps=("echo one", "echo two"), env=(("A", "b"),), gpu="B200"),
        cache=Cache(volume="v", mount="/w"),
    )
    layers = plan_layers(exp, ("echo one", "echo two"))
    tooling = len(install_commands())
    assert [layer.command for layer in layers[:tooling]] == install_commands()
    assert layers[tooling] == Layer("env", env=(("A", "b"),))
    assert [layer.command for layer in layers[tooling + 1 : tooling + 3]] == [
        "echo one",
        "echo two",
    ]
    assert layers[tooling + 1].gpu == "B200"
    assert layers[-1] == Layer("env", env=(("HF_HOME", "/w/huggingface"),))


def test_build_secrets_only_reach_steps_that_use_them() -> None:
    exp = _experiment(
        build=Build(
            steps=("echo no secret", 'curl -H "Authorization: $TOKEN" x', "echo ${TOKEN}y"),
            secrets=("TOKEN", "UNUSED"),
        )
    )
    layers = plan_layers(exp, exp.build.steps)
    run_layers = layers[len(install_commands()) :]
    assert [layer.secrets for layer in run_layers] == [(), ("TOKEN",), ("TOKEN",)]


def test_experiment_without_build_adds_no_layers() -> None:
    assert len(plan_layers(_experiment(), ())) == len(install_commands())


def test_fingerprint_is_stable_and_tracks_inputs() -> None:
    exp = _experiment(build=Build(steps=("echo one",)))
    layers = plan_layers(exp, exp.build.steps)
    first = fingerprint(exp.image, layers)
    assert first == fingerprint(exp.image, layers)
    assert len(first) == 12
    assert fingerprint("base:2", layers) != first
    changed = plan_layers(_experiment(build=Build(steps=("echo two",))), ("echo two",))
    assert fingerprint(exp.image, changed) != first


def test_moving_a_branch_changes_the_fingerprint() -> None:
    exp = _experiment(build=Build(steps=(GitSource("https://x/r", "/opt/r", "main"),)))
    before = prepare(exp, lambda repo, ref: SHA)
    after = prepare(exp, lambda repo, ref: OTHER)
    same = prepare(exp, lambda repo, ref: SHA)
    assert before.fingerprint == same.fingerprint
    assert before.fingerprint != after.fingerprint


def test_info_text_lists_sources_and_secret_names_only() -> None:
    exp = _experiment(
        secrets=("HF_TOKEN",),
        build=Build(steps=(GitSource("https://x/r", "/opt/r", "dev"),)),
    )
    text = prepare(exp, lambda repo, ref: SHA).info_text(exp)
    assert "image_fingerprint:" in text
    assert f"source /opt/r: https://x/r dev -> {SHA}" in text
    assert "secrets (names only): HF_TOKEN" in text


def test_missing_env_reports_names_never_values() -> None:
    exp = _experiment(
        secrets=("HF_TOKEN", "OTHER"),
        build=Build(steps=("curl $GH",), secrets=("GH", "UNUSED")),
    )
    prepared = prepare(exp, lambda repo, ref: SHA)
    assert missing_env(exp, prepared, {"HF_TOKEN": "x", "UNUSED": ""}) == ["GH", "OTHER"]
    assert missing_env(exp, prepared, {"HF_TOKEN": "x", "OTHER": "y", "GH": "z"}) == []
