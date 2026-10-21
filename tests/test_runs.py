from pathlib import Path

import pytest

from modelsight.runs import RunState, StateError, list_states, load_state, save_state


def _state(name: str = "b200-20261021-194951", started: str = "2026-10-21T19:49:51") -> RunState:
    return RunState(
        kind="run",
        name=name,
        app_id="ap-123",
        function_call_id="fc-456",
        profile="eagerloading",
        output="/work/results",
        volume="modelsight-results",
        started_at=started,
    )


def test_a_saved_run_loads_back_by_name(tmp_path: Path) -> None:
    path = save_state(_state(), tmp_path)
    assert path == tmp_path / "b200-20261021-194951.json"
    assert load_state("b200-20261021-194951", tmp_path) == _state()


def test_a_run_loads_by_the_path_of_its_file(tmp_path: Path) -> None:
    path = save_state(_state(), tmp_path)
    assert load_state(str(path)) == _state()


def test_state_is_written_in_the_working_directory_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    path = save_state(_state())
    assert path == tmp_path / ".ms" / "runs" / "b200-20261021-194951.json"
    assert load_state("b200-20261021-194951") == _state()


def test_an_unknown_run_says_where_it_looked(tmp_path: Path) -> None:
    with pytest.raises(StateError, match="directory where the run was started"):
        load_state("nope", tmp_path)


def test_a_damaged_file_is_reported_not_crashed_on(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text("{not json")
    with pytest.raises(StateError, match="could not read"):
        load_state("bad", tmp_path)
    (tmp_path / "short.json").write_text('{"kind": "run"}')
    with pytest.raises(StateError, match="could not read"):
        load_state("short", tmp_path)


def test_listing_is_oldest_first_and_skips_unreadable_files(tmp_path: Path) -> None:
    save_state(_state("second", "2026-10-21T20:00:00"), tmp_path)
    save_state(_state("first", "2026-10-21T19:00:00"), tmp_path)
    (tmp_path / "junk.json").write_text("nope")
    assert [s.name for s in list_states(tmp_path)] == ["first", "second"]


def test_listing_a_missing_directory_is_empty(tmp_path: Path) -> None:
    assert list_states(tmp_path / "missing") == []
