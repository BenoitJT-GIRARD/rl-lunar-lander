"""The exporter's verdict on a policy whose grids fall below the solved threshold.

An untrained checkpoint crashes on every episode, so its mean is far below 200. The exports go
to `tmp_path`: the script's output directory and its CSV writer are redirected there, and the
published files under `reports/` are never touched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import evaluate_and_export as script
import pytest

from rl_lander.training import evaluate


@pytest.fixture
def exporter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, untrained_checkpoint: Path):
    """Run `main()` on the untrained policy, two episodes on one grid, writing to `tmp_path`."""
    monkeypatch.setattr(script, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(script, "ensure_dirs", lambda: None)
    monkeypatch.setattr(
        script,
        "write_csv",
        lambda records: evaluate.write_csv(records, tmp_path / "evaluation_episodes.csv"),
    )

    def run(*extra: str) -> None:
        argv = ["evaluate_and_export.py", "--model", str(untrained_checkpoint)]
        monkeypatch.setattr(sys, "argv", [*argv, "--episodes", "2", "--seeds", "1", *extra])
        script.main()

    return run


def test_a_grid_below_the_threshold_ends_the_command_with_an_error(
    exporter, tmp_path: Path
) -> None:
    """The exports are written first, then the command fails and names the option to pass."""
    with pytest.raises(SystemExit) as stopped:
        exporter()

    assert "--allow-below-threshold" in str(stopped.value.code)
    assert (tmp_path / "evaluation_episodes.csv").is_file()
    assert (tmp_path / "evaluation_summary.json").is_file()


def test_the_option_the_error_names_is_accepted_and_keeps_the_exports(
    exporter, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exporter("--allow-below-threshold")

    assert "Exported anyway" in capsys.readouterr().out
    summary = json.loads((tmp_path / "evaluation_summary.json").read_text(encoding="utf-8"))
    assert summary["metrics"]["mean_reward"] < evaluate.SOLVED_THRESHOLD
