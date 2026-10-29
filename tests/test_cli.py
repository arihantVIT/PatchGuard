import json

import patchguard.storage as storage
from patchguard.__main__ import main


def test_cli_status_shows_test_history_and_best_checkpoint(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(storage, "STORAGE_DIR", tmp_path)
    storage.save_version("demo", "v1", {"module.py": "value = 1\n"})
    storage.record_test_result(
        "demo",
        "v1",
        "example_suite_v1",
        {
            "status": "passed",
            "passed_count": 2,
            "failed_count": 0,
            "stdout": "2 passed",
        },
    )
    monkeypatch.setattr("sys.argv", ["patchguard", "status", "demo"])

    assert main() == 0
    output = capsys.readouterr().out
    assert "example_suite_v1: passed" in output
    assert '"version_id": "v1"' in output
    assert "passed=2" in output


def test_cli_restore_requires_approval_and_writes_when_confirmed(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(storage, "STORAGE_DIR", tmp_path / "snapshots")
    storage.save_version("demo", "v1", {"src/module.py": "value = 1\n"})
    target = tmp_path / "working-copy"

    monkeypatch.setattr(
        "sys.argv",
        ["patchguard", "restore", "demo", "v1", str(target)],
    )
    assert main() == 2
    assert not target.exists()

    monkeypatch.setattr(
        "sys.argv",
        ["patchguard", "restore", "demo", "v1", str(target), "--confirm"],
    )
    assert main() == 0
    assert (target / "src/module.py").read_text(encoding="utf-8") == "value = 1\n"
    assert "Restored demo v1" in capsys.readouterr().out


def test_cli_test_returns_zero_for_failed_behavioral_tests(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(storage, "STORAGE_DIR", tmp_path)
    storage.save_version(
        "calculator",
        "v1",
        {
            "calculator.py": (
                "def calculate_total(price, discount):\n"
                "    return price + discount\n"
            )
        },
    )
    capsys.readouterr()
    monkeypatch.setattr("sys.argv", ["patchguard", "test", "calculator", "v1"])

    assert main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "failed"
    assert result["failed_count"] > 0


def test_cli_restore_rejects_unsafe_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "STORAGE_DIR", tmp_path / "snapshots")
    storage.save_version("demo", "v1", {"../outside.py": "value = 1\n"})
    target = tmp_path / "working-copy"
    monkeypatch.setattr(
        "sys.argv",
        [
            "patchguard", "restore", "demo", "v1", str(target),
            "--confirm",
        ],
    )

    assert main() == 1
    assert not target.exists()


def test_cli_eval_command(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(storage, "STORAGE_DIR", tmp_path / "patches")
    monkeypatch.setattr("sys.argv", ["patchguard", "eval"])
    assert main() == 0
    out = capsys.readouterr().out
    assert "PATCHGUARD EVALUATION SUITE" in out
    assert "Total Cases: 4" in out
    assert "False Positive Rate: 0.0%" in out


def test_cli_dashboard_export(tmp_path, monkeypatch, capsys):
    export_file = tmp_path / "dash.html"
    monkeypatch.setattr(storage, "STORAGE_DIR", tmp_path / "patches")
    monkeypatch.setattr(
        "sys.argv",
        ["patchguard", "dashboard", "--export", str(export_file)],
    )
    assert main() == 0
    assert export_file.exists()
    out = capsys.readouterr().out
    assert "Standalone PatchGuard dashboard exported to" in out
