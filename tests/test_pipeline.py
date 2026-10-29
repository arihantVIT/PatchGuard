import json

import patchguard.storage as storage
from patchguard.behavioral_testing import (
    CALCULATOR_SUITE_ID,
    run_behavioral_tests,
)
from patchguard.pipeline import compare_versions, test_version as run_version


def test_runner_returns_explicit_counts():
    result = run_behavioral_tests(
        "v1",
        "def calculate_total(price, discount):\n    return price - discount\n",
    )

    assert result["status"] == "failed"
    assert result["passed_count"] == 1
    assert result["failed_count"] == 2
    assert result["error_count"] == 0
    assert result["skipped_count"] == 0


def test_test_version_persists_and_promotes_passing_result(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(storage, "STORAGE_DIR", tmp_path)
    storage.save_version(
        "calculator",
        "v1",
        {
            "calculator.py": (
                "def calculate_total(price, discount):\n"
                "    if discount < 0 or discount > price:\n"
                "        raise ValueError()\n"
                "    return price - discount\n"
            )
        },
    )

    result = run_version("calculator", "v1")
    metadata = storage.load_checkpoint_metadata("calculator")

    assert result["status"] == "passed"
    assert result["passed_count"] == 3
    assert result["failed_count"] == 0
    assert metadata["best_checkpoints"][CALCULATOR_SUITE_ID]["version_id"] == "v1"
    runs = metadata["versions"]["v1"]["test_runs"][CALCULATOR_SUITE_ID]
    assert len(runs) == 1
    assert runs[0]["stdout"]


def test_unsupported_version_is_not_run_or_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "STORAGE_DIR", tmp_path)
    storage.save_version("other", "v1", {"module.py": "value = 3\n"})

    result = run_version("other", "v1")

    assert result["status"] == "not_run"
    metadata = storage.load_checkpoint_metadata("other")
    assert metadata["best_checkpoints"] == {}
    assert metadata["versions"]["v1"]["test_runs"] == {}


def test_comparison_uses_persisted_version_testing(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "STORAGE_DIR", tmp_path)
    passing_code = (
        "def calculate_total(price, discount):\n"
        "    if discount < 0 or discount > price:\n"
        "        raise ValueError()\n"
        "    return price - discount\n"
    )
    storage.save_version(
        "calculator", "v1", {"calculator.py": passing_code}
    )
    storage.save_version(
        "calculator",
        "v2",
        {"calculator.py": passing_code.replace("return price - discount", "return price + discount")},
    )

    report = compare_versions(
        "calculator", "v1", "v2", run_llm=False
    )
    metadata = storage.load_checkpoint_metadata("calculator")

    assert report["tests"]["original"]["status"] == "passed"
    assert report["tests"]["candidate"]["status"] == "failed"
    assert metadata["best_checkpoints"][CALCULATOR_SUITE_ID]["version_id"] == "v1"
    assert len(
        metadata["versions"]["v1"]["test_runs"][CALCULATOR_SUITE_ID]
    ) == 1
    assert json.loads(json.dumps(report["tests"]))["candidate"]["failed_count"] > 0
