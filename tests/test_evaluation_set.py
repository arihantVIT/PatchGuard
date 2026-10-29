"""Tests for PatchGuard realistic evaluation dataset."""

import pytest
import patchguard.storage as storage
from patchguard.evaluation import (
    EVALUATION_CASES,
    seed_evaluation_fixtures,
    evaluate_case,
    run_evaluation,
)


def test_seed_and_run_evaluation(tmp_path, monkeypatch):
    """Verify evaluation suite runs and succeeds in an isolated environment."""
    monkeypatch.setattr(storage, "STORAGE_DIR", tmp_path / "patches")
    seed_evaluation_fixtures(tmp_path / "patches")

    result = run_evaluation(seed_if_missing=False)
    summary = result["summary"]

    assert summary["total_cases"] == 4
    assert summary["passed_cases"] == 4
    assert summary["detection_rate_pct"] == 100.0
    assert summary["false_positive_rate_pct"] == 0.0
    assert summary["all_passed"] is True


def test_harmless_refactor_produces_zero_risks(tmp_path, monkeypatch):
    """Verify specifically that harmless refactor triggers 0 false positives."""
    monkeypatch.setattr(storage, "STORAGE_DIR", tmp_path / "patches")
    seed_evaluation_fixtures(tmp_path / "patches")

    harmless_case = EVALUATION_CASES["eval_harmless_refactor"]
    case_result = evaluate_case("eval_harmless_refactor", harmless_case)

    assert case_result["passed"] is True
    assert case_result["detected_risks"] == []
    assert case_result["false_alarm"] is False
    assert case_result["actual_accepted"] is True


def test_validation_removal_triggers_warning(tmp_path, monkeypatch):
    """Verify validation removal is flagged as high severity risk."""
    monkeypatch.setattr(storage, "STORAGE_DIR", tmp_path / "patches")
    seed_evaluation_fixtures(tmp_path / "patches")

    case = EVALUATION_CASES["eval_validation_removal"]
    case_result = evaluate_case("eval_validation_removal", case)

    assert case_result["passed"] is True
    assert "validation_removed" in case_result["detected_risks"]
    assert case_result["actual_accepted"] is False


def test_test_regression_triggers_warning(tmp_path, monkeypatch):
    """Verify test regression is caught and rejected."""
    monkeypatch.setattr(storage, "STORAGE_DIR", tmp_path / "patches")
    seed_evaluation_fixtures(tmp_path / "patches")

    case = EVALUATION_CASES["eval_test_regression"]
    case_result = evaluate_case("eval_test_regression", case)

    assert case_result["passed"] is True
    assert "test_regression" in case_result["detected_risks"]
    assert case_result["actual_accepted"] is False


def test_repeated_edits_triggers_warning(tmp_path, monkeypatch):
    """Verify repeated rewrites trigger repeated_function_edits."""
    monkeypatch.setattr(storage, "STORAGE_DIR", tmp_path / "patches")
    seed_evaluation_fixtures(tmp_path / "patches")

    case = EVALUATION_CASES["eval_repeated_edits"]
    case_result = evaluate_case("eval_repeated_edits", case)

    assert case_result["passed"] is True
    assert "repeated_function_edits" in case_result["detected_risks"]
