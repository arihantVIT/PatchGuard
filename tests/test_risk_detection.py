import patchguard.storage as storage
from patchguard.pipeline import compare_versions


def _save(patch_id, version_id, source, path="module.py"):
    storage.save_version(patch_id, version_id, {path: source})


def _categories(report):
    return {risk["category"] for risk in report["risks"]}


def test_repeated_function_edits_include_version_evidence(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(storage, "STORAGE_DIR", tmp_path)
    _save("repeat", "alpha", "def calculate(value):\n    return value\n")
    _save(
        "repeat",
        "beta",
        "def calculate(value):\n    return value + 1\n",
    )
    _save(
        "repeat",
        "gamma",
        "def calculate(value):\n    return value + 2\n",
    )

    report = compare_versions("repeat", "beta", "gamma", run_llm=False)

    warning = next(
        risk
        for risk in report["risks"]
        if risk["category"] == "repeated_function_edits"
    )
    assert warning["function"] == "calculate"
    assert warning["version_ids"] == ["beta", "gamma"]
    assert "beta, gamma" in warning["evidence"]


def test_validation_removal_includes_removed_guard_evidence(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(storage, "STORAGE_DIR", tmp_path)
    _save(
        "validation",
        "before",
        "def calculate(value):\n"
        "    if value < 0:\n"
        "        raise ValueError('negative')\n"
        "    return value\n",
    )
    _save(
        "validation",
        "after",
        "def calculate(value):\n    return value\n",
    )

    report = compare_versions("validation", "before", "after", run_llm=False)

    warning = next(
        risk
        for risk in report["risks"]
        if risk["category"] == "validation_removed"
    )
    assert warning["function"] == "calculate"
    assert "raise ValueError('negative')" in warning["evidence"]


def test_removed_logic_reports_the_removed_statement(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "STORAGE_DIR", tmp_path)
    _save(
        "logic",
        "before",
        "def process(value):\n    return value\n",
    )
    _save(
        "logic",
        "added",
        "def process(value):\n"
        "    audit_log.append(value)\n"
        "    return value\n",
    )
    _save(
        "logic",
        "after",
        "def process(value):\n    return value\n",
    )

    report = compare_versions("logic", "before", "after", run_llm=False)

    warning = next(
        risk
        for risk in report["risks"]
        if risk["category"] == "removed_logic"
    )
    assert warning["function"] == "process"
    assert "audit_log.append(value)" in warning["evidence"]
    assert warning["version_ids"] == ["before", "added", "after"]


def test_test_regression_uses_latest_saved_passing_checkpoint(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(storage, "STORAGE_DIR", tmp_path)
    valid = (
        "def calculate_total(price, discount):\n"
        "    if discount < 0 or discount > price:\n"
        "        raise ValueError()\n"
        "    return price - discount\n"
    )
    invalid = (
        "def calculate_total(price, discount):\n"
        "    return price + discount\n"
    )
    _save("history", "baseline", valid, "calculator.py")
    _save("history", "older_bad", invalid, "calculator.py")
    _save("history", "latest_good", valid, "calculator.py")
    _save("history", "candidate", invalid, "calculator.py")

    storage.record_test_result(
        "history",
        "baseline",
        "calculator_behavior_v1",
        {"status": "passed", "passed_count": 3, "failed_count": 0},
    )
    storage.record_test_result(
        "history",
        "latest_good",
        "calculator_behavior_v1",
        {"status": "passed", "passed_count": 3, "failed_count": 0},
    )

    report = compare_versions(
        "history",
        "older_bad",
        "candidate",
        run_llm=False,
    )

    warning = next(
        risk
        for risk in report["risks"]
        if risk["category"] == "test_regression"
    )
    assert warning["version_ids"] == ["latest_good", "candidate"]
    assert "latest_good (passed" in warning["evidence"]
    assert report["tests"]["original"]["status"] == "failed"
    validation_warning = next(
        risk
        for risk in report["risks"]
        if risk["category"] == "validation_removed"
    )
    assert validation_warning["version_ids"] == ["latest_good", "candidate"]
    assert "raise ValueError()" in validation_warning["evidence"]


def test_validation_warning_includes_unified_diff_evidence(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(storage, "STORAGE_DIR", tmp_path)
    _save(
        "diff",
        "before",
        "def process(value):\n"
        "    if value < 0:\n"
        "        raise ValueError('negative')\n"
        "    return value\n",
    )
    _save("diff", "after", "def process(value):\n    return value\n")

    report = compare_versions("diff", "before", "after", run_llm=False)
    warning = next(
        risk
        for risk in report["risks"]
        if risk["category"] == "validation_removed"
    )

    assert "-    if value < 0:" in warning["diff_evidence"]


def test_test_regression_reports_prior_and_candidate_counts(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(storage, "STORAGE_DIR", tmp_path)
    _save(
        "calculator",
        "passing",
        "def calculate_total(price, discount):\n"
        "    if discount < 0 or discount > price:\n"
        "        raise ValueError()\n"
        "    return price - discount\n",
        "calculator.py",
    )
    _save(
        "calculator",
        "regressed",
        "def calculate_total(price, discount):\n"
        "    if discount < 0 or discount > price:\n"
        "        raise ValueError()\n"
        "    return price + discount\n",
        "calculator.py",
    )

    report = compare_versions(
        "calculator",
        "passing",
        "regressed",
        run_llm=False,
    )

    warning = next(
        risk
        for risk in report["risks"]
        if risk["category"] == "test_regression"
    )
    assert "passing (passed, passed=3, failed=0)" in warning["evidence"]
    assert "regressed (failed" in warning["evidence"]
    assert report["tests"]["candidate"]["status"] == "failed"


def test_cli_prints_risks_without_llm(
    tmp_path,
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(storage, "STORAGE_DIR", tmp_path)
    _save(
        "cli",
        "before",
        "def process(value):\n"
        "    if value < 0:\n"
        "        raise ValueError('negative')\n"
        "    return value\n",
    )
    _save("cli", "after", "def process(value):\n    return value\n")
    monkeypatch.setattr(
        "sys.argv",
        ["patchguard", "compare", "cli", "before", "after", "--no-llm"],
    )

    from patchguard.__main__ import main

    assert main() == 0
    output = capsys.readouterr().out
    assert "--- Risk Warnings ---" in output
    assert "validation_removed" in output
    assert "raise ValueError('negative')" in output
    assert "Diff:" in output
    assert "-    if value < 0:" in output