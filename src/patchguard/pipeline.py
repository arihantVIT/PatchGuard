"""Pipeline for comparing and analyzing stored PatchGuard versions."""

from .behavioral_testing import (
    CALCULATOR_SUITE_ID,
    run_behavioral_tests,
)
from .diff_engine import generate_diff
from .function_tracker import track_changed_functions
from .risk_detection import detect_risks
from .storage import (
    load_version,
    list_versions,
    record_test_result,
)


def test_version(patch_id, version_id):
    """Run and persist the supported behavioral suite for one version."""
    version = load_version(patch_id, version_id)
    files = version.get("files", {})
    if not isinstance(files, dict):
        files = {}

    test_file_content = files.get("test_behavior.py") or files.get("tests.py")
    if isinstance(test_file_content, str):
        suite_id = f"{patch_id}_behavior_suite"
        result = run_behavioral_tests(
            version_id,
            files=files,
            test_code=test_file_content,
            suite_id=suite_id,
        )
        result["suite_id"] = suite_id
        record_test_result(
            patch_id,
            version_id,
            suite_id,
            result,
        )
        return result

    calculator_code = files.get("calculator.py")
    if isinstance(calculator_code, str):
        result = run_behavioral_tests(version_id, calculator_code)
        result["suite_id"] = CALCULATOR_SUITE_ID
        record_test_result(
            patch_id,
            version_id,
            CALCULATOR_SUITE_ID,
            result,
        )
        return result

    return {
        "version": version_id,
        "status": "not_run",
        "passed": False,
        "passed_count": 0,
        "failed_count": 0,
        "error_count": 0,
        "skipped_count": 0,
        "suite_id": None,
        "message": "No supported behavioral test suite for this version.",
    }


def compare_versions(
    patch_id,
    old_version_id,
    new_version_id,
    run_llm=True,
):
    """
    Compare two stored patch versions.

    Returns a report containing source differences, affected functions,
    and an optional LLM diagnosis.

    Test execution is reported separately because stored versions do not
    necessarily contain a runnable test suite.
    """

    # 1. Load both versions.
    old_version = load_version(patch_id, old_version_id)
    new_version = load_version(patch_id, new_version_id)

    old_files = old_version.get("files", {})
    new_files = new_version.get("files", {})

    if not isinstance(old_files, dict) or not isinstance(new_files, dict):
        raise ValueError(
            "Each version must contain a 'files' dictionary."
        )

    # 2. Compare the files in both versions.
    old_paths = set(old_files)
    new_paths = set(new_files)

    added_files = sorted(new_paths - old_paths)
    removed_files = sorted(old_paths - new_paths)
    common_files = sorted(old_paths & new_paths)

    file_diffs = {}
    function_changes = {}

    # Aggregate source code for the diagnosis model.
    original_parts = []
    modified_parts = []

    # 3. Compare files that exist in both versions.
    for path in common_files:
        old_code = old_files[path]
        new_code = new_files[path]

        if not isinstance(old_code, str) or not isinstance(new_code, str):
            raise ValueError(
                f"Source code for '{path}' must be stored as a string."
            )

        original_parts.append(f"# File: {path}\n{old_code}")
        modified_parts.append(f"# File: {path}\n{new_code}")

        diff = generate_diff(
            old_code,
            new_code,
            f"{old_version_id}/{path}",
            f"{new_version_id}/{path}",
        )

        if diff:
            file_diffs[path] = diff

        try:
            function_changes[path] = track_changed_functions(
                old_code,
                new_code,
            )
        except SyntaxError as exc:
            # Preserve the line-level diff even if AST analysis fails.
            function_changes[path] = {
                "error": (
                    f"Cannot analyze Python functions in {path}: {exc}"
                )
            }

    # 4. Handle files added in the new version.
    for path in added_files:
        new_code = new_files[path]

        if not isinstance(new_code, str):
            raise ValueError(
                f"Source code for '{path}' must be stored as a string."
            )

        modified_parts.append(f"# File: {path}\n{new_code}")

        file_diffs[path] = generate_diff(
            "",
            new_code,
            "/dev/null",
            f"{new_version_id}/{path}",
        )

        try:
            function_changes[path] = track_changed_functions(
                "",
                new_code,
            )
        except SyntaxError as exc:
            function_changes[path] = {
                "error": f"Cannot analyze Python functions: {exc}"
            }

    # 5. Handle files removed from the new version.
    for path in removed_files:
        old_code = old_files[path]

        if not isinstance(old_code, str):
            raise ValueError(
                f"Source code for '{path}' must be stored as a string."
            )

        original_parts.append(f"# File: {path}\n{old_code}")

        file_diffs[path] = generate_diff(
            old_code,
            "",
            f"{old_version_id}/{path}",
            "/dev/null",
        )

        try:
            function_changes[path] = track_changed_functions(
                old_code,
                "",
            )
        except SyntaxError as exc:
            function_changes[path] = {
                "error": f"Cannot analyze Python functions: {exc}"
            }

    # 6. Build a combined report.
    report = {
        "patch_id": patch_id,
        "old_version": old_version_id,
        "new_version": new_version_id,
        "available_versions": list_versions(patch_id),
        "added_files": added_files,
        "removed_files": removed_files,
        "modified_files": sorted(
            path
            for path in common_files
            if old_files[path] != new_files[path]
        ),
        "diffs": file_diffs,
        "function_changes": function_changes,
        "tests": {
            "status": "not_run",
            "message": "No compatible behavioral test suite was selected.",
            "original": None,
            "candidate": None,
            "candidate_accepted": False,
        },
        "diagnosis": None,
        "diagnosis_error": None,
    }

    # 7. Run behavioral tests if a suite is available for this patch.
    original_test_result = test_version(patch_id, old_version_id)
    candidate_test_result = test_version(patch_id, new_version_id)

    if (
        original_test_result["status"] != "not_run"
        or candidate_test_result["status"] != "not_run"
    ):
        if (
            original_test_result["status"] == "error"
            or candidate_test_result["status"] == "error"
        ):
            test_status = "error"

        elif (
            original_test_result["status"] == "failed"
            or candidate_test_result["status"] == "failed"
        ):
            test_status = "completed_with_failures"

        elif (
            original_test_result["status"] == "not_run"
            or candidate_test_result["status"] == "not_run"
        ):
            test_status = "not_run"
        else:
            test_status = "completed"

        report["tests"] = {
            "status": test_status,
            "suite": candidate_test_result.get("suite_id") or original_test_result.get("suite_id"),
            "original": original_test_result,
            "candidate": candidate_test_result,
            "candidate_accepted": (
                candidate_test_result["status"] == "passed"
            ),
        }

    report["risks"] = detect_risks(
        patch_id=patch_id,
        old_version_id=old_version_id,
        new_version_id=new_version_id,
        old_files=old_files,
        new_files=new_files,
        function_changes=function_changes,
        tests=report["tests"],
        diffs=file_diffs,
    )

    # 8. Optionally explain deterministic warnings without changing them.
    for index, risk in enumerate(report["risks"], start=1):
        risk["warning_id"] = f"W{index:03d}"
        risk["explanation"] = None
        risk["explanation_status"] = (
            "not_requested" if not run_llm else "pending"
        )

    if run_llm and report["risks"]:
        try:
            from .diagnosis import explain_risks

            explanations = explain_risks(report["risks"])
            for index, risk in enumerate(report["risks"], start=1):
                explanation = explanations[f"W{index:03d}"]
                risk["explanation"] = explanation["explanation"]
                risk["explanation_refs"] = explanation["evidence_refs"]
                risk["explanation_status"] = explanation["validation_status"]
        except Exception as exc:
            # Provider and parsing details can contain sensitive request data.
            report["diagnosis_error"] = type(exc).__name__
            for risk in report["risks"]:
                risk["explanation_status"] = "unavailable_or_rejected"

    return report