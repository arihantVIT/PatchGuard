"""Deterministic risk signals for stored patch comparisons."""

import ast

from . import storage
from .diff_engine import generate_diff
from .function_tracker import track_changed_functions


def _functions(source):
    tree = ast.parse(source)
    return {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _statement_evidence(node, source):
    return ast.get_source_segment(source, node) or ast.dump(
        node,
        include_attributes=False,
    )


def _function_statements(function):
    return {
        ast.dump(statement, include_attributes=False): statement
        for statement in function.body
    }


def _diff_excerpt(diff):
    lines = [
        line
        for line in diff.splitlines()
        if line.startswith(("@@", "+", "-"))
        and not line.startswith(("+++", "---"))
    ]
    return "\n".join(lines[:12])


def _historically_removed_logic(
    patch_id,
    old_version_id,
    new_version_id,
    path,
    function_name,
    old_source,
    new_source,
    metadata,
):
    versions = sorted(
        (
            (entry.get("sequence"), version_id)
            for version_id, entry in metadata.get("versions", {}).items()
            if isinstance(entry.get("sequence"), int)
        ),
        key=lambda item: item[0],
    )
    sequence_by_id = {version_id: sequence for sequence, version_id in versions}
    old_sequence = sequence_by_id.get(old_version_id)
    new_sequence = sequence_by_id.get(new_version_id)
    if old_sequence is None or new_sequence is None or old_sequence >= new_sequence:
        return []

    try:
        old_functions = _functions(old_source)
        new_functions = _functions(new_source)
    except SyntaxError:
        return []
    old_function = old_functions.get(function_name)
    new_function = new_functions.get(function_name)
    if old_function is None or new_function is None:
        return []

    baseline = _function_statements(old_function)
    candidate = _function_statements(new_function)
    findings = []
    for sequence, intermediate_id in versions:
        if not old_sequence < sequence < new_sequence:
            continue
        try:
            intermediate_files = storage.load_version(
                patch_id,
                intermediate_id,
            ).get("files", {})
            intermediate_source = intermediate_files.get(path)
            if not isinstance(intermediate_source, str):
                continue
            intermediate_function = _functions(intermediate_source).get(
                function_name
            )
        except (FileNotFoundError, SyntaxError, ValueError):
            continue
        if intermediate_function is None:
            continue

        intermediate = _function_statements(intermediate_function)
        added_then_removed = (
            intermediate.keys() - baseline.keys() - candidate.keys()
        )
        for fingerprint in sorted(added_then_removed):
            statement_evidence = _statement_evidence(
                intermediate[fingerprint],
                intermediate_source,
            )
            diff = generate_diff(
                intermediate_source,
                new_source,
                f"{intermediate_id}/{path}",
                f"{new_version_id}/{path}",
            )
            findings.append({
                "function": function_name,
                "intermediate_version": intermediate_id,
                "evidence": statement_evidence,
                "diff_evidence": _diff_excerpt(diff),
            })
    return findings


def _validation_constructs(function, source):
    constructs = {}
    for node in ast.walk(function):
        guarded_if = isinstance(node, ast.If) and any(
            isinstance(child, (ast.Raise, ast.Assert))
            for child in ast.walk(node)
        )
        has_exception_handler = isinstance(node, ast.Try) and bool(node.handlers)
        is_validation = (
            guarded_if
            or has_exception_handler
            or isinstance(node, (ast.Assert, ast.Raise))
        )
        if is_validation:
            fingerprint = ast.dump(node, include_attributes=False)
            constructs[fingerprint] = _statement_evidence(node, source)
    return constructs


def _repeated_edit_versions(patch_id, candidate_sequence, function_name):
    metadata = storage.load_checkpoint_metadata(patch_id)
    versions = sorted(
        (
            (entry.get("sequence"), version_id)
            for version_id, entry in metadata.get("versions", {}).items()
            if isinstance(entry.get("sequence"), int)
            and entry["sequence"] <= candidate_sequence
        ),
        key=lambda item: item[0],
    )

    edited_versions = []
    for (previous_sequence, previous_id), (_, current_id) in zip(
        versions,
        versions[1:],
    ):
        if previous_sequence is None:
            continue
        try:
            previous_files = storage.load_version(patch_id, previous_id).get(
                "files", {}
            )
            current_files = storage.load_version(patch_id, current_id).get(
                "files", {}
            )
        except (FileNotFoundError, ValueError):
            continue

        for path in sorted(set(previous_files) & set(current_files)):
            previous_source = previous_files[path]
            current_source = current_files[path]
            if not isinstance(previous_source, str) or not isinstance(
                current_source, str
            ):
                continue
            try:
                changes = track_changed_functions(
                    previous_source,
                    current_source,
                )
            except SyntaxError:
                continue
            if function_name in changes["modified"]:
                edited_versions.append(current_id)
                break
    return edited_versions


def _test_regression(patch_id, new_version_id, test_report, metadata):
    candidate = test_report.get("candidate")
    candidate_sequence = metadata.get("versions", {}).get(
        new_version_id,
        {},
    ).get("sequence")
    suite_id = candidate.get("suite_id") if isinstance(candidate, dict) else None
    if not isinstance(candidate, dict) or not isinstance(candidate_sequence, int):
        return None

    latest_passing = _latest_passing_run(
        metadata,
        candidate_sequence,
        suite_id,
    )
    if latest_passing is None:
        return None
    _, baseline_version_id, original = latest_passing

    candidate_regressed = candidate.get("status") == "failed"
    candidate_regressed |= candidate.get("failed_count", 0) > original.get(
        "failed_count", 0
    )
    candidate_regressed |= candidate.get("passed_count", 0) < original.get(
        "passed_count", 0
    )
    if not candidate_regressed:
        return None

    suite = suite_id or "unknown suite"
    evidence = (
        f"{baseline_version_id} ({original.get('status')}, "
        f"passed={original.get('passed_count', 0)}, "
        f"failed={original.get('failed_count', 0)}) -> "
        f"{new_version_id} ({candidate.get('status')}, "
        f"passed={candidate.get('passed_count', 0)}, "
        f"failed={candidate.get('failed_count', 0)})"
    )
    return {
        "category": "test_regression",
        "severity": "high",
        "file": None,
        "function": None,
        "message": f"Behavioral tests regressed for {suite}.",
        "evidence": evidence,
        "version_ids": [baseline_version_id, new_version_id],
    }


def _latest_passing_run(metadata, candidate_sequence, suite_id):
    if not isinstance(candidate_sequence, int) or not suite_id:
        return None
    passing_history = []
    for version_id, version_data in metadata.get("versions", {}).items():
        sequence = version_data.get("sequence")
        if not isinstance(sequence, int) or sequence >= candidate_sequence:
            continue
        for run in version_data.get("test_runs", {}).get(suite_id, []):
            if run.get("status") == "passed":
                passing_history.append((sequence, version_id, run))
    if not passing_history:
        return None
    return max(
        passing_history,
        key=lambda item: item[0],
    )


def detect_risks(
    patch_id,
    old_version_id,
    new_version_id,
    old_files,
    new_files,
    function_changes,
    tests,
    diffs,
):
    """Return evidence-backed risk warnings for one version comparison."""
    risks = []
    metadata = storage.load_checkpoint_metadata(patch_id)
    candidate_metadata = metadata.get("versions", {}).get(new_version_id, {})
    candidate_sequence = candidate_metadata.get("sequence")
    candidate_test = tests.get("candidate")
    suite_id = (
        candidate_test.get("suite_id")
        if isinstance(candidate_test, dict)
        else tests.get("suite")
    )
    passing_checkpoint = _latest_passing_run(
        metadata,
        candidate_sequence,
        suite_id,
    )
    validation_baseline_id = (
        passing_checkpoint[1]
        if passing_checkpoint is not None
        else old_version_id
    )
    try:
        validation_baseline_files = storage.load_version(
            patch_id,
            validation_baseline_id,
        ).get("files", {})
    except (FileNotFoundError, ValueError):
        validation_baseline_files = old_files

    for path in sorted(set(old_files) & set(new_files)):
        old_source = old_files[path]
        new_source = new_files[path]
        if not isinstance(old_source, str) or not isinstance(new_source, str):
            continue
        validation_baseline_source = validation_baseline_files.get(path)
        if not isinstance(validation_baseline_source, str):
            validation_baseline_source = old_source
        try:
            old_functions = _functions(validation_baseline_source)
            new_functions = _functions(new_source)
        except SyntaxError:
            continue

        changes = function_changes.get(path, {})
        changed_names = set(changes.get("modified", []))
        all_common_functions = set(old_functions) & set(new_functions)
        for function_name in sorted(all_common_functions):
            old_function = old_functions.get(function_name)
            new_function = new_functions.get(function_name)
            if old_function is None or new_function is None:
                continue

            old_validation = _validation_constructs(
                old_function,
                validation_baseline_source,
            )
            new_validation = _validation_constructs(new_function, new_source)
            removed_validation = [
                evidence
                for fingerprint, evidence in old_validation.items()
                if fingerprint not in new_validation
            ]
            if removed_validation:
                risks.append({
                    "category": "validation_removed",
                    "severity": "high",
                    "file": path,
                    "function": function_name,
                    "message": "Validation or exception-handling code was removed from the latest passing checkpoint.",
                    "evidence": "\n".join(removed_validation),
                    "diff_evidence": _diff_excerpt(diffs.get(path, "")),
                    "version_ids": [validation_baseline_id, new_version_id],
                })

            historical_removals = _historically_removed_logic(
                patch_id,
                old_version_id,
                new_version_id,
                path,
                function_name,
                old_source,
                new_source,
                metadata,
            )
            for finding in historical_removals:
                risks.append({
                    "category": "removed_logic",
                    "severity": "medium",
                    "file": path,
                    "function": finding["function"],
                    "message": (
                        "Logic added in an intermediate version is absent "
                        "from the candidate."
                    ),
                    "evidence": finding["evidence"],
                    "diff_evidence": finding["diff_evidence"],
                    "version_ids": [
                        old_version_id,
                        finding["intermediate_version"],
                        new_version_id,
                    ],
                })

            if function_name in changed_names and isinstance(candidate_sequence, int):
                edit_versions = _repeated_edit_versions(
                    patch_id,
                    candidate_sequence,
                    function_name,
                )
                if len(edit_versions) >= 2:
                    risks.append({
                        "category": "repeated_function_edits",
                        "severity": "medium",
                        "file": path,
                        "function": function_name,
                        "message": "This function has changed repeatedly across stored versions.",
                        "evidence": (
                            f"Function changed in versions: "
                            f"{', '.join(edit_versions)}"
                        ),
                        "diff_evidence": _diff_excerpt(diffs.get(path, "")),
                        "version_ids": edit_versions,
                    })

    regression = _test_regression(
        patch_id,
        new_version_id,
        tests,
        metadata,
    )
    if regression:
        risks.append(regression)
    return risks