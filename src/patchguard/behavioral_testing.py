"""Behavioral testing for stored PatchGuard versions."""

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


CALCULATOR_TESTS = """
import pytest
from implementation import calculate_total


def test_normal_calculation():
    assert calculate_total(100, 20) == 80


def test_negative_discount_rejected():
    with pytest.raises(ValueError):
        calculate_total(100, -10)


def test_discount_greater_than_price_rejected():
    with pytest.raises(ValueError):
        calculate_total(100, 120)
"""

CALCULATOR_SUITE_ID = "calculator_behavior_v1"


def _test_counts(output):
    counts = {
        "passed_count": 0,
        "failed_count": 0,
        "error_count": 0,
        "skipped_count": 0,
    }
    summary_lines = [
        line for line in output.splitlines()
        if re.search(r"\b(?:passed|failed|errors?|skipped|xfailed|xpassed)\b", line)
    ]
    if not summary_lines:
        return counts

    summary = summary_lines[-1]
    for count, outcome in re.findall(
        r"(\d+)\s+(passed|failed|errors?|skipped|xfailed|xpassed)",
        summary,
    ):
        outcome = outcome.rstrip("s")
        if outcome == "error":
            counts["error_count"] += int(count)
        elif outcome in {"failed", "passed", "skipped"}:
            counts[f"{outcome}_count"] += int(count)
    return counts


def run_behavioral_tests(
    version_name,
    code=None,
    files=None,
    test_code=None,
    suite_id=None,
):
    """Run behavioral tests against one stored version in an isolated sandbox.

    Supports:
    1. Legacy calculator behavioral suite (code string).
    2. Multi-file snapshots (files dict) containing test_behavior.py.
    3. Explicitly supplied dynamic test code (test_code string).
    """

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Write all snapshot files into sandbox
        if files and isinstance(files, dict):
            for filename, content in files.items():
                if isinstance(content, str):
                    file_path = temp_path / filename
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_path.write_text(content, encoding="utf-8")
                    # Provide implementation.py alias for test suites that import implementation
                    if filename not in ("test_behavior.py", "tests.py") and filename.endswith(".py"):
                        impl_path = temp_path / "implementation.py"
                        if not impl_path.exists():
                            impl_path.write_text(content, encoding="utf-8")

        test_file = temp_path / "test_behavior.py"
        if isinstance(test_code, str):
            test_file.write_text(test_code, encoding="utf-8")
        elif not test_file.exists():
            # Legacy default: calculator tests
            if isinstance(code, str):
                (temp_path / "implementation.py").write_text(code, encoding="utf-8")
            test_file.write_text(CALCULATOR_TESTS, encoding="utf-8")

        env = os.environ.copy()
        env["PYTHONPATH"] = (
            str(temp_path)
            + os.pathsep
            + env.get("PYTHONPATH", "")
        )
        env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    str(test_file),
                    "--tb=short",
                    "-p",
                    "no:cacheprovider",
                ],
                cwd=temp_path,
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )

        except subprocess.TimeoutExpired:
            return {
                "version": version_name,
                "status": "error",
                "passed": False,
                "passed_count": 0,
                "failed_count": 0,
                "error_count": 0,
                "skipped_count": 0,
                "message": "Behavioral tests timed out.",
                "stdout": "",
                "stderr": "",
            }
        except OSError as exc:
            return {
                "version": version_name,
                "status": "error",
                "passed": False,
                "passed_count": 0,
                "failed_count": 0,
                "error_count": 0,
                "skipped_count": 0,
                "message": f"Could not start pytest: {exc}",
                "stdout": "",
                "stderr": str(exc),
            }

        output = result.stdout + "\n" + result.stderr
        counts = _test_counts(output)

        # A failed assertion is different from a broken test environment.
        infrastructure_errors = (
            "ERROR collecting",
            "ImportError while importing test module",
            "ModuleNotFoundError",
            "no tests ran",
        )

        if any(error in output for error in infrastructure_errors):
            status = "error"
        elif result.returncode == 0:
            status = "passed"
        else:
            status = "failed"

        return {
            "version": version_name,
            "status": status,
            "passed": status == "passed",
            **counts,
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
