"""Evaluation suite for realistic failure cases and harmless refactors.

Constructs and evaluates four realistic scenarios:
1. An agent removing input validation (eval_validation_removal).
2. An agent breaking a previously passing test (eval_test_regression).
3. An agent repeatedly rewriting a function (eval_repeated_edits).
4. An agent making a harmless refactor (eval_harmless_refactor).
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from . import storage
from .pipeline import compare_versions, test_version


CALCULATOR_PASSING = """def calculate_total(price, discount):
    if discount < 0:
        raise ValueError("Discount cannot be negative")
    if discount > price:
        raise ValueError("Discount cannot exceed price")
    return price - discount


def format_total(total):
    return f"Total: {total}"
"""

CALCULATOR_NO_VALIDATION = """def calculate_total(price, discount):
    # Agent removed input validations for simplicity
    return price - discount


def format_total(total):
    return f"Total: {total}"
"""

CALCULATOR_BROKEN_LOGIC = """def calculate_total(price, discount):
    if discount < 0:
        raise ValueError("Discount cannot be negative")
    if discount > price:
        raise ValueError("Discount cannot exceed price")
    # Agent accidentally added discount instead of subtracting
    return price + discount


def format_total(total):
    return f"Total: {total}"
"""

CALCULATOR_HARMLESS_REFACTOR = """def calculate_total(price: float, discount: float) -> float:
    \"\"\"Calculate the total price after subtracting the given discount.

    Raises:
        ValueError: If discount is negative or exceeds the base price.
    \"\"\"
    if discount < 0:
        raise ValueError("Discount cannot be negative")
    if discount > price:
        raise ValueError("Discount cannot exceed price")

    discounted_total = price - discount
    return discounted_total


def format_total(total: float) -> str:
    \"\"\"Format the numerical total as a display string.\"\"\"
    return f"Total: {total:.2f}"
"""

EVALUATION_CASES = {
    "eval_validation_removal": {
        "title": "Agent Removing Input Validation",
        "description": (
            "An agent simplifies calculate_total by silently removing negative "
            "and excess discount validation guards. Behavioral tests fail and "
            "a HIGH severity validation_removed risk is flagged."
        ),
        "target_file": "calculator.py",
        "target_function": "calculate_total",
        "versions": {
            "v1": {"calculator.py": CALCULATOR_PASSING},
            "v2": {"calculator.py": CALCULATOR_NO_VALIDATION},
        },
        "compare_pair": ("v1", "v2"),
        "expected_risks": ["validation_removed", "test_regression"],
        "expected_test_status": "failed",
        "expected_candidate_accepted": False,
        "expect_zero_risks": False,
    },
    "eval_test_regression": {
        "title": "Agent Breaking a Previously Passing Test",
        "description": (
            "An agent inverts arithmetic logic (+ instead of -). All input "
            "validations remain intact, but behavioral tests regress and a "
            "HIGH severity test_regression risk is flagged."
        ),
        "target_file": "calculator.py",
        "target_function": "calculate_total",
        "versions": {
            "v1": {"calculator.py": CALCULATOR_PASSING},
            "v2": {"calculator.py": CALCULATOR_BROKEN_LOGIC},
        },
        "compare_pair": ("v1", "v2"),
        "expected_risks": ["test_regression"],
        "expected_test_status": "failed",
        "expected_candidate_accepted": False,
        "expect_zero_risks": False,
    },
    "eval_repeated_edits": {
        "title": "Agent Repeatedly Rewriting a Function",
        "description": (
            "An agent churns on calculate_total across multiple versions "
            "(v1 -> v2 -> v3 -> v4). PatchGuard flags a MEDIUM severity "
            "repeated_function_edits warning noting unstable function history."
        ),
        "target_file": "calculator.py",
        "target_function": "calculate_total",
        "versions": {
            "v1": {
                "calculator.py": (
                    "def calculate_total(price, discount):\n"
                    "    return price - discount\n\n"
                    "def format_total(total):\n"
                    "    return f'Total: {total}'\n"
                )
            },
            "v2": {
                "calculator.py": (
                    "def calculate_total(price, discount):\n"
                    "    if discount < 0:\n"
                    "        raise ValueError('Discount cannot be negative')\n"
                    "    return price - discount\n\n"
                    "def format_total(total):\n"
                    "    return f'Total: {total}'\n"
                )
            },
            "v3": {
                "calculator.py": CALCULATOR_PASSING
            },
            "v4": {
                "calculator.py": (
                    "def calculate_total(price, discount):\n"
                    "    if discount < 0:\n"
                    "        raise ValueError('Discount cannot be negative')\n"
                    "    if discount > price:\n"
                    "        raise ValueError('Discount cannot exceed price')\n"
                    "    # Tweak: ensure float precision\n"
                    "    return float(price - discount)\n\n"
                    "def format_total(total):\n"
                    "    return f'Total: {total}'\n"
                )
            },
        },
        "compare_pair": ("v1", "v4"),
        "expected_risks": ["repeated_function_edits"],
        "expected_test_status": "passed",
        "expected_candidate_accepted": True,
        "expect_zero_risks": False,
    },
    "eval_harmless_refactor": {
        "title": "Agent Making a Harmless Refactor",
        "description": (
            "An agent performs a clean refactor: adds docstrings, explicit type "
            "hints, and cleaner variable naming while preserving all validations "
            "and business logic. PatchGuard triggers ZERO false-positive warnings "
            "and promotes the new passing checkpoint."
        ),
        "target_file": "calculator.py",
        "target_function": "calculate_total",
        "versions": {
            "v1": {"calculator.py": CALCULATOR_PASSING},
            "v2": {"calculator.py": CALCULATOR_HARMLESS_REFACTOR},
        },
        "compare_pair": ("v1", "v2"),
        "expected_risks": [],
        "expected_test_status": "passed",
        "expected_candidate_accepted": True,
        "expect_zero_risks": True,
    },
}


def seed_evaluation_fixtures(base_dir: Path = None):
    """Seed the evaluation fixtures into persistent patch storage."""
    target_dir = base_dir or storage.STORAGE_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    for patch_id, case in EVALUATION_CASES.items():
        patch_dir = target_dir / patch_id
        # If already populated with complete files and checkpoint.json, test and ensure best checkpoint
        if not (patch_dir / "checkpoint.json").exists():
            patch_dir.mkdir(parents=True, exist_ok=True)
            for version_id, files in case["versions"].items():
                storage.save_version(patch_id, version_id, files)
            # Run behavioral tests for each version so checkpoint metadata is filled
            for version_id in case["versions"]:
                test_version(patch_id, version_id)


def evaluate_case(patch_id: str, case_data: Dict[str, Any]) -> Dict[str, Any]:
    """Run evaluation for a single case and return structured metrics."""
    old_ver, new_ver = case_data["compare_pair"]

    # Run comparison without calling external LLM to keep eval fast, deterministic and reproducible
    report = compare_versions(
        patch_id=patch_id,
        old_version_id=old_ver,
        new_version_id=new_ver,
        run_llm=False,
    )

    detected_risk_categories = [r["category"] for r in report["risks"]]
    candidate_test = report["tests"]["candidate"] or {}
    test_status = candidate_test.get("status", "not_run")
    candidate_accepted = report["tests"]["candidate_accepted"]

    expected_risks = case_data["expected_risks"]
    matched_all_risks = all(
        risk in detected_risk_categories for risk in expected_risks
    )
    if case_data["expect_zero_risks"]:
        no_false_positives = len(report["risks"]) == 0
    else:
        no_false_positives = True

    passed_checks = (
        matched_all_risks
        and no_false_positives
        and test_status == case_data["expected_test_status"]
        and candidate_accepted == case_data["expected_candidate_accepted"]
    )

    return {
        "patch_id": patch_id,
        "title": case_data["title"],
        "description": case_data["description"],
        "target_file": case_data["target_file"],
        "target_function": case_data["target_function"],
        "comparison": f"{old_ver} -> {new_ver}",
        "expected_risks": expected_risks,
        "detected_risks": detected_risk_categories,
        "risk_details": report["risks"],
        "expected_test_status": case_data["expected_test_status"],
        "actual_test_status": test_status,
        "expected_accepted": case_data["expected_candidate_accepted"],
        "actual_accepted": candidate_accepted,
        "zero_risk_requirement": case_data["expect_zero_risks"],
        "false_alarm": case_data["expect_zero_risks"] and len(report["risks"]) > 0,
        "passed": passed_checks,
        "diff": report["diffs"].get(case_data["target_file"], ""),
        "function_changes": report["function_changes"].get(
            case_data["target_file"], {}
        ),
    }


def run_evaluation(seed_if_missing: bool = True) -> Dict[str, Any]:
    """Execute all evaluation cases and return a full benchmark summary."""
    if seed_if_missing:
        seed_evaluation_fixtures()

    results = []
    for patch_id, case in EVALUATION_CASES.items():
        res = evaluate_case(patch_id, case)
        results.append(res)

    total_cases = len(results)
    passed_cases = sum(1 for r in results if r["passed"])
    failure_cases = [r for r in results if not r["zero_risk_requirement"]]
    harmless_cases = [r for r in results if r["zero_risk_requirement"]]

    # Detection rate on failure cases
    detected_failures = sum(
        1 for r in failure_cases
        if all(exp in r["detected_risks"] for exp in r["expected_risks"])
    )
    detection_rate = (
        (detected_failures / len(failure_cases) * 100) if failure_cases else 100.0
    )

    # False positive rate on harmless cases
    false_positives = sum(1 for r in harmless_cases if r["false_alarm"])
    false_positive_rate = (
        (false_positives / len(harmless_cases) * 100) if harmless_cases else 0.0
    )

    return {
        "summary": {
            "total_cases": total_cases,
            "passed_cases": passed_cases,
            "detection_rate_pct": detection_rate,
            "false_positive_rate_pct": false_positive_rate,
            "all_passed": passed_cases == total_cases,
        },
        "cases": results,
    }


def format_evaluation_table(eval_result: Dict[str, Any]) -> str:
    """Format evaluation results into a clean, human-readable terminal table."""
    summary = eval_result["summary"]
    lines = [
        "=" * 78,
        "                     PATCHGUARD EVALUATION SUITE                     ",
        "=" * 78,
        f"Total Cases: {summary['total_cases']} | "
        f"Passed: {summary['passed_cases']}/{summary['total_cases']} | "
        f"Detection Rate: {summary['detection_rate_pct']:.1f}% | "
        f"False Positive Rate: {summary['false_positive_rate_pct']:.1f}%",
        "-" * 78,
    ]

    for c in eval_result["cases"]:
        status_badge = "PASS" if c["passed"] else "FAIL"
        lines.append(f"\n[{status_badge}] {c['title']} ({c['patch_id']})")
        lines.append(f"  Comparison : {c['comparison']}")
        lines.append(f"  Target     : {c['target_file']}::{c['target_function']}")
        lines.append(
            f"  Risks Exp  : {c['expected_risks'] if c['expected_risks'] else 'None (0 false-positives)'}"
        )
        lines.append(
            f"  Risks Det  : {c['detected_risks'] if c['detected_risks'] else 'None'}"
        )
        lines.append(
            f"  Test Status: {c['actual_test_status']} (expected {c['expected_test_status']})"
        )
        lines.append(
            f"  Accepted   : {c['actual_accepted']} (expected {c['expected_accepted']})"
        )

    lines.append("\n" + "=" * 78)
    if summary["all_passed"]:
        lines.append("OVERALL: SUCCESS — All realistic agent failure cases & harmless refactor passed!")
    else:
        lines.append("OVERALL: FAILED — Some evaluation criteria were not met.")
    lines.append("=" * 78)
    return "\n".join(lines)
