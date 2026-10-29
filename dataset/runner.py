import json
import os
import subprocess
import sys
import tempfile

from pathlib import Path

from dataset.examples import EXAMPLES


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_FILE = RESULTS_DIR / "test_results.json"


def run_tests(example_name, version_name, code, tests):
    """Run pytest against one version of an implementation."""

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        implementation_file = temp_path / "implementation.py"
        test_file = temp_path / "test_behavior.py"

        implementation_file.write_text(code)
        test_file.write_text(tests)

        env = os.environ.copy()

        # Make pytest and Python able to import implementation.py.
        env["PYTHONPATH"] = str(temp_path) + os.pathsep + env.get(
            "PYTHONPATH", ""
        )

        env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"

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

        return {
            "example": example_name,
            "version": version_name,
            "passed": result.returncode == 0,
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }


def run_example(example_name):
    """Test both original and candidate versions."""

    example = EXAMPLES[example_name]

    original_result = run_tests(
        example_name,
        "original",
        example["original"],
        example["tests"],
    )

    candidate_result = run_tests(
        example_name,
        "candidate",
        example["candidate"],
        example["tests"],
    )

    return {
        "description": example["description"],
        "original": original_result,
        "candidate": candidate_result,
        "candidate_accepted": (
            original_result["passed"] is False
            and candidate_result["passed"] is True
        ),
    }


def main():
    """Run the complete dataset and save a JSON report."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = {}

    for example_name in EXAMPLES:
        print(f"\nRunning: {example_name}")

        result = run_example(example_name)
        results[example_name] = result

        print(
            f"  Original:  "
            f"{'PASS' if result['original']['passed'] else 'FAIL'}"
        )

        print(
            f"  Candidate: "
            f"{'PASS' if result['candidate']['passed'] else 'FAIL'}"
        )

        print(
            f"  Candidate accepted: "
            f"{result['candidate_accepted']}"
        )

    RESULTS_FILE.write_text(
        json.dumps(results, indent=4)
    )

    print(f"\nResults saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()