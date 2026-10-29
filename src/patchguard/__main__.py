"""Command-line interface for PatchGuard."""

import argparse
import json
import sys

from .pipeline import compare_versions, test_version
from .restoration import restore_version
from .storage import load_checkpoint_metadata, list_versions


def main():
    parser = argparse.ArgumentParser(
        description="Compare and analyze PatchGuard patch versions."
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    list_parser = subparsers.add_parser(
        "list",
        help="List stored versions of a patch.",
    )
    list_parser.add_argument("patch_id")

    test_parser = subparsers.add_parser(
        "test",
        help="Run the supported behavioral suite for one stored version.",
    )
    test_parser.add_argument("patch_id")
    test_parser.add_argument("version")

    status_parser = subparsers.add_parser(
        "status",
        help="Show saved test results and best checkpoints for a patch.",
    )
    status_parser.add_argument("patch_id")

    restore_parser = subparsers.add_parser(
        "restore",
        help="Restore a stored version to a directory after explicit approval.",
    )
    restore_parser.add_argument("patch_id")
    restore_parser.add_argument("version")
    restore_parser.add_argument("target_dir")
    restore_parser.add_argument(
        "--confirm",
        action="store_true",
        help="Approve writing snapshot files into the target directory.",
    )

    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare two stored versions.",
    )
    compare_parser.add_argument("patch_id")
    compare_parser.add_argument("old_version")
    compare_parser.add_argument("new_version")
    compare_parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip NVIDIA Nemotron risk explanations; show deterministic findings only.",
    )

    dashboard_parser = subparsers.add_parser(
        "dashboard",
        help="Launch the interactive web dashboard or export static HTML.",
    )
    dashboard_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the dashboard web server on (default: 8000).",
    )
    dashboard_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind the dashboard web server on (default: 127.0.0.1).",
    )
    dashboard_parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open the dashboard in a web browser.",
    )
    dashboard_parser.add_argument(
        "--export",
        metavar="PATH",
        help="Export the complete standalone dashboard HTML to a file.",
    )

    eval_parser = subparsers.add_parser(
        "eval",
        help="Run evaluation against realistic failure cases and harmless refactors.",
    )
    eval_parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw benchmark JSON instead of formatted table.",
    )

    agent_parser = subparsers.add_parser(
        "agent-demo",
        help="Run autonomous coding agent demonstration with dynamic Nemotron test generation.",
    )
    agent_parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Run with deterministic verified AST tests instead of the live NVIDIA API.",
    )
    agent_parser.add_argument(
        "--patch-id",
        default="agent_bank_demo",
        help="Patch identifier to store the demonstration under.",
    )
    agent_parser.add_argument(
        "--target",
        help="Target directory to restore best checkpoint into.",
    )

    workflow_parser = subparsers.add_parser(
        "agent",
        help="Generate tests, patch a Python module, and retry until its tests pass.",
    )
    workflow_parser.add_argument("file", help="Python source file to update.")
    workflow_parser.add_argument("task", help="Requested change to implement.")
    workflow_parser.add_argument(
        "--output",
        required=True,
        help="Write the accepted candidate here; must differ from the source file.",
    )
    workflow_parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum patch-and-test attempts (1-5, default: 3).",
    )

    gen_tests_parser = subparsers.add_parser(
        "generate-tests",
        help="Generate verified pytest behavioral tests for any Python source file using NVIDIA Nemotron.",
    )
    gen_tests_parser.add_argument(
        "file",
        help="Path to the Python source file to generate tests for.",
    )
    gen_tests_parser.add_argument(
        "--output",
        "-o",
        help="Optional destination path to write the generated test file.",
    )

    args = parser.parse_args()

    try:
        if args.command == "list":
            versions = list_versions(args.patch_id)

            if not versions:
                print(f"No versions found for '{args.patch_id}'.")
                return 1

            print(f"Versions of {args.patch_id}:")
            for version in versions:
                print(f"  - {version}")

            return 0

        if args.command == "test":
            result = test_version(args.patch_id, args.version)
            print(json.dumps(result, indent=2))
            return 0

        if args.command == "status":
            metadata = load_checkpoint_metadata(args.patch_id)
            print(f"Checkpoint status for {args.patch_id}:")
            print("\nBest checkpoints:")
            if metadata["best_checkpoints"]:
                print(json.dumps(metadata["best_checkpoints"], indent=2))
            else:
                print("  No passing checkpoints recorded.")

            print("\nVersion test history:")
            if not metadata["versions"]:
                print("  No test results recorded.")
            for version_id, version_data in metadata["versions"].items():
                print(f"  {version_id} (sequence: {version_data.get('sequence', 'unknown')})")
                test_runs = version_data.get("test_runs", {})
                if not test_runs:
                    print("    No test runs recorded.")
                for suite_id, runs in test_runs.items():
                    for run in runs:
                        counts = (
                            f"passed={run.get('passed_count', run.get('passed', 0))}, "
                            f"failed={run.get('failed_count', run.get('failed', 0))}, "
                            f"errors={run.get('error_count', 0)}"
                        )
                        print(
                            f"    {suite_id}: {run.get('status')} ({counts}) "
                            f"at {run.get('recorded_at', 'unknown time')}"
                        )
            return 0

        if args.command == "restore":
            result = restore_version(
                args.patch_id,
                args.version,
                args.target_dir,
                confirmed=args.confirm,
            )
            if not result["restored"]:
                print(
                    "Restore not applied. Re-run with --confirm after reviewing "
                    "the selected version and target directory."
                )
                return 2

            print(f"Restored {args.patch_id} {args.version} to {args.target_dir}:")
            for path in result["paths"]:
                print(f"  {path}")
            print(f"Backup: {result['backup_dir'] or 'not needed (no files overwritten)'}")
            return 0

        if args.command == "dashboard":
            from .dashboard import run_dashboard
            return run_dashboard(
                port=args.port,
                host=args.host,
                open_browser=not args.no_browser,
                export_path=args.export,
            )

        if args.command == "eval":
            from .evaluation import run_evaluation, format_evaluation_table
            results = run_evaluation(seed_if_missing=True)
            if args.json:
                print(json.dumps(results, indent=2))
            else:
                print(format_evaluation_table(results))
            return 0 if results["summary"]["all_passed"] else 1

        if args.command == "agent-demo":
            from pathlib import Path
            from .agent import run_agent_demo, format_agent_demo_output
            target = Path(args.target) if args.target else None
            trace = run_agent_demo(
                patch_id=args.patch_id,
                use_llm=not args.no_llm,
                target_dir=target,
            )
            print(format_agent_demo_output(trace))
            return 0

        if args.command == "agent":
            from difflib import unified_diff
            from pathlib import Path
            from .agent import run_patch_test_revise

            result = run_patch_test_revise(
                Path(args.file),
                args.task,
                Path(args.output),
                max_attempts=args.max_attempts,
            )
            print("Generated behavior tests:")
            print(f"  Existing-behavior regression tests: {result['regression_test_count']}")
            print(f"  Task-acceptance tests: {result['acceptance_test_count']}")
            if result["initial_acceptance_status"] == "failed":
                print("  Task tests failed on the original source as expected.")

            for attempt in result["attempts"]:
                regression = attempt["regression"]
                acceptance = attempt["acceptance"]
                print(
                    f"Attempt {attempt['attempt']}: "
                    f"regression={regression['status']} "
                    f"({regression['passed_count']} passed, {regression['failed_count']} failed); "
                    f"acceptance={acceptance['status']} "
                    f"({acceptance['passed_count']} passed, {acceptance['failed_count']} failed)"
                )

            if not result["accepted"]:
                print("No candidate passed both test suites; original source was not changed.")
                return 1

            original = Path(args.file).read_text(encoding="utf-8").splitlines(keepends=True)
            candidate = Path(result["output_path"]).read_text(encoding="utf-8").splitlines(keepends=True)
            print(f"Accepted candidate written to: {result['output_path']}")
            print("\nCandidate diff:")
            print("".join(unified_diff(
                original,
                candidate,
                fromfile=str(Path(args.file)),
                tofile=str(Path(result['output_path'])),
            )), end="")
            return 0

        if args.command == "generate-tests":
            from pathlib import Path
            from .agent import generate_tests_with_nemotron
            src_path = Path(args.file).resolve()
            if not src_path.is_file():
                print(f"Error: File '{src_path}' not found.", file=sys.stderr)
                return 1
            code = src_path.read_text(encoding="utf-8")
            mod_name = src_path.stem
            print(f"Analyzing {src_path.name} and generating verified pytest suite via NVIDIA Nemotron...")
            test_code, from_llm = generate_tests_with_nemotron(code, mod_name, src_path.name)
            if args.output:
                out_path = Path(args.output).resolve()
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(test_code, encoding="utf-8")
                print(f"Verified test suite written to: {out_path} (source: {'NVIDIA Nemotron' if from_llm else 'AST Fallback'})")
            else:
                print("\n# ==================== VERIFIED PYTEST SUITE ====================")
                print(test_code)
            return 0

        report = compare_versions(
            patch_id=args.patch_id,
            old_version_id=args.old_version,
            new_version_id=args.new_version,
            run_llm=not args.no_llm,
        )

        print("\n========== PATCHGUARD REPORT ==========")
        print(
            f"Patch: {report['patch_id']}\n"
            f"Comparison: {report['old_version']} "
            f"-> {report['new_version']}"
        )

        print("\n--- File Changes ---")
        print(f"Added:    {report['added_files']}")
        print(f"Removed:  {report['removed_files']}")
        print(f"Modified: {report['modified_files']}")

        print("\n--- Unified Diffs ---")
        if report["diffs"]:
            for path, diff in report["diffs"].items():
                print(f"\nFile: {path}")
                print(diff)
        else:
            print("No source-code differences found.")

        print("\n--- Function Changes ---")
        print(json.dumps(report["function_changes"], indent=2))

        print("\n--- Test Status ---")
        print(json.dumps(report["tests"], indent=2))

        print("\n--- Risk Warnings ---")
        if report["risks"]:
            for risk in report["risks"]:
                location = risk.get("file") or "test suite"
                if risk.get("function"):
                    location += f"::{risk['function']}"
                print(
                    f"[{risk['severity'].upper()}] {risk['category']} "
                    f"at {location}: {risk['message']}"
                )
                print(f"  Evidence: {risk['evidence']}")
                if risk.get("diff_evidence"):
                    print(f"  Diff:\n{risk['diff_evidence']}")
                print(
                    f"  Versions: {', '.join(risk['version_ids'])}"
                )
                if risk["explanation_status"] == "validated":
                    print(f"  Explanation: {risk['explanation']}")
                    print(
                        "  Explanation evidence: "
                        f"{', '.join(risk['explanation_refs'])}"
                    )
                elif risk["explanation_status"] == "unavailable_or_rejected":
                    print("  Explanation: unavailable or failed evidence validation.")
        else:
            print("No deterministic risk warnings found.")

        if report["diagnosis"]:
            print("\n--- LLM Diagnosis ---")
            print(report["diagnosis"])

        if report["diagnosis_error"]:
            print("\n--- Explanation Status ---")
            print(
                "NVIDIA Nemotron explanations are unavailable or failed evidence validation; "
                "deterministic warnings are unchanged."
            )

        # This exit code indicates comparison errors only.
        # It does not imply that the patch passed behavioral tests.
        return 0

    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"PatchGuard error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())