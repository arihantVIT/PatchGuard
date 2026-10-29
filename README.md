## Checkpoint and Recovery Demo

The `calculator_demo` snapshots demonstrate a passing checkpoint followed by a regressed patch. The stored sequence metadata explicitly identifies v1 through v4 as save order 1 through 4. The behavioral suite has the stable identity `calculator_behavior_v1`.

From the project root, run:

```bash
export PYTHONPATH=src
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
PYTHON=${PYTHON:-myenv/bin/python}

$PYTHON -m patchguard test calculator_demo v3
$PYTHON -m patchguard test calculator_demo v4
$PYTHON -m patchguard status calculator_demo
```

Expected outcome: v3 passes all three behavioral tests and becomes the best checkpoint. v4 fails because it adds the discount instead of subtracting it; its result is recorded, but v3 remains the best passing checkpoint.

Restore v3 into a separate working directory. The first command previews the required approval behavior and performs no writes:

```bash
TARGET_DIR=$(mktemp -d)
$PYTHON -m patchguard restore calculator_demo v3 "$TARGET_DIR"
```

Approve the restore explicitly, then inspect the restored file:

```bash
$PYTHON -m patchguard restore calculator_demo v3 "$TARGET_DIR" --confirm
cat "$TARGET_DIR/calculator.py"
```

The restore command prints each restored path and the backup directory if it overwrote files. Existing target files are copied to a timestamped directory beside the target before replacement. Unrelated files are preserved. Restore never modifies the immutable stored snapshots or the best-checkpoint pointer.

To compare snapshots without calling the LLM:

```bash
$PYTHON -m patchguard compare calculator_demo v3 v4 --no-llm
```

The compare command's exit status reports whether comparison completed; it does not mean the candidate passed behavioral tests. Inspect the test status in its report or use the `status` command for persisted results.

## NVIDIA Nemotron Risk Explanations

By default, `compare` can ask NVIDIA's hosted Nemotron API to explain deterministic risk warnings. Get an API key from NVIDIA Build and place it in the ignored `.env` file:

```dotenv
NVIDIA_API_KEY=your-nvidia-api-key
NVIDIA_MODEL=nvidia/nemotron-3.5-super-vl-120b-a12b
```

PatchGuard uses the OpenAI-compatible NVIDIA endpoint `https://integrate.api.nvidia.com/v1`. It sends warning categories and severity, file/function and version identifiers, the detector's evidence text, and a short diff excerpt when available. It does not send complete old/new source files or raw pytest output for risk explanations. This evidence is sent to NVIDIA; do not request model explanations for confidential patches unless that disclosure is acceptable.

The model must return structured JSON referencing the evidence IDs included in its request. PatchGuard checks warning IDs, evidence references, and any quoted snippets before displaying the explanation. Explanations are advisory and cannot change the deterministic warning, category, severity, or evidence. If the response is malformed, unsupported, or the request fails, the deterministic warning remains available and the explanation is marked unavailable or rejected.

Use `--no-llm` to avoid the NVIDIA API request and show deterministic findings only:

```bash
$PYTHON -m patchguard compare calculator_demo v3 v4 --no-llm
```

## Dynamic AI Patch-Test-Revise Workflow

The `agent` command works on an arbitrary Python module rather than using the demo's hard-coded calculator tests. It asks NVIDIA Nemotron to generate two pytest suites from the source: regression tests for behavior already present, and acceptance tests for the requested task. Baseline regression tests must pass before patching. The candidate is written to a separate, previously nonexistent output path only after both suites pass. Failed candidate output is fed back to Nemotron for up to three attempts; the original file is never overwritten.

Set `NVIDIA_API_KEY` in `.env` (the default model is `nvidia/nemotron-3.5-super-vl-120b-a12b`; `NVIDIA_MODEL` can override it), then run:

```bash
$PYTHON -m patchguard agent path/to/module.py "Add a parse_duration(text) function supporting seconds and minutes" --output /tmp/module_updated.py
```

Optionally set `--max-attempts 1` through `5`. Review the printed diff before copying or merging the accepted output. This remains a prototype: generated tests may miss requirements, and the local pytest subprocess is not a security sandbox for untrusted source or test code. Source, task text and test output may be sent to NVIDIA; do not use it with confidential code unless that disclosure is acceptable. The command requires an NVIDIA API key and does not silently fall back to weak existence-only tests.

## Running the Agent Workflow From the Dashboard

The dashboard's **Agent Workflow** tab runs the same patch-test-revise workflow without leaving the browser. Paste a module, describe the change, and press **Run Agent Workflow**. The page polls the server once per second and streams each stage: generated regression tests, generated acceptance tests, every patch attempt with its pass/fail counts, and the final decision.

A passing candidate is saved as a new `agent_web_*` patch (original source as `v1`, candidate as `v2`, with the generated regression suite), so it immediately appears in the Patch Inspector with its diff, risks, and checkpoint timeline. A rejected run saves nothing and leaves the snippet untouched.

The endpoints behind the tab are `POST /api/agent/run` and `GET /api/agent/status?job=<id>`. Jobs live in memory, so they disappear when the dashboard server stops.

## Interactive Web Dashboard (October 25)

PatchGuard provides a modern, responsive web dashboard with zero external framework dependencies:

```bash
# Launch interactive web dashboard on http://127.0.0.1:8000
$PYTHON -m patchguard dashboard

# Or run on a custom port without auto-opening a browser
$PYTHON -m patchguard dashboard --port 8080 --no-browser

# Or export a standalone static HTML file for offline viewing
$PYTHON -m patchguard dashboard --export dashboard.html
```

### Dashboard Features for Judges:
- **Patch & Candidate Inspector**: Select any patch and compare baseline (`old_version`) against candidate (`new_version`).
- **Unified Diff & AST Changes**: Color-coded syntax changes alongside AST-level function status badges (`modified`, `added`, `removed`, `unchanged`).
- **Interactive Checkpoint Timeline**: Visual chronological sequence with save order, test execution status, and a star badge identifying the **Best Passing Checkpoint**.
- **Deterministic Risk Signals**: Immediate warnings with severity pills, affected locations, concrete evidence from diffs, and optional NVIDIA Nemotron advisory explanations.
- **Behavioral Test Suite & Terminal Logs**: Complete pytest execution results, pass/fail counts, and live terminal console output.
- **Two-Stage Safe Restore Action**: Dry-run preview requiring explicit confirmation, with automatic timestamped backups before overwriting existing files.
- **Evaluation Showcase Tab**: Dedicated tab demonstrating all 4 realistic failure and refactoring benchmarks with live pass/fail status and false-positive metrics.

## Realistic Agent Failure & Refactor Evaluation Suite (October 26)

Run the automated evaluation benchmark across four realistic agent scenarios:

```bash
$PYTHON -m patchguard eval
```

Or get machine-readable JSON output:

```bash
$PYTHON -m patchguard eval --json
```

### Evaluated Scenarios:
1. **Agent Removing Input Validation (`eval_validation_removal`)**:
   - The agent silently removes input guards (`discount < 0` and `discount > price`).
   - PatchGuard flags High Severity `validation_removed`, behavioral tests fail, and checkpoint promotion is rejected.
2. **Agent Breaking a Previously Passing Test (`eval_test_regression`)**:
   - Validations remain intact, but arithmetic logic is changed from subtraction to addition.
   - PatchGuard catches the regression from the last passing checkpoint, flags High Severity `test_regression`, and refuses promotion.
3. **Agent Repeatedly Rewriting a Function (`eval_repeated_edits`)**:
   - An agent repeatedly modifies `calculate_total` across consecutive iterations (v1 &rarr; v2 &rarr; v3 &rarr; v4).
   - PatchGuard detects code churn and flags Medium Severity `repeated_function_edits`.
4. **Agent Making a Harmless Refactor (`eval_harmless_refactor`)**:
   - The agent adds comprehensive docstrings, explicit type annotations, and cleaner intermediate variable names while preserving validations and logic.
   - **Result**: PatchGuard records 100% test pass and **ZERO false-positive risk warnings**, safely promoting the new checkpoint.

