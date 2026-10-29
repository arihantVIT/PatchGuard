## Plan: Checkpointing and Recovery

Implement a demonstrable checkpoint lifecycle around PatchGuard's existing immutable version snapshots: record tests for each candidate version, keep the newest passing checkpoint, and restore that snapshot into an explicitly chosen directory only after confirmation and backup.

**Steps**
1. **Define persisted checkpoint metadata.** In `src/patchguard/storage.py`, add per-patch metadata APIs (prefer a manifest such as `data/patches/<patch_id>/checkpoint.json`) to save/retrieve per-version test results and the best checkpoint. Keep version JSON files immutable and backward-compatible. Use the configured behavioral suite identity; only a `passed` result can promote the best pointer. Promote to the newest passing version, not the highest count. Do not promote on failed/error runs; preserve each run's status and pytest output/counts.
2. **Test each version and record the result.** In `src/patchguard/behavioral_testing.py`, retain the existing isolated pytest runner and ensure its result includes explicit passed/failed counts and distinguishes test failure from infrastructure error. In `src/patchguard/pipeline.py` (or a focused version-testing orchestration function there), load one stored version, run the applicable configured suite against it, persist results via storage, and update best checkpoint only on pass. Ensure versions without a supported test suite are reported as `not_run` and cannot become best-tested. Reuse this orchestration from comparisons rather than making compare the only route to tests.
3. **Expose checkpoint workflow in the CLI.** In `src/patchguard/__main__.py`, add commands to test a stored version and display checkpoint/test status. Show test status and current best checkpoint. Keep `compare` compatible; avoid treating its exit status as test pass/fail. Use deterministic flags for running without an LLM.
4. **Implement safe, approved restoration.** Add a restore operation (storage or a focused module, wired from the CLI) that loads a selected stored version and writes its `files` mapping beneath an explicit target directory. Reject unsafe/absolute/traversal paths and non-string source contents. Before writes, show or require an explicit `--confirm`/interactive confirmation; without approval, make no changes. Preserve overwritten files in a backup location and avoid deleting unrelated target files. Restoration does not mutate historical snapshots or silently change the best checkpoint. Clearly report restored paths and backup location.
5. **Build the regression/recovery demonstration.** Add a deliberately regressed version after `calculator_demo/v3` (e.g. `v4` reintroduces negative discounts or returns the wrong calculation) as committed demo fixture data. Ensure the sequence tests v3 successfully, then tests v4 as failed, retains v3 as best, and restores v3 into a temporary chosen directory only after approval. Document exact commands and expected outcome in the currently empty `README.md`; do not overwrite fixture versions or rely on lexical version sorting.
6. **Focused tests and full validation.** Extend `tests/test_storage.py` for metadata persistence and newest-passing selection; add tests for one-version result recording, failed/error non-promotion, restore refusal without confirmation, path traversal rejection, backups, file contents, and preservation of unrelated files. Run focused tests and then the complete suite.

**Relevant files**
- `src/patchguard/storage.py` — existing `save_version`, `load_version`, and `list_versions`; add durable test/checkpoint metadata APIs without changing snapshot immutability.
- `src/patchguard/behavioral_testing.py` — existing `run_behavioral_tests`; make test counts/status explicit and retain isolated pytest behavior.
- `src/patchguard/pipeline.py` — existing `compare_versions`; add reusable per-version testing/recording flow and feed persisted results into comparison reports where applicable.
- `src/patchguard/__main__.py` — existing `list` and `compare` commands; add test/status/restore commands with explicit confirmation.
- `tests/test_storage.py` and new focused checkpoint/restore tests — validate persistence, ranking, approval, validation, and recoverability.
- `data/patches/calculator_demo/` — add a bad follow-up snapshot to demonstrate regression after the passing v3 checkpoint.
- `README.md` — document the exact end-to-end demo and expected best-checkpoint/restore behavior.

**Verification**
1. Run focused tests for storage, checkpoint promotion, and restoration using temporary storage and temporary restore targets; verify no workspace files are modified by tests.
2. Run `PYTHONPATH=src myenv/bin/python -m pytest -q`.
3. Run the documented CLI sequence against `calculator_demo`: test v3 (passes and becomes best), test the regressed version (fails and leaves v3 best), inspect status, attempt restore without approval (no writes), then restore v3 with approval to a temporary directory and inspect restored contents and backup behavior.
4. Verify legacy v1-v3 snapshots still load and `compare calculator_demo v1 v3 --no-llm` still produces a useful report.

**Decisions**
- “Best-tested” means the newest version passing the configured behavioral suite. Failed and infrastructure-error results are recorded but cannot promote a checkpoint.
- Restore writes to an explicitly chosen directory; requires explicit user approval, validates relative paths, backs up overwritten files, and never removes unrelated files.
- Existing snapshots remain immutable; checkpoint/run metadata is stored separately.
- Scope for this deliverable is the existing calculator behavioral suite and a concrete demo. A general user-supplied pytest-suite system is excluded unless existing architecture makes it trivial; unsupported suites must report `not_run` rather than imply a pass.

## Plan: Risk Detection for Suspicious Patch Edits

TL;DR: Add deterministic risk checks alongside the existing compare pipeline so PatchGuard flags suspicious edits with concrete evidence from diffs, function changes, and saved test history before any approval or restore action.

### Steps

1. **Define a structured risk report.** Extend the comparison result in `src/patchguard/pipeline.py` with a `risks` field. Each warning should include a category, severity, file, function, evidence, and message so the CLI and tests can consume the same structure.
2. **Reuse existing diff and function analysis.** Build on `src/patchguard/diff_engine.py` and `src/patchguard/function_tracker.py` to find repeated edits to a function, deleted guard clauses or validation checks, removed exception handling, and logic removed from a later version.
3. **Add historical regression checks.** Read version ordering and saved test outcomes from `src/patchguard/storage.py`. Compare the candidate against earlier passing checkpoints and flag recurring function edits, newly failing tests, or safety checks missing from the last known good version.
4. **Implement the initial detectors.** Cover repeated function edits, validation/exception removal, test regressions, and removal of previously added logic. Keep all checks deterministic and make warnings include evidence from the actual source diff or test history.
5. **Surface warnings in the CLI.** Update `src/patchguard/__main__.py` to print a “Risk Warnings” section for `compare`, including when `--no-llm` is used. Keep warnings concise but show the relevant file/function and a diff excerpt or earlier-version/test context.
6. **Add regression tests.** Extend comparison and CLI tests for repeated edits, removed validation or exception handling, behavioral test regressions, and removed previously added logic. Exercise the stored-version workflow, not only isolated mocks.

### Relevant files

- `src/patchguard/pipeline.py` — build and return the structured risks list.
- `src/patchguard/storage.py` — version history, ordering, and saved test metadata.
- `src/patchguard/diff_engine.py` — unified diff evidence.
- `src/patchguard/function_tracker.py` — function-level change detection.
- `src/patchguard/__main__.py` — CLI display of risk warnings.
- `tests/test_pipeline.py` and `tests/test_cli.py` — end-to-end regression tests.
- `tests/test_diff_engine.py` and `tests/test_function_tracker.py` — lower-level diff and AST behavior tests.

### Verification

1. Run focused pipeline and CLI tests with the project virtual environment and `PYTHONPATH=src`.
2. Confirm the comparison report returns structured risk evidence for each targeted suspicious-edit case.
3. Confirm the CLI prints warnings with `--no-llm`.
4. Confirm a passing prior version followed by a failing candidate produces a warning without making comparison itself fail.

### Decisions and boundaries

- Limit the first implementation to the four deterministic risk patterns above; this is not a general security scanner.
- Risk warnings are additive and do not replace existing diffs, test results, or optional diagnosis.
- Sparse history should result in fewer warnings, not speculative ones.
- Keep warning categories stable for future CLI and automation consumers.
- This phase does not change restore behavior; warnings support review and approval.

## October 24 — Add Groq Explanations for Risk Warnings

Keep PatchGuard's deterministic warnings authoritative, and use the configured Groq endpoint to provide a readable explanation for each important warning. The model must ground its explanation in evidence from the actual diff and detector output.

### Steps

1. Define a structured request and response contract in `src/patchguard/diagnosis.py`. Include a warning ID, category, severity, file/function, version IDs, and identified evidence. Require a structured response with an explanation and references to the supplied evidence.
2. In `src/patchguard/pipeline.py`, send only the relevant diff excerpt, deterministic evidence, and necessary test status/counts. Avoid sending unrelated source or raw test output.
3. Use the existing Groq configuration and endpoint. Handle missing configuration, API errors, timeouts, empty responses, and malformed structured output without losing deterministic warnings.
4. Validate each response before displaying it. Reject unknown warning IDs or evidence references, and verify quoted snippets against the exact evidence submitted. Keep the model explanation advisory; it must not change a warning's category, severity, or evidence.
5. Update `src/patchguard/__main__.py` to show validated explanations and clearly label explanations that are unavailable or rejected. Preserve the deterministic-only `--no-llm` path.
6. Update `README.md` with Groq setup requirements, what comparison data is sent to the service, and how to run without model explanations.
7. Add tests using a mocked Groq client for request configuration and evidence selection, valid and invalid references, malformed responses, API failures, and `--no-llm`. Verify deterministic warnings remain intact in every failure case.
8. Run focused tests and the full suite. Run a real Groq smoke test only with valid configuration and approval to send the selected diff and evidence.

### Relevant files

- `src/patchguard/diagnosis.py` — Groq client, request, and response validation
- `src/patchguard/pipeline.py` — per-warning evidence selection and integration
- `src/patchguard/risk_detection.py` — deterministic findings and evidence
- `src/patchguard/__main__.py` — explanation output
- `tests/test_pipeline.py`, `tests/test_cli.py`, and focused diagnosis tests — coverage
- `README.md` — setup and data-disclosure documentation

### Decisions

- Use the existing configured Groq integration; Nemotron migration is out of scope for this task.
- Model explanations are advisory and cannot create, suppress, or alter deterministic warnings.
- Evidence-reference validation demonstrates traceability to submitted material; it does not prove every semantic claim in generated prose.

## October 25 — Build the Interactive Web Dashboard

A unified, visually polished dark-mode web dashboard allowing judges and developers to inspect patches, risk warnings, test logs, checkpoint histories, and safe restore workflows without reading source code.

### Delivered Components
1. **Interactive Web Dashboard & Server (`src/patchguard/dashboard.py`)**:
   - Zero-external-dependency threaded HTTP server built on Python standard library `http.server`.
   - REST API endpoints for patches, single patch metadata, comparison, live behavioral testing, safe restoration, and evaluation benchmarks.
   - Comprehensive Single Page Application (SPA) UI styled with modern design principles (Plus Jakarta Sans, JetBrains Mono, Linear/Vercel-inspired dark theme).
   - Dynamic Checkpoint Timeline visualizing version sequences, test statuses, and the crown-marked Best Checkpoint.
   - Syntax-highlighted unified diff viewer and AST function tracking badges.
   - Deterministic risk warning cards with severity tags, concrete diff/AST evidence, and Groq advisory explanations.
   - Two-stage Safe Restore modal supporting dry-run preview and confirmed restoration with timestamped pre-overwrite backups.
   - Standalone static export capability (`patchguard dashboard --export dashboard.html`).

## October 26 — Realistic Failure & Harmless Refactor Evaluation Set

Constructed a reproducible benchmark evaluating PatchGuard against real-world AI coding failure modes alongside a harmless refactor:

### Evaluated Cases (`src/patchguard/evaluation.py`)
1. **Validation Removal (`eval_validation_removal`)**:
   - Agent removes input guards; behavioral tests fail, High Severity `validation_removed` and `test_regression` flagged, checkpoint promotion rejected.
2. **Test Regression (`eval_test_regression`)**:
   - Agent flips subtraction to addition; behavioral tests fail, High Severity `test_regression` flagged, checkpoint promotion rejected.
3. **Repeated Function Rewrites (`eval_repeated_edits`)**:
   - High churn on `calculate_total` across v1 -> v2 -> v3 -> v4; Medium Severity `repeated_function_edits` flagged.
4. **Harmless Refactor (`eval_harmless_refactor`)**:
   - Agent cleans code, adds type annotations and docstrings; tests pass 100%, and **ZERO false positive warnings** are emitted.

### Verification
- `patchguard eval` CLI command and API output 100% detection rate and 0.0% false positive rate.
- 62 tests passing across unit, CLI, dashboard, and evaluation test suites.
