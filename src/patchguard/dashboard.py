"""Web Dashboard and API server for PatchGuard.

Provides a rich, interactive web UI to visualize:
- Current and previous patch diffs
- AST function tracking
- Deterministic risk warnings and AI explanations
- Behavioral test results with terminal logs
- Checkpoint timeline with best-checkpoint indicators
- Safe restore workflow with preview and automated backup
- Realistic failure cases evaluation suite
"""

import ast
import json
import mimetypes
import os
from datetime import datetime, timezone
from difflib import unified_diff
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
import shutil
from socketserver import ThreadingMixIn
import sys
import tempfile
import threading
import urllib.parse
import uuid
import webbrowser

from . import storage
from .agent import run_patch_test_revise
from .pipeline import compare_versions, test_version
from .restoration import restore_version
from .evaluation import run_evaluation, seed_evaluation_fixtures


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in separate threads for non-blocking test runs."""
    daemon_threads = True


# In-memory registry for dashboard-launched patch-test-revise runs.
_AGENT_JOBS = {}
_AGENT_JOBS_LOCK = threading.Lock()
AGENT_TERMINAL_STATES = ("accepted", "rejected", "error")


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _agent_job_snapshot(job_id):
    """Return a copy of one job's public state, or None when unknown."""
    with _AGENT_JOBS_LOCK:
        job = _AGENT_JOBS.get(job_id)
        if job is None:
            return None
        snapshot = {key: value for key, value in job.items() if not key.startswith("_")}
        snapshot["stages"] = list(job["stages"])
        snapshot["attempts"] = list(job["attempts"])
        return snapshot


def _record_agent_stage(job_id, event, details):
    with _AGENT_JOBS_LOCK:
        job = _AGENT_JOBS.get(job_id)
        if job is None:
            return
        job["stage"] = event
        job["stages"].append({"event": event, "details": details, "at": _now()})
        if event == "attempt_finished":
            job["attempts"].append(dict(details))


def _store_agent_candidate(filename, source_code, candidate_code, regression_tests):
    """Persist an accepted candidate so it is visible in the patch inspector."""
    patch_id = f"agent_web_{uuid.uuid4().hex[:8]}"
    storage.save_version(
        patch_id,
        "v1",
        {filename: source_code, "test_behavior.py": regression_tests},
    )
    storage.save_version(
        patch_id,
        "v2",
        {filename: candidate_code, "test_behavior.py": regression_tests},
    )
    test_version(patch_id, "v1")
    test_version(patch_id, "v2")
    return patch_id


def _execute_agent_job(job_id, source_code, filename, task, max_attempts):
    """Run the patch-test-revise workflow for a dashboard job."""
    temp_dir = Path(tempfile.mkdtemp(prefix="patchguard-agent-web-"))
    try:
        source_path = temp_dir / filename
        source_path.write_text(source_code, encoding="utf-8")
        output_path = temp_dir / f"candidate_{filename}"

        result = run_patch_test_revise(
            source_path,
            task,
            output_path,
            max_attempts=max_attempts,
            progress=lambda event, details: _record_agent_stage(job_id, event, details),
        )

        update = {
            "finished_at": _now(),
            "regression_test_count": result["regression_test_count"],
            "acceptance_test_count": result["acceptance_test_count"],
            "initial_acceptance_status": result["initial_acceptance_status"],
        }
        if result["accepted"]:
            candidate_code = output_path.read_text(encoding="utf-8")
            update["status"] = "accepted"
            update["candidate_code"] = candidate_code
            update["diff"] = "".join(unified_diff(
                source_code.splitlines(keepends=True),
                candidate_code.splitlines(keepends=True),
                fromfile=filename,
                tofile=f"candidate/{filename}",
            ))
            try:
                update["patch_id"] = _store_agent_candidate(
                    filename,
                    source_code,
                    candidate_code,
                    result["regression_tests"],
                )
            except Exception as exc:
                update["patch_id"] = None
                update["note"] = f"Candidate accepted but could not be saved as a patch: {exc}"
        else:
            update["status"] = "rejected"

        with _AGENT_JOBS_LOCK:
            job = _AGENT_JOBS.get(job_id)
            if job is not None:
                job.update(update)
    except Exception as exc:
        with _AGENT_JOBS_LOCK:
            job = _AGENT_JOBS.get(job_id)
            if job is not None:
                job.update({
                    "status": "error",
                    "error": str(exc),
                    "finished_at": _now(),
                })
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def start_agent_job(source_code, filename, task, max_attempts):
    """Validate input and launch a background agent workflow job."""
    if not isinstance(source_code, str) or not source_code.strip():
        raise ValueError("Source code is required.")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("A task description is required.")
    if not isinstance(filename, str) or not filename.endswith(".py"):
        raise ValueError("Filename must end with .py.")
    if not filename[:-3].isidentifier():
        raise ValueError("Filename must be a valid Python module name.")
    if not isinstance(max_attempts, int) or not 1 <= max_attempts <= 5:
        raise ValueError("max_attempts must be between 1 and 5.")
    try:
        ast.parse(source_code)
    except SyntaxError as exc:
        raise ValueError(f"Source code is not valid Python: {exc}") from exc
    if not os.getenv("NVIDIA_API_KEY"):
        raise ValueError("NVIDIA_API_KEY is not configured in .env.")

    job_id = uuid.uuid4().hex[:12]
    with _AGENT_JOBS_LOCK:
        _AGENT_JOBS[job_id] = {
            "job_id": job_id,
            "status": "running",
            "stage": "queued",
            "task": task.strip(),
            "filename": filename,
            "started_at": _now(),
            "finished_at": None,
            "stages": [],
            "attempts": [],
            "candidate_code": None,
            "diff": None,
            "patch_id": None,
            "error": None,
        }

    thread = threading.Thread(
        target=_execute_agent_job,
        args=(job_id, source_code, filename, task.strip(), max_attempts),
        daemon=True,
    )
    thread.start()
    return job_id


def _json_response(handler, status_code, payload):
    data = json.dumps(payload, indent=2).encode("utf-8")
    handler.send_response(status_code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")
    handler.end_headers()
    handler.wfile.write(data)


def _html_response(handler, html_content):
    data = html_content.encode("utf-8")
    handler.send_response(200)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


class PatchGuardRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        sys.stderr.write(f"[PatchGuard Dashboard] {self.address_string()} - {format % args}\n")

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        # 1. Main Dashboard UI
        if path in ("", "/", "/index.html"):
            seed_evaluation_fixtures()
            html = get_dashboard_html()
            _html_response(self, html)
            return

        # 2. API: List all patches
        if path == "/api/patches":
            seed_evaluation_fixtures()
            patches = []
            if storage.STORAGE_DIR.exists():
                for item in sorted(storage.STORAGE_DIR.iterdir()):
                    if item.is_dir():
                        patch_id = item.name
                        versions = storage.list_versions(patch_id)
                        meta = storage.load_checkpoint_metadata(patch_id)
                        patches.append({
                            "patch_id": patch_id,
                            "versions": versions,
                            "best_checkpoints": meta.get("best_checkpoints", {}),
                            "version_count": len(versions),
                        })
            _json_response(self, 200, {"patches": patches})
            return

        # 3. API: Get details for a single patch
        if path.startswith("/api/patch/"):
            patch_id = path[len("/api/patch/"):]
            try:
                versions = storage.list_versions(patch_id)
                meta = storage.load_checkpoint_metadata(patch_id)
                _json_response(self, 200, {
                    "patch_id": patch_id,
                    "versions": versions,
                    "metadata": meta,
                })
            except Exception as exc:
                _json_response(self, 404, {"error": str(exc)})
            return

        # 4. API: Compare versions
        if path == "/api/compare":
            patch_id = query.get("patch_id", [None])[0]
            old_ver = query.get("old", [None])[0]
            new_ver = query.get("new", [None])[0]
            llm_param = query.get("llm", ["false"])[0].lower() in ("true", "1", "yes")

            if not patch_id or not old_ver or not new_ver:
                _json_response(self, 400, {
                    "error": "Missing required query parameters: patch_id, old, new"
                })
                return

            try:
                report = compare_versions(
                    patch_id=patch_id,
                    old_version_id=old_ver,
                    new_version_id=new_ver,
                    run_llm=llm_param,
                )
                meta = storage.load_checkpoint_metadata(patch_id)
                report["checkpoint_metadata"] = meta
                _json_response(self, 200, report)
            except Exception as exc:
                _json_response(self, 500, {"error": str(exc)})
            return

        # 5. API: Evaluation set benchmarks
        if path == "/api/evaluation":
            try:
                eval_data = run_evaluation(seed_if_missing=True)
                _json_response(self, 200, eval_data)
            except Exception as exc:
                _json_response(self, 500, {"error": str(exc)})
            return

        # 6. API: Live status of a dashboard-launched agent workflow
        if path == "/api/agent/status":
            job_id = query.get("job", [None])[0]
            if not job_id:
                _json_response(self, 400, {"error": "job is required"})
                return
            job = _agent_job_snapshot(job_id)
            if job is None:
                _json_response(self, 404, {"error": f"Unknown agent job: {job_id}"})
                return
            _json_response(self, 200, {"job": job})
            return

        # 404 for unknown endpoints
        _json_response(self, 404, {"error": f"Endpoint not found: {path}"})

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get("Content-Length", 0))
        post_body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        try:
            body = json.loads(post_body.decode("utf-8")) if post_body else {}
        except json.JSONDecodeError:
            _json_response(self, 400, {"error": "Invalid JSON body"})
            return

        # API: Test a version
        if path == "/api/test":
            patch_id = body.get("patch_id")
            version_id = body.get("version_id")
            if not patch_id or not version_id:
                _json_response(self, 400, {"error": "patch_id and version_id are required"})
                return

            try:
                result = test_version(patch_id, version_id)
                meta = storage.load_checkpoint_metadata(patch_id)
                _json_response(self, 200, {
                    "result": result,
                    "metadata": meta,
                })
            except Exception as exc:
                _json_response(self, 500, {"error": str(exc)})
            return

        # API: Restore a version
        if path == "/api/restore":
            patch_id = body.get("patch_id")
            version_id = body.get("version_id")
            target_dir = body.get("target_dir")
            confirm = bool(body.get("confirm", False))

            if not patch_id or not version_id or not target_dir:
                _json_response(self, 400, {
                    "error": "patch_id, version_id, and target_dir are required"
                })
                return

            try:
                res = restore_version(
                    patch_id=patch_id,
                    version_id=version_id,
                    target_dir=target_dir,
                    confirmed=confirm,
                )
                _json_response(self, 200, res)
            except Exception as exc:
                _json_response(self, 400, {"error": str(exc)})
            return

        # API: Launch the patch-test-revise workflow on submitted source
        if path == "/api/agent/run":
            try:
                job_id = start_agent_job(
                    body.get("source_code"),
                    body.get("filename") or "module.py",
                    body.get("task"),
                    body.get("max_attempts", 3),
                )
            except ValueError as exc:
                _json_response(self, 400, {"error": str(exc)})
                return

            _json_response(self, 202, {
                "job_id": job_id,
                "job": _agent_job_snapshot(job_id),
            })
            return

        _json_response(self, 404, {"error": f"Endpoint not found: {path}"})


def _build_offline_bundle():
    """Build pre-computed bundle for offline/standalone file:// browsing."""
    try:
        seed_evaluation_fixtures()
        patches_list = []
        comparisons = {}

        if storage.STORAGE_DIR.exists():
            for item in sorted(storage.STORAGE_DIR.iterdir()):
                if item.is_dir():
                    patch_id = item.name
                    versions = storage.list_versions(patch_id)
                    meta = storage.load_checkpoint_metadata(patch_id)
                    patches_list.append({
                        "patch_id": patch_id,
                        "versions": versions,
                        "best_checkpoints": meta.get("best_checkpoints", {}),
                        "metadata": meta,
                        "version_count": len(versions),
                    })

                    # Store key pairwise comparisons for offline viewing
                    for i in range(len(versions)):
                        for j in range(len(versions)):
                            if i != j or len(versions) == 1:
                                old_v, new_v = versions[i], versions[j]
                                try:
                                    rep = compare_versions(patch_id, old_v, new_v, run_llm=False)
                                    rep["checkpoint_metadata"] = meta
                                    comparisons[f"{patch_id}:{old_v}:{new_v}"] = rep
                                except Exception:
                                    pass

        eval_data = run_evaluation(seed_if_missing=False)
        return {
            "patches": patches_list,
            "comparisons": comparisons,
            "evaluation": eval_data,
        }
    except Exception as exc:
        return {"patches": [], "comparisons": {}, "evaluation": {"summary": {}, "cases": []}}


def get_dashboard_html() -> str:
    """Return the complete standalone Single-Page Application HTML."""
    offline_bundle = json.dumps(_build_offline_bundle())

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PatchGuard &mdash; Patch Safety &amp; Checkpoint Recovery</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg-base: #0B0F17;
      --bg-surface: #111827;
      --bg-card: #162032;
      --bg-card-hover: #1C283F;
      --border-color: #24324D;
      --border-focus: #3B82F6;
      --text-main: #F3F4F6;
      --text-muted: #94A3B8;
      --text-dim: #64748B;
      
      --accent-primary: #6366F1;
      --accent-primary-hover: #4F46E5;
      --accent-glow: rgba(99, 102, 241, 0.25);
      
      --success: #10B981;
      --success-bg: rgba(16, 185, 129, 0.12);
      --success-border: rgba(16, 185, 129, 0.3);
      
      --warning: #F59E0B;
      --warning-bg: rgba(245, 158, 11, 0.12);
      --warning-border: rgba(245, 158, 11, 0.3);
      
      --danger: #EF4444;
      --danger-bg: rgba(239, 68, 68, 0.12);
      --danger-border: rgba(239, 68, 68, 0.3);
      
      --info: #38BDF8;
      --info-bg: rgba(56, 189, 248, 0.12);
      
      --radius-sm: 6px;
      --radius-md: 10px;
      --radius-lg: 16px;
      --font-sans: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
    }}

    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}

    body {{
      background-color: var(--bg-base);
      color: var(--text-main);
      font-family: var(--font-sans);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      overflow-x: hidden;
      line-height: 1.5;
    }}

    /* Scrollbars */
    ::-webkit-scrollbar {{
      width: 8px;
      height: 8px;
    }}
    ::-webkit-scrollbar-track {{
      background: var(--bg-base);
    }}
    ::-webkit-scrollbar-thumb {{
      background: var(--border-color);
      border-radius: 4px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
      background: var(--text-dim);
    }}

    /* Toast Notification System */
    #toast-container {{
      position: fixed;
      top: 1.25rem;
      right: 1.25rem;
      z-index: 99999;
      display: flex;
      flex-direction: column;
      gap: 0.65rem;
      pointer-events: none;
    }}

    .toast {{
      pointer-events: auto;
      min-width: 320px;
      max-width: 480px;
      padding: 0.85rem 1.15rem;
      border-radius: var(--radius-md);
      background: var(--bg-surface);
      border: 1px solid var(--border-color);
      color: var(--text-main);
      font-size: 0.86rem;
      box-shadow: 0 12px 32px rgba(0,0,0,0.6);
      display: flex;
      align-items: center;
      gap: 0.75rem;
      animation: toastSlideIn 0.25s cubic-bezier(0.16, 1, 0.3, 1);
      transition: opacity 0.3s ease, transform 0.3s ease;
    }}

    @keyframes toastSlideIn {{
      from {{ transform: translateX(100%); opacity: 0; }}
      to {{ transform: translateX(0); opacity: 1; }}
    }}

    .toast.success {{ border-color: var(--success); background: #064E3B; color: #ECFDF5; }}
    .toast.error {{ border-color: var(--danger); background: #7F1D1D; color: #FEF2F2; }}
    .toast.info {{ border-color: var(--accent-primary); background: #1E1B4B; color: #EEF2FF; }}
    .toast.warning {{ border-color: var(--warning); background: #78350F; color: #FFFBEB; }}

    /* Top Navigation Header */
    header {{
      background: rgba(17, 24, 39, 0.9);
      backdrop-filter: blur(16px);
      border-bottom: 1px solid var(--border-color);
      position: sticky;
      top: 0;
      z-index: 50;
      padding: 0.85rem 1.75rem;
    }}

    .header-inner {{
      max-width: 1600px;
      margin: 0 auto;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 1.5rem;
    }}

    .brand {{
      display: flex;
      align-items: center;
      gap: 0.85rem;
      text-decoration: none;
      color: inherit;
    }}

    .brand-logo {{
      width: 38px;
      height: 38px;
      background: linear-gradient(135deg, #6366F1, #38BDF8);
      border-radius: var(--radius-sm);
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 0 20px rgba(99, 102, 241, 0.4);
    }}

    .brand-logo svg {{
      width: 22px;
      height: 22px;
      fill: white;
    }}

    .brand-title {{
      font-size: 1.25rem;
      font-weight: 800;
      letter-spacing: -0.02em;
      background: linear-gradient(to right, #FFFFFF, #CBD5E1);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }}

    .brand-tag {{
      font-size: 0.7rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      padding: 0.15rem 0.5rem;
      background: rgba(99, 102, 241, 0.18);
      border: 1px solid rgba(99, 102, 241, 0.4);
      color: #A5B4FC;
      border-radius: 9999px;
    }}

    .nav-tabs {{
      display: flex;
      gap: 0.5rem;
      background: rgba(11, 15, 23, 0.6);
      padding: 0.3rem;
      border-radius: var(--radius-md);
      border: 1px solid var(--border-color);
    }}

    .nav-btn {{
      background: transparent;
      border: none;
      color: var(--text-muted);
      font-family: inherit;
      font-weight: 600;
      font-size: 0.85rem;
      padding: 0.45rem 1rem;
      border-radius: var(--radius-sm);
      cursor: pointer;
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }}

    .nav-btn:hover {{
      color: var(--text-main);
      background: var(--bg-card);
    }}

    .nav-btn.active {{
      color: white;
      background: var(--accent-primary);
      box-shadow: 0 0 12px var(--accent-glow);
    }}

    .header-actions {{
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }}

    /* Mode Banner */
    #mode-banner {{
      display: none;
      background: #78350F;
      color: #FEF3C7;
      border-bottom: 1px solid rgba(245, 158, 11, 0.3);
      padding: 0.4rem 1.5rem;
      font-size: 0.82rem;
      text-align: center;
      font-weight: 500;
    }}

    /* Controls Bar */
    .controls-bar {{
      background: var(--bg-surface);
      border-bottom: 1px solid var(--border-color);
      padding: 1rem 1.75rem;
    }}

    .controls-inner {{
      max-width: 1600px;
      margin: 0 auto;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 1rem;
    }}

    .selectors-group {{
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 1rem;
    }}

    .control-label {{
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-dim);
      display: flex;
      flex-direction: column;
      gap: 0.35rem;
    }}

    select, input[type="text"] {{
      background: var(--bg-base);
      border: 1px solid var(--border-color);
      color: var(--text-main);
      padding: 0.5rem 0.85rem;
      border-radius: var(--radius-sm);
      font-family: inherit;
      font-size: 0.88rem;
      font-weight: 500;
      outline: none;
      transition: border 0.15s ease;
    }}

    select:focus, input[type="text"]:focus {{
      border-color: var(--border-focus);
      box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
    }}

    .btn {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 0.5rem;
      padding: 0.55rem 1.15rem;
      border-radius: var(--radius-sm);
      font-family: inherit;
      font-size: 0.88rem;
      font-weight: 600;
      cursor: pointer;
      border: 1px solid transparent;
      transition: all 0.15s ease;
      user-select: none;
    }}

    .btn-primary {{
      background: var(--accent-primary);
      color: white;
    }}
    .btn-primary:hover {{
      background: var(--accent-primary-hover);
      box-shadow: 0 0 16px var(--accent-glow);
    }}

    .btn-secondary {{
      background: var(--bg-card);
      border-color: var(--border-color);
      color: var(--text-main);
    }}
    .btn-secondary:hover {{
      background: var(--bg-card-hover);
      border-color: var(--text-dim);
    }}

    .btn-success {{
      background: var(--success);
      color: #052E16;
    }}
    .btn-success:hover {{
      background: #059669;
      color: white;
      box-shadow: 0 0 16px rgba(16, 185, 129, 0.3);
    }}

    .btn-danger {{
      background: var(--danger);
      color: white;
    }}
    .btn-danger:hover {{
      background: #DC2626;
      box-shadow: 0 0 16px rgba(239, 68, 68, 0.3);
    }}

    .btn-sm {{
      padding: 0.38rem 0.8rem;
      font-size: 0.8rem;
    }}

    .btn-xs {{
      padding: 0.2rem 0.5rem;
      font-size: 0.74rem;
      border-radius: 4px;
    }}

    .toggle-container {{
      display: flex;
      align-items: center;
      gap: 0.6rem;
      font-size: 0.85rem;
      font-weight: 500;
      color: var(--text-muted);
      cursor: pointer;
      user-select: none;
    }}

    .switch {{
      position: relative;
      display: inline-block;
      width: 40px;
      height: 22px;
    }}
    .switch input {{
      opacity: 0;
      width: 0;
      height: 0;
    }}
    .slider {{
      position: absolute;
      cursor: pointer;
      top: 0; left: 0; right: 0; bottom: 0;
      background-color: var(--border-color);
      transition: .2s;
      border-radius: 34px;
    }}
    .slider:before {{
      position: absolute;
      content: "";
      height: 16px;
      width: 16px;
      left: 3px;
      bottom: 3px;
      background-color: white;
      transition: .2s;
      border-radius: 50%;
    }}
    input:checked + .slider {{
      background-color: var(--accent-primary);
    }}
    input:checked + .slider:before {{
      transform: translateX(18px);
    }}

    /* Main Container */
    main {{
      max-width: 1600px;
      width: 100%;
      margin: 0 auto;
      padding: 1.5rem 1.75rem 3rem;
      display: flex;
      flex-direction: column;
      gap: 1.75rem;
      flex: 1;
    }}

    /* KPI Summary Cards */
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 1rem;
    }}

    .kpi-card {{
      background: var(--bg-surface);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-md);
      padding: 1.15rem 1.35rem;
      display: flex;
      flex-direction: column;
      gap: 0.4rem;
      position: relative;
      overflow: hidden;
      transition: transform 0.2s ease, border-color 0.2s ease;
    }}

    .kpi-card:hover {{
      border-color: var(--border-focus);
      transform: translateY(-2px);
    }}

    .kpi-title {{
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-dim);
    }}

    .kpi-value {{
      font-size: 1.5rem;
      font-weight: 800;
      color: var(--text-main);
      display: flex;
      align-items: baseline;
      gap: 0.5rem;
    }}

    .kpi-badge {{
      font-size: 0.75rem;
      font-weight: 600;
      padding: 0.15rem 0.5rem;
      border-radius: var(--radius-sm);
    }}

    .badge-success {{
      background: var(--success-bg);
      border: 1px solid var(--success-border);
      color: var(--success);
    }}

    .badge-danger {{
      background: var(--danger-bg);
      border: 1px solid var(--danger-border);
      color: var(--danger);
    }}

    .badge-warning {{
      background: var(--warning-bg);
      border: 1px solid var(--warning-border);
      color: var(--warning);
    }}

    .badge-info {{
      background: var(--info-bg);
      border: 1px solid rgba(56, 189, 248, 0.3);
      color: var(--info);
    }}

    /* Checkpoint Timeline Section */
    .section-card {{
      background: var(--bg-surface);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-lg);
      overflow: hidden;
    }}

    .section-header {{
      padding: 1.15rem 1.5rem;
      border-bottom: 1px solid var(--border-color);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 1rem;
      background: rgba(17, 24, 39, 0.4);
    }}

    .section-title {{
      font-size: 1.05rem;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 0.65rem;
      color: var(--text-main);
    }}

    .timeline-container {{
      padding: 2rem 1.5rem 1.5rem;
      overflow-x: auto;
    }}

    .timeline-track {{
      display: flex;
      align-items: center;
      position: relative;
      min-width: 650px;
      padding: 1rem 0;
    }}

    .timeline-line {{
      position: absolute;
      top: 36px;
      left: 3rem;
      right: 3rem;
      height: 3px;
      background: var(--border-color);
      z-index: 1;
    }}

    .timeline-nodes {{
      display: flex;
      justify-content: space-between;
      width: 100%;
      position: relative;
      z-index: 2;
    }}

    .timeline-node {{
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 0.6rem;
      transition: all 0.2s ease;
      padding: 0.5rem;
      border-radius: var(--radius-md);
      position: relative;
    }}

    .timeline-node.active .node-dot {{
      border-color: var(--accent-primary);
      box-shadow: 0 0 20px var(--accent-glow);
      transform: scale(1.08);
    }}

    .node-dot {{
      width: 48px;
      height: 48px;
      border-radius: 50%;
      background: var(--bg-card);
      border: 3px solid var(--border-color);
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 700;
      font-size: 0.92rem;
      transition: all 0.2s ease;
      position: relative;
      cursor: pointer;
    }}

    .node-dot:hover {{
      transform: scale(1.12);
      box-shadow: 0 0 16px var(--accent-glow);
    }}

    .node-dot.passed {{
      border-color: var(--success);
      background: rgba(16, 185, 129, 0.2);
      color: var(--success);
    }}

    .node-dot.failed {{
      border-color: var(--danger);
      background: rgba(239, 68, 68, 0.2);
      color: var(--danger);
    }}

    .node-dot.not-run {{
      border-color: var(--text-dim);
      color: var(--text-muted);
    }}

    .best-crown {{
      position: absolute;
      top: -12px;
      right: -8px;
      background: #F59E0B;
      color: #78350F;
      border-radius: 50%;
      width: 22px;
      height: 22px;
      font-size: 12px;
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 0 12px rgba(245, 158, 11, 0.7);
      border: 2px solid var(--bg-surface);
      font-weight: 800;
    }}

    .node-info {{
      text-align: center;
      display: flex;
      flex-direction: column;
      gap: 0.2rem;
      align-items: center;
    }}

    .node-version {{
      font-weight: 700;
      font-size: 0.95rem;
      color: var(--text-main);
    }}

    .node-seq {{
      font-size: 0.75rem;
      color: var(--text-dim);
    }}

    .node-actions {{
      display: flex;
      gap: 0.35rem;
      margin-top: 0.4rem;
    }}

    /* Split Grid for Inspector */
    .inspector-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1.5rem;
    }}

    @media (max-width: 1024px) {{
      .inspector-grid {{
        grid-template-columns: 1fr;
      }}
    }}

    /* Risk Cards */
    .risks-container {{
      display: flex;
      flex-direction: column;
      gap: 1rem;
      padding: 1.25rem;
    }}

    .risk-card {{
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-md);
      padding: 1.15rem;
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
      border-left: 4px solid var(--text-dim);
      transition: transform 0.15s ease;
    }}

    .risk-card.high {{
      border-left-color: var(--danger);
      background: linear-gradient(to right, rgba(239, 68, 68, 0.05), var(--bg-card) 20%);
    }}

    .risk-card.medium {{
      border-left-color: var(--warning);
      background: linear-gradient(to right, rgba(245, 158, 11, 0.05), var(--bg-card) 20%);
    }}

    .risk-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.75rem;
    }}

    .risk-location {{
      font-family: var(--font-mono);
      font-size: 0.82rem;
      color: var(--info);
    }}

    .risk-msg {{
      font-weight: 600;
      font-size: 0.95rem;
      color: var(--text-main);
    }}

    .risk-evidence-block {{
      background: rgba(11, 15, 23, 0.7);
      border: 1px solid rgba(255, 255, 255, 0.07);
      border-radius: var(--radius-sm);
      padding: 0.65rem 0.85rem;
      font-family: var(--font-mono);
      font-size: 0.8rem;
      color: #E2E8F0;
      white-space: pre-wrap;
      max-height: 140px;
      overflow-y: auto;
    }}

    .risk-llm-explanation {{
      background: rgba(99, 102, 241, 0.08);
      border: 1px solid rgba(99, 102, 241, 0.25);
      border-radius: var(--radius-sm);
      padding: 0.75rem 0.95rem;
      font-size: 0.84rem;
      color: #CBD5E1;
      display: flex;
      flex-direction: column;
      gap: 0.35rem;
    }}

    .risk-llm-title {{
      font-size: 0.72rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: #818CF8;
      display: flex;
      align-items: center;
      gap: 0.4rem;
    }}

    /* Clean Card */
    .clean-state {{
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      text-align: center;
      padding: 3.5rem 1.5rem;
      gap: 1rem;
    }}

    .clean-icon {{
      width: 56px;
      height: 56px;
      border-radius: 50%;
      background: var(--success-bg);
      border: 2px solid var(--success-border);
      color: var(--success);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 24px;
      box-shadow: 0 0 24px rgba(16, 185, 129, 0.2);
    }}

    /* Test Results View */
    .test-summary-bar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 1rem;
      padding: 0.9rem 1.5rem;
      background: rgba(11, 15, 23, 0.5);
      border-bottom: 1px solid var(--border-color);
      flex-wrap: wrap;
    }}

    .terminal-window {{
      background: #06090E;
      padding: 1.25rem;
      font-family: var(--font-mono);
      font-size: 0.82rem;
      color: #94A3B8;
      min-height: 200px;
      max-height: 380px;
      overflow-y: auto;
      white-space: pre-wrap;
      line-height: 1.6;
    }}

    .terminal-window .pass {{ color: var(--success); font-weight: 600; }}
    .terminal-window .fail {{ color: var(--danger); font-weight: 600; }}
    .terminal-window .info {{ color: var(--info); }}

    /* Diff View */
    .diff-block {{
      background: #090D14;
      font-family: var(--font-mono);
      font-size: 0.82rem;
      overflow-x: auto;
      max-height: 440px;
    }}

    .diff-line {{
      display: flex;
      padding: 0.15rem 0.85rem;
      white-space: pre;
    }}

    .diff-add {{
      background: rgba(16, 185, 129, 0.15);
      color: #6EE7B7;
    }}
    .diff-sub {{
      background: rgba(239, 68, 68, 0.15);
      color: #FCA5A5;
    }}
    .diff-hdr {{
      background: rgba(99, 102, 241, 0.12);
      color: #818CF8;
      font-weight: 600;
    }}

    /* Function Badges */
    .functions-list {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      padding: 0.85rem 1.25rem;
      background: rgba(11, 15, 23, 0.4);
      border-bottom: 1px solid var(--border-color);
    }}

    .func-badge {{
      font-family: var(--font-mono);
      font-size: 0.78rem;
      padding: 0.25rem 0.65rem;
      border-radius: var(--radius-sm);
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
    }}
    .func-modified {{
      background: rgba(245, 158, 11, 0.15);
      color: #FCD34D;
      border: 1px solid rgba(245, 158, 11, 0.3);
    }}
    .func-unchanged {{
      background: rgba(100, 116, 139, 0.15);
      color: #94A3B8;
      border: 1px solid rgba(100, 116, 139, 0.25);
    }}
    .func-added {{
      background: rgba(16, 185, 129, 0.15);
      color: #6EE7B7;
      border: 1px solid rgba(16, 185, 129, 0.3);
    }}
    .func-removed {{
      background: rgba(239, 68, 68, 0.15);
      color: #FCA5A5;
      border: 1px solid rgba(239, 68, 68, 0.3);
    }}

    /* Modal Dialog */
    .modal-backdrop {{
      position: fixed;
      top: 0; left: 0; right: 0; bottom: 0;
      background: rgba(0, 0, 0, 0.75);
      backdrop-filter: blur(8px);
      display: none;
      align-items: center;
      justify-content: center;
      z-index: 100;
      padding: 1.5rem;
    }}

    .modal-backdrop.show {{
      display: flex;
    }}

    .modal-card {{
      background: var(--bg-surface);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-lg);
      width: 100%;
      max-width: 580px;
      overflow: hidden;
      box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5);
      animation: modalFadeIn 0.2s ease;
    }}

    @keyframes modalFadeIn {{
      from {{ opacity: 0; transform: scale(0.96); }}
      to {{ opacity: 1; transform: scale(1); }}
    }}

    .modal-header {{
      padding: 1.25rem 1.5rem;
      border-bottom: 1px solid var(--border-color);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }}

    .modal-body {{
      padding: 1.5rem;
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
    }}

    .modal-footer {{
      padding: 1.15rem 1.5rem;
      border-top: 1px solid var(--border-color);
      background: rgba(11, 15, 23, 0.4);
      display: flex;
      justify-content: flex-end;
      gap: 0.75rem;
    }}

    /* Evaluation Grid */
    .eval-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 1.25rem;
    }}

    .eval-card {{
      background: var(--bg-surface);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-md);
      padding: 1.25rem;
      display: flex;
      flex-direction: column;
      gap: 0.85rem;
      position: relative;
    }}

    .eval-card.success-border {{
      border-top: 4px solid var(--success);
    }}

    .eval-card.fail-border {{
      border-top: 4px solid var(--danger);
    }}

    /* Spinner */
    .spinner {{
      border: 2px solid rgba(255, 255, 255, 0.2);
      border-radius: 50%;
      border-top: 2px solid white;
      width: 14px;
      height: 14px;
      animation: spin 0.8s linear infinite;
      display: inline-block;
      vertical-align: middle;
    }}

    @keyframes spin {{
      0% {{ transform: rotate(0deg); }}
      100% {{ transform: rotate(360deg); }}
    }}
  </style>
</head>
<body>

  <!-- Floating Toast Notifications -->
  <div id="toast-container"></div>

  <!-- Offline / Standalone Notice Banner -->
  <div id="mode-banner">
    &#9888; <strong>Standalone File View:</strong> Browsing precomputed snapshots and benchmark results. To run live pytest executions and safe restores, start the server: <code>python -m patchguard dashboard</code>
  </div>

  <!-- Top Navigation Header -->
  <header>
    <div class="header-inner">
      <a href="#" class="brand" onclick="switchTab('explorer')">
        <div class="brand-logo">
          <svg viewBox="0 0 24 24">
            <path d="M12 2L4 5v6.09c0 5.05 3.41 9.76 8 10.91 4.59-1.15 8-5.86 8-10.91V5l-8-3zm0 4.5c1.38 0 2.5 1.12 2.5 2.5s-1.12 2.5-2.5 2.5-2.5-1.12-2.5-2.5 1.12-2.5 2.5-2.5zm0 13c-2.33 0-4.32-1.45-5.12-3.5.03-1.7 3.42-2.63 5.12-2.63 1.69 0 5.09.93 5.12 2.63-.8 2.05-2.79 3.5-5.12 3.5z"/>
          </svg>
        </div>
        <div>
          <div style="display:flex; align-items:center; gap:0.5rem;">
            <span class="brand-title">PatchGuard</span>
            <span class="brand-tag">Autonomous Guardrails</span>
          </div>
        </div>
      </a>

      <div class="nav-tabs">
        <button class="nav-btn active" id="tab-btn-explorer" onclick="switchTab('explorer')">
          <span>Patch Inspector</span>
        </button>
        <button class="nav-btn" id="tab-btn-evaluation" onclick="switchTab('evaluation')">
          <span>Realistic Failure Suite</span>
          <span class="badge-success" style="font-size: 0.68rem; padding: 0.05rem 0.4rem; border-radius: 4px;">4 Cases</span>
        </button>
        <button class="nav-btn" id="tab-btn-agent" onclick="switchTab('agent')">
          <span>Agent Workflow</span>
          <span class="badge-info" style="font-size: 0.68rem; padding: 0.05rem 0.4rem; border-radius: 4px;">Live</span>
        </button>
      </div>

      <div class="header-actions">
        <div id="connection-status-pill" class="kpi-badge badge-success" style="font-size:0.75rem;">
          &#9679; Live Server
        </div>
        <button class="btn btn-primary btn-sm" onclick="openRestoreModal()">
          <span>&#8634; Safe Restore</span>
        </button>
      </div>
    </div>
  </header>

  <!-- Controls Bar -->
  <div class="controls-bar" id="controls-bar">
    <div class="controls-inner">
      <div class="selectors-group">
        <label class="control-label">
          Active Patch Target
          <select id="patch-select" onchange="onPatchChange()">
            <!-- Populated via API -->
          </select>
        </label>

        <label class="control-label">
          Baseline (Previous)
          <select id="old-version-select" onchange="triggerComparison()">
            <!-- Populated via API -->
          </select>
        </label>

        <div style="font-size: 1.2rem; color: var(--text-dim); margin-top: 1rem;">&rarr;</div>

        <label class="control-label">
          Candidate (Current)
          <select id="new-version-select" onchange="triggerComparison()">
            <!-- Populated via API -->
          </select>
        </label>

        <label class="toggle-container" style="margin-top: 1.25rem;">
          <div class="switch">
            <input type="checkbox" id="llm-toggle" onchange="triggerComparison()">
            <span class="slider"></span>
          </div>
          <span>NVIDIA Nemotron Explanations</span>
        </label>
      </div>

      <div style="display: flex; gap: 0.65rem; align-items: center; flex-wrap: wrap;">
        <button class="btn btn-primary btn-sm" onclick="runCandidateTests()" id="run-tests-btn" title="Run behavioral pytest on the selected candidate version">
          <span>&#9654; Run Tests on Candidate</span>
        </button>
        <button class="btn btn-secondary btn-sm" onclick="testAllVersions()" id="test-all-btn" title="Run tests on all versions of this patch sequentially">
          <span>&#9654;&#9654; Test All Checkpoints</span>
        </button>
        <button class="btn btn-secondary btn-sm" onclick="refreshCurrentPatch()" title="Reload data">
          <span>&#8635; Refresh</span>
        </button>
      </div>
    </div>
  </div>

  <!-- Main View Area -->
  <main>
    <!-- TAB 3: LIVE PATCH-TEST-REVISE AGENT WORKFLOW -->
    <div id="view-agent" style="display: none; flex-direction: column; gap: 1.5rem;">
      <div class="section-card" style="padding: 1.5rem;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem; flex-wrap: wrap;">
          <div>
            <h2 style="font-size: 1.4rem; font-weight: 800; color: white;">Live Patch-Test-Revise Agent</h2>
            <p style="font-size: 0.9rem; color: var(--text-muted); margin-top: 0.25rem;">
              Generates regression and acceptance tests, then retries the patch until both suites pass.
              The submitted snippet is never modified; accepted output is saved as a new patch.
            </p>
          </div>
          <button class="btn btn-primary" id="agent-run-btn" onclick="startAgentWorkflow()">
            <span>&#9654; Run Agent Workflow</span>
          </button>
        </div>

        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; margin-top: 1.25rem;">
          <label class="control-label">
            Module Filename
            <input id="agent-filename" value="text_utils.py" spellcheck="false"
              style="width:100%; margin-top:0.35rem; padding:0.5rem 0.65rem; background:rgba(15,23,42,0.85); border:1px solid #1f2937; border-radius:6px; color:#e5e7eb; font-family:var(--font-mono);" />
          </label>
          <label class="control-label">
            Max Attempts
            <input id="agent-attempts" type="number" min="1" max="5" value="3"
              style="width:100%; margin-top:0.35rem; padding:0.5rem 0.65rem; background:rgba(15,23,42,0.85); border:1px solid #1f2937; border-radius:6px; color:#e5e7eb; font-family:var(--font-mono);" />
          </label>
          <label class="control-label" style="grid-column: 1 / -1;">
            Requested Change
            <input id="agent-task" placeholder="Add slugify(text) that lowercases and hyphenates text" spellcheck="false"
              style="width:100%; margin-top:0.35rem; padding:0.5rem 0.65rem; background:rgba(15,23,42,0.85); border:1px solid #1f2937; border-radius:6px; color:#e5e7eb;" />
          </label>
        </div>

        <label class="control-label" style="margin-top: 1rem;">
          Python Source
          <textarea id="agent-source" rows="10" spellcheck="false"
            style="width:100%; margin-top:0.35rem; padding:0.75rem; background:rgba(15,23,42,0.85); border:1px solid #1f2937; border-radius:6px; color:#e5e7eb; font-family:var(--font-mono); font-size:0.85rem;"></textarea>
        </label>
        <div style="display:flex; gap:0.75rem; margin-top:0.75rem; flex-wrap:wrap; align-items:center;">
          <button class="btn btn-secondary btn-sm" onclick="loadAgentSample()">Load Sample Module</button>
          <span style="font-size:0.78rem; color: var(--text-muted);">
            Requires NVIDIA_API_KEY in .env. Source, task, and test output are sent to NVIDIA.
          </span>
        </div>
      </div>

      <div class="section-card" style="padding: 1.5rem;">
        <div style="display:flex; justify-content:space-between; align-items:center; gap:1rem; flex-wrap:wrap;">
          <h3 style="font-weight: 700; color: white;">Live Workflow Progress</h3>
          <div id="agent-status-pill" class="kpi-badge badge-info">IDLE</div>
        </div>
        <div class="terminal-window" id="agent-log" style="margin-top:1rem; max-height:280px; overflow:auto;"># Workflow events stream here</div>
        <div id="agent-result"></div>
      </div>
    </div>

    <!-- TAB 1: INSPECTOR & COMPARE -->
    <div id="view-explorer" style="display: flex; flex-direction: column; gap: 1.5rem;">
      
      <!-- KPI Status Overview -->
      <div class="kpi-grid">
        <div class="kpi-card">
          <span class="kpi-title">Current Candidate Status</span>
          <div class="kpi-value" id="kpi-candidate-status">
            <span>&mdash;</span>
          </div>
          <span style="font-size: 0.8rem; color: var(--text-muted);" id="kpi-candidate-sub">Select version</span>
        </div>

        <div class="kpi-card">
          <span class="kpi-title">Best Passing Checkpoint</span>
          <div class="kpi-value" id="kpi-best-checkpoint">
            <span>&mdash;</span>
          </div>
          <span style="font-size: 0.8rem; color: var(--text-muted);" id="kpi-best-sub">Safe rollback pointer</span>
        </div>

        <div class="kpi-card">
          <span class="kpi-title">Deterministic Risk Signals</span>
          <div class="kpi-value" id="kpi-risks">
            <span>0</span>
          </div>
          <span style="font-size: 0.8rem; color: var(--text-muted);" id="kpi-risks-sub">Safety warnings</span>
        </div>

        <div class="kpi-card">
          <span class="kpi-title">Behavioral Test Suite</span>
          <div class="kpi-value" id="kpi-tests">
            <span>&mdash;</span>
          </div>
          <span style="font-size: 0.8rem; color: var(--text-muted);" id="kpi-tests-sub">Suite execution</span>
        </div>
      </div>

      <!-- Checkpoint Timeline -->
      <div class="section-card">
        <div class="section-header">
          <div class="section-title">
            <span>&#9201; Checkpoint &amp; Version Timeline</span>
          </div>
          <div style="display:flex; align-items:center; gap:0.75rem;">
            <button class="btn btn-secondary btn-xs" onclick="testAllVersions()">
              <span>&#9654; Test All Versions</span>
            </button>
            <span style="font-size: 0.78rem; color: var(--text-muted);">
              Click any node to select
            </span>
          </div>
        </div>
        <div class="timeline-container">
          <div class="timeline-track">
            <div class="timeline-line"></div>
            <div class="timeline-nodes" id="timeline-nodes">
              <!-- Timeline nodes populated dynamically -->
            </div>
          </div>
        </div>
      </div>

      <!-- Inspector Split View -->
      <div class="inspector-grid">
        
        <!-- Left Column: Source Diff & AST Analysis -->
        <div class="section-card">
          <div class="section-header">
            <div class="section-title">
              <span>&lt;/&gt; Unified Code Diff</span>
            </div>
            <div id="diff-file-badge" class="kpi-badge badge-info">calculator.py</div>
          </div>

          <!-- Function Tracking Summary -->
          <div class="functions-list" id="functions-list">
            <!-- Badges populated dynamically -->
          </div>

          <!-- Unified Diff Lines -->
          <div class="diff-block" id="diff-content">
            <div style="padding: 2rem; color: var(--text-dim); text-align: center;">
              Select patch versions to view code changes.
            </div>
          </div>
        </div>

        <!-- Right Column: Risk Indicators & Behavioral Tests -->
        <div style="display: flex; flex-direction: column; gap: 1.5rem;">
          
          <!-- Risk Warnings Card -->
          <div class="section-card">
            <div class="section-header">
              <div class="section-title">
                <span>&#9888;&#65039; Risk Indicators</span>
                <span id="risk-count-badge" class="kpi-badge badge-warning" style="display:none;">0</span>
              </div>
              <span style="font-size: 0.78rem; color: var(--text-dim);">Deterministic diff &amp; AST analysis</span>
            </div>

            <div class="risks-container" id="risks-container">
              <!-- Risk items populated dynamically -->
            </div>
          </div>

          <!-- Behavioral Test Runner Output -->
          <div class="section-card">
            <div class="section-header">
              <div class="section-title">
                <span>&#10003; Behavioral Test Results</span>
              </div>
              <div style="display: flex; align-items: center; gap: 0.65rem;">
                <div id="test-status-pill" class="kpi-badge badge-info">PENDING</div>
                <button class="btn btn-success btn-xs" onclick="runCandidateTests()" id="card-run-test-btn">
                  <span>&#9654; Run Tests Now</span>
                </button>
              </div>
            </div>

            <div class="test-summary-bar" id="test-counts-bar">
              <span style="font-size: 0.85rem; color: var(--text-muted);">Awaiting comparison data...</span>
            </div>

            <div class="terminal-window" id="test-console-output">
# Pytest console log will appear here
            </div>
          </div>

        </div>

      </div>

    </div>

    <!-- TAB 2: REALISTIC EVALUATION SUITE -->
    <div id="view-evaluation" style="display: none; flex-direction: column; gap: 1.5rem;">
      <div class="section-card" style="padding: 1.5rem; background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(17, 24, 39, 0.9));">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
          <div>
            <h2 style="font-size: 1.4rem; font-weight: 800; color: white;">Realistic Agent Failure &amp; Refactor Benchmark</h2>
            <p style="font-size: 0.9rem; color: var(--text-muted); margin-top: 0.25rem;">
              Demonstrating detection of critical agent anti-patterns while proving 0% false positives on harmless refactoring.
            </p>
          </div>
          <button class="btn btn-primary" onclick="loadEvaluationBenchmarks()" id="run-eval-btn">
            <span>&#8635; Re-run Evaluation Suite</span>
          </button>
        </div>

        <div class="kpi-grid" style="margin-top: 1.5rem;">
          <div class="kpi-card" style="background: rgba(17, 24, 39, 0.7);">
            <span class="kpi-title">Detection Rate (Failure Modes)</span>
            <div class="kpi-value" style="color: var(--success);" id="eval-kpi-detect">100%</div>
            <span style="font-size: 0.78rem; color: var(--text-muted);">Validation drop, test break, function churn</span>
          </div>
          <div class="kpi-card" style="background: rgba(17, 24, 39, 0.7);">
            <span class="kpi-title">False Positive Rate (Harmless Refactor)</span>
            <div class="kpi-value" style="color: var(--success);" id="eval-kpi-fpr">0.0%</div>
            <span style="font-size: 0.78rem; color: var(--text-muted);">Zero alarms on comments, types &amp; cleanups</span>
          </div>
          <div class="kpi-card" style="background: rgba(17, 24, 39, 0.7);">
            <span class="kpi-title">Checkpoint Safety</span>
            <div class="kpi-value" style="color: var(--info);" id="eval-kpi-safety">Guarded</div>
            <span style="font-size: 0.78rem; color: var(--text-muted);">Regressed patches cannot be promoted</span>
          </div>
        </div>
      </div>

      <!-- Evaluation Cases Grid -->
      <div class="eval-grid" id="eval-cases-grid">
        <!-- Filled dynamically -->
      </div>
    </div>
  </main>

  <!-- Restore Modal Dialog -->
  <div class="modal-backdrop" id="restore-modal">
    <div class="modal-card">
      <div class="modal-header">
        <h3 style="font-weight: 700; font-size: 1.1rem; color: white;">Safe Checkpoint Restoration</h3>
        <button style="background:none; border:none; color:var(--text-dim); cursor:pointer; font-size:1.2rem;" onclick="closeRestoreModal()">&times;</button>
      </div>

      <div class="modal-body">
        <div style="background: rgba(99, 102, 241, 0.08); border: 1px solid rgba(99, 102, 241, 0.2); padding: 0.85rem; border-radius: var(--radius-sm); font-size: 0.84rem; color: #CBD5E1;">
          PatchGuard enforces two-stage restore: previews safety checks, validates path traversal, and creates automatic timestamped backups before overwriting existing files.
        </div>

        <label class="control-label">
          Version to Restore
          <select id="modal-restore-version">
            <!-- options -->
          </select>
        </label>

        <label class="control-label">
          Target Destination Directory
          <input type="text" id="modal-restore-target" value="/tmp/patchguard-restored-demo">
        </label>

        <!-- Preview feedback box -->
        <div id="restore-feedback-box" style="display:none; padding: 0.85rem; border-radius: var(--radius-sm); font-size: 0.82rem; font-family: var(--font-mono);">
        </div>
      </div>

      <div class="modal-footer">
        <button class="btn btn-secondary btn-sm" onclick="closeRestoreModal()">Cancel</button>
        <button class="btn btn-secondary btn-sm" onclick="executeRestore(false)" id="modal-preview-btn">1. Dry-Run Preview</button>
        <button class="btn btn-success btn-sm" onclick="executeRestore(true)" id="modal-confirm-btn">2. Confirm &amp; Restore</button>
      </div>
    </div>
  </div>

  <script>
    // Embedded Offline Fallback Data
    const OFFLINE_DATA = {offline_bundle};

    // State
    let isOfflineMode = (window.location.protocol === "file:");
    let currentPatches = [];
    let activePatchId = null;
    let activePatchMeta = null;
    let comparisonReport = null;

    // Toast Utility
    function showToast(message, type = "info") {{
      const container = document.getElementById("toast-container");
      const toast = document.createElement("div");
      toast.className = `toast ${{type}}`;
      
      const icon = type === "success" ? "&#10003;" : (type === "error" ? "&#10007;" : (type === "warning" ? "&#9888;" : "&#8505;"));
      toast.innerHTML = `<span style="font-weight:700; font-size:1.1rem;">${{icon}}</span><span>${{message}}</span>`;
      
      container.appendChild(toast);
      setTimeout(() => {{
        toast.style.opacity = "0";
        toast.style.transform = "translateX(50px)";
        setTimeout(() => toast.remove(), 300);
      }}, 4000);
    }}

    // LIVE AGENT WORKFLOW
    const NL = String.fromCharCode(10);
    let agentJobId = null;
    let agentPollHandle = null;

    const AGENT_EVENT_LABELS = {{
      regression_tests_ready: "Generated regression tests",
      acceptance_tests_ready: "Generated acceptance tests",
      attempt_started: "Attempt started",
      attempt_finished: "Attempt finished",
      accepted: "Candidate accepted",
      rejected: "No candidate passed"
    }};

    function loadAgentSample() {{
      document.getElementById("agent-filename").value = "text_utils.py";
      document.getElementById("agent-source").value = `def normalize_spaces(text):
    '''Collapse repeated whitespace and trim the ends.'''
    return " ".join(text.split())
`;
      document.getElementById("agent-task").value =
        "Add slugify(text) that lowercases text, replaces runs of non-alphanumeric " +
        "characters with a single hyphen, and trims hyphens from both ends. " +
        "Preserve normalize_spaces behavior.";
      showToast("Sample module loaded. Adjust the task, then run the workflow.", "info");
    }}

    function setAgentPill(label, badgeClass) {{
      const pill = document.getElementById("agent-status-pill");
      pill.className = `kpi-badge ${{badgeClass}}`;
      pill.textContent = label;
    }}

    function resetAgentButton() {{
      const btn = document.getElementById("agent-run-btn");
      btn.disabled = false;
      btn.innerHTML = '<span>&#9654; Run Agent Workflow</span>';
    }}

    async function startAgentWorkflow() {{
      if (isOfflineMode) {{
        showToast("The agent workflow needs the live dashboard server.", "warning");
        return;
      }}

      const source = document.getElementById("agent-source").value;
      const task = document.getElementById("agent-task").value.trim();
      const filename = document.getElementById("agent-filename").value.trim() || "module.py";
      const attempts = parseInt(document.getElementById("agent-attempts").value, 10) || 3;

      if (!task) {{ showToast("Describe the requested change first.", "warning"); return; }}
      if (!source.trim()) {{ showToast("Add Python source to patch.", "warning"); return; }}

      const btn = document.getElementById("agent-run-btn");
      btn.disabled = true;
      btn.innerHTML = '<span class="spinner"></span> Running...';
      document.getElementById("agent-result").innerHTML = "";
      document.getElementById("agent-log").textContent = "# Starting workflow...";
      setAgentPill("RUNNING", "badge-warning");

      try {{
        const res = await fetch("/api/agent/run", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify({{
            source_code: source,
            filename: filename,
            task: task,
            max_attempts: attempts
          }})
        }});
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Could not start the workflow");

        agentJobId = data.job_id;
        renderAgentJob(data.job);
        if (agentPollHandle) clearInterval(agentPollHandle);
        agentPollHandle = setInterval(pollAgentJob, 1000);
      }} catch (err) {{
        setAgentPill("ERROR", "badge-danger");
        document.getElementById("agent-log").textContent = `# ${{err.message}}`;
        resetAgentButton();
      }}
    }}

    async function pollAgentJob() {{
      if (!agentJobId) return;
      try {{
        const res = await fetch(`/api/agent/status?job=${{encodeURIComponent(agentJobId)}}`, {{ cache: "no-store" }});
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Status unavailable");

        renderAgentJob(data.job);
        if (["accepted", "rejected", "error"].includes(data.job.status)) {{
          clearInterval(agentPollHandle);
          agentPollHandle = null;
          resetAgentButton();
        }}
      }} catch (err) {{
        clearInterval(agentPollHandle);
        agentPollHandle = null;
        setAgentPill("ERROR", "badge-danger");
        document.getElementById("agent-log").textContent += NL + `# ${{err.message}}`;
        resetAgentButton();
      }}
    }}

    function formatAgentDetails(details) {{
      if (!details) return "";
      const parts = [];
      if (details.attempt) parts.push(`attempt ${{details.attempt}}`);
      if (details.status) parts.push(details.status);
      if (details.passed_count !== undefined) parts.push(`${{details.passed_count}} passed`);
      if (details.failed_count) parts.push(`${{details.failed_count}} failed`);
      if (details.regression) {{
        parts.push(`regression=${{details.regression.status}} ` +
          `(${{details.regression.passed_count}}/${{details.regression.passed_count + details.regression.failed_count}})`);
      }}
      if (details.acceptance) {{
        parts.push(`acceptance=${{details.acceptance.status}} ` +
          `(${{details.acceptance.passed_count}}/${{details.acceptance.passed_count + details.acceptance.failed_count}})`);
      }}
      if (details.generation_error) parts.push(details.generation_error);
      return parts.length ? " - " + parts.join(", ") : "";
    }}

    function renderAgentJob(job) {{
      if (!job) return;

      const pillClass = job.status === "accepted" ? "badge-success"
        : job.status === "running" ? "badge-warning" : "badge-danger";
      setAgentPill(job.status.toUpperCase(), pillClass);

      const lines = [
        `# Job ${{job.job_id}} - ${{job.status}}`,
        `# Module ${{job.filename}}`,
        `# Task ${{job.task}}`,
        ""
      ];
      (job.stages || []).forEach(stage => {{
        const label = AGENT_EVENT_LABELS[stage.event] || stage.event;
        lines.push(`[${{stage.at.replace("T", " ")}}] ${{label}} ${{formatAgentDetails(stage.details)}}`);
      }});
      if (job.status === "running") lines.push("", "# Waiting for the next event...");

      const log = document.getElementById("agent-log");
      log.textContent = lines.join(NL);
      log.scrollTop = log.scrollHeight;

      const resultEl = document.getElementById("agent-result");
      if (job.status === "accepted") {{
        const openButton = job.patch_id
          ? `<button class="btn btn-success btn-sm" onclick="openAgentPatch('${{job.patch_id}}')">Open in Patch Inspector</button>`
          : `<span style="font-size:0.8rem; color:var(--text-muted);">${{escapeHtml(job.note || "")}}</span>`;
        resultEl.innerHTML = `
          <div style="display:flex; gap:0.75rem; align-items:center; margin-top:1rem; flex-wrap:wrap;">
            ${{openButton}}
            <span style="font-size:0.8rem; color:var(--text-muted);">
              Passed ${{job.regression_test_count}} regression and ${{job.acceptance_test_count}} acceptance tests.
            </span>
          </div>
          <h4 style="margin-top:1.25rem; color:white;">Candidate Diff</h4>
          <div class="diff-block" style="max-height:260px; overflow:auto;">${{formatDiff(job.diff)}}</div>
          <h4 style="margin-top:1.25rem; color:white;">Candidate Source</h4>
          <pre class="terminal-window" style="max-height:320px; overflow:auto; white-space:pre-wrap;">${{escapeHtml(job.candidate_code)}}</pre>
        `;
      }} else if (job.status === "rejected") {{
        resultEl.innerHTML = `<p style="margin-top:1rem; color:var(--text-muted); font-size:0.85rem;">
          No candidate passed both suites, so nothing was saved. See the attempt log above.</p>`;
      }} else if (job.status === "error") {{
        resultEl.innerHTML = `<p style="margin-top:1rem; color:#f87171; font-size:0.85rem;">
          ${{escapeHtml(job.error || "Workflow failed.")}}</p>`;
      }} else {{
        resultEl.innerHTML = "";
      }}
    }}

    async function openAgentPatch(patchId) {{
      switchTab("explorer");
      try {{
        const res = await fetch("/api/patches", {{ cache: "no-store" }});
        const data = await res.json();
        currentPatches = data.patches || [];
        renderPatchDropdown();
        if (currentPatches.some(p => p.patch_id === patchId)) {{
          document.getElementById("patch-select").value = patchId;
          await onPatchChange();
          showToast(`Loaded ${{patchId}} in the patch inspector.`, "success");
        }}
      }} catch (err) {{
        showToast("Could not refresh the patch list.", "warning");
      }}
    }}

    // Initialize
    document.addEventListener("DOMContentLoaded", async () => {{
      await initDashboard();
    }});

    async function initDashboard() {{
      const statusPill = document.getElementById("connection-status-pill");
      const modeBanner = document.getElementById("mode-banner");

      if (window.location.protocol === "file:") {{
        isOfflineMode = true;
        statusPill.className = "kpi-badge badge-warning";
        statusPill.textContent = "● Offline File View";
        modeBanner.style.display = "block";
        showToast("Opened via file:// &mdash; Embedded offline data activated.", "warning");
        loadOfflinePatches();
        return;
      }}

      try {{
        const res = await fetch("/api/patches", {{ cache: "no-store" }});
        if (!res.ok) throw new Error("Server returned non-200");
        const data = await res.json();
        currentPatches = data.patches || [];

        statusPill.className = "kpi-badge badge-success";
        statusPill.textContent = "● Live Server Connected";
        modeBanner.style.display = "none";
        renderPatchDropdown();
      }} catch (err) {{
        console.warn("Backend API unreachable, activating offline fallback:", err);
        isOfflineMode = true;
        statusPill.className = "kpi-badge badge-warning";
        statusPill.textContent = "● Standalone View";
        modeBanner.style.display = "block";
        loadOfflinePatches();
      }}
    }}

    function loadOfflinePatches() {{
      currentPatches = OFFLINE_DATA.patches || [];
      renderPatchDropdown();
    }}

    function renderPatchDropdown() {{
      const select = document.getElementById("patch-select");
      select.innerHTML = "";

      const demoGroup = document.createElement("optgroup");
      demoGroup.label = "Demonstrations";
      const evalGroup = document.createElement("optgroup");
      evalGroup.label = "Evaluation Failure Cases";

      currentPatches.forEach(p => {{
        const opt = document.createElement("option");
        opt.value = p.patch_id;
        opt.textContent = `${{p.patch_id}} (${{p.version_count}} versions)`;
        if (p.patch_id.startsWith("eval_")) {{
          evalGroup.appendChild(opt);
        }} else {{
          demoGroup.appendChild(opt);
        }}
      }});

      if (demoGroup.children.length) select.appendChild(demoGroup);
      if (evalGroup.children.length) select.appendChild(evalGroup);

      if (currentPatches.length > 0) {{
        activePatchId = currentPatches[0].patch_id;
        select.value = activePatchId;
        onPatchChange();
      }}
    }}

    function switchTab(tabId) {{
      document.querySelectorAll(".nav-btn").forEach(b => b.classList.remove("active"));
      document.getElementById(`tab-btn-${{tabId}}`).classList.add("active");

      ["explorer", "evaluation", "agent"].forEach(id => {{
        const view = document.getElementById(`view-${{id}}`);
        if (view) view.style.display = id === tabId ? "flex" : "none";
      }});

      document.getElementById("controls-bar").style.display = tabId === "explorer" ? "block" : "none";

      if (tabId === "evaluation") loadEvaluationBenchmarks();
    }}

    async function onPatchChange(preserveOld = null, preserveNew = null) {{
      activePatchId = document.getElementById("patch-select").value;
      if (!activePatchId) return;

      let versions = [];
      if (isOfflineMode) {{
        const pObj = currentPatches.find(p => p.patch_id === activePatchId);
        activePatchMeta = pObj ? pObj.metadata : {{}};
        versions = pObj ? pObj.versions : [];
      }} else {{
        try {{
          const res = await fetch(`/api/patch/${{activePatchId}}`);
          const data = await res.json();
          activePatchMeta = data.metadata;
          versions = data.versions || [];
        }} catch (err) {{
          console.error("Failed to load patch details:", err);
          return;
        }}
      }}

      // Populate selects
      const oldSelect = document.getElementById("old-version-select");
      const newSelect = document.getElementById("new-version-select");
      oldSelect.innerHTML = "";
      newSelect.innerHTML = "";

      versions.forEach(v => {{
        const o1 = document.createElement("option");
        o1.value = v; o1.textContent = v;
        const o2 = document.createElement("option");
        o2.value = v; o2.textContent = v;
        oldSelect.appendChild(o1);
        newSelect.appendChild(o2);
      }});

      // Handle version selection
      if (preserveOld && versions.includes(preserveOld)) {{
        oldSelect.value = preserveOld;
      }} else if (versions.length >= 2) {{
        oldSelect.value = versions[versions.length - 2];
      }} else if (versions.length === 1) {{
        oldSelect.value = versions[0];
      }}

      if (preserveNew && versions.includes(preserveNew)) {{
        newSelect.value = preserveNew;
      }} else if (versions.length >= 2) {{
        newSelect.value = versions[versions.length - 1];
      }} else if (versions.length === 1) {{
        newSelect.value = versions[0];
      }}

      updateCandidateButtonLabel();
      renderTimeline(versions, activePatchMeta);
      await triggerComparison();
    }}

    function updateCandidateButtonLabel() {{
      const candidateVer = document.getElementById("new-version-select")?.value || "";
      const runBtn = document.getElementById("run-tests-btn");
      const cardBtn = document.getElementById("card-run-test-btn");
      if (runBtn) runBtn.innerHTML = `<span>&#9654; Run Tests on Candidate (${{candidateVer}})</span>`;
      if (cardBtn) cardBtn.innerHTML = `<span>&#9654; Run Tests (${{candidateVer}})</span>`;
    }}

    function renderTimeline(versions, metadata) {{
      const container = document.getElementById("timeline-nodes");
      container.innerHTML = "";

      const bestMap = metadata.best_checkpoints || {{}};
      const bestVerId = Object.values(bestMap)[0]?.version_id;
      const currentCandidate = document.getElementById("new-version-select").value;

      versions.forEach(v => {{
        const vData = metadata.versions?.[v] || {{}};
        const seq = vData.sequence || "-";
        
        // Determine status
        let status = "not-run";
        let statusText = "Not Run";
        const runs = vData.test_runs ? Object.values(vData.test_runs).flat() : [];
        if (runs.length > 0) {{
          const lastRun = runs[runs.length - 1];
          status = lastRun.status;
          statusText = status === "passed" ? "PASSED" : "FAILED";
        }}

        const isBest = v === bestVerId;

        const node = document.createElement("div");
        node.className = `timeline-node ${{v === currentCandidate ? "active" : ""}}`;

        node.innerHTML = `
          <div class="node-dot ${{status}}" onclick="selectCandidateVersion('${{v}}')" title="Click to inspect ${{v}}">
            ${{v}}
            ${{isBest ? '<div class="best-crown" title="Best Passing Checkpoint">&#9733;</div>' : ''}}
          </div>
          <div class="node-info">
            <div class="node-version">${{v}}</div>
            <div class="node-seq">Seq #${{seq}}</div>
            <span class="kpi-badge ${{status === 'passed' ? 'badge-success' : (status === 'failed' ? 'badge-danger' : 'badge-info')}}" style="font-size:0.68rem; margin-top:0.2rem;">
              ${{isBest ? 'BEST CHECKPOINT' : statusText}}
            </span>
            <div class="node-actions">
              <button class="btn btn-secondary btn-xs" onclick="runCandidateTests('${{v}}')" title="Run pytest on ${{v}}">&#9654; Test</button>
              <button class="btn btn-secondary btn-xs" onclick="selectCandidateVersion('${{v}}')" title="Inspect as candidate">View</button>
            </div>
          </div>
        `;
        container.appendChild(node);
      }});
    }}

    function selectCandidateVersion(v) {{
      const newSelect = document.getElementById("new-version-select");
      if (newSelect) {{
        newSelect.value = v;
        updateCandidateButtonLabel();
        triggerComparison();
        showToast(`Selected ${{v}} as comparison candidate.`, "info");
      }}
    }}

    async function triggerComparison() {{
      const oldVer = document.getElementById("old-version-select").value;
      const newVer = document.getElementById("new-version-select").value;
      const llmChecked = document.getElementById("llm-toggle").checked;

      if (!activePatchId || !oldVer || !newVer) return;
      updateCandidateButtonLabel();

      // Highlight active node
      document.querySelectorAll(".timeline-node").forEach(n => {{
        n.classList.remove("active");
        if (n.querySelector(".node-version")?.textContent === newVer) {{
          n.classList.add("active");
        }}
      }});

      if (isOfflineMode) {{
        const key = `${{activePatchId}}:${{oldVer}}:${{newVer}}`;
        const rep = OFFLINE_DATA.comparisons?.[key];
        if (rep) {{
          comparisonReport = rep;
          renderReport(rep);
        }} else {{
          // Generic fallback
          renderReport({{
            patch_id: activePatchId,
            old_version: oldVer,
            new_version: newVer,
            diffs: {{}},
            function_changes: {{}},
            tests: {{ status: "not_run", candidate: {{ status: "not_run" }} }},
            risks: []
          }});
        }}
        return;
      }}

      try {{
        const url = `/api/compare?patch_id=${{activePatchId}}&old=${{oldVer}}&new=${{newVer}}&llm=${{llmChecked}}`;
        const res = await fetch(url);
        const report = await res.json();
        comparisonReport = report;
        renderReport(report);
      }} catch (err) {{
        console.error("Comparison failed:", err);
      }}
    }}

    function renderReport(report) {{
      const testReport = report.tests || {{}};
      const candidateTest = testReport.candidate || {{}};
      const candidateAccepted = testReport.candidate_accepted;

      const kpiStatus = document.getElementById("kpi-candidate-status");
      const kpiSub = document.getElementById("kpi-candidate-sub");
      if (candidateAccepted) {{
        kpiStatus.innerHTML = `<span style="color: var(--success);">&#10003; ACCEPTED</span>`;
        kpiSub.textContent = "Candidate passed all behavioral tests";
      }} else if (candidateTest.status === "failed") {{
        kpiStatus.innerHTML = `<span style="color: var(--danger);">&#10007; REJECTED</span>`;
        kpiSub.textContent = "Candidate failed behavioral tests";
      }} else {{
        kpiStatus.innerHTML = `<span style="color: var(--text-muted);">${{candidateTest.status || 'NOT RUN'}}</span>`;
        kpiSub.textContent = "No supported test suite run";
      }}

      // Best Checkpoint
      const meta = report.checkpoint_metadata || activePatchMeta || {{}};
      const bestObj = Object.values(meta.best_checkpoints || {{}})[0];
      const kpiBest = document.getElementById("kpi-best-checkpoint");
      if (bestObj) {{
        kpiBest.innerHTML = `<span style="color: #F59E0B;">&#9733; ${{bestObj.version_id}}</span>`;
      }} else {{
        kpiBest.innerHTML = `<span>None</span>`;
      }}

      // Risks KPI
      const risks = report.risks || [];
      const kpiRisks = document.getElementById("kpi-risks");
      if (risks.length === 0) {{
        kpiRisks.innerHTML = `<span style="color: var(--success);">0 (Clean)</span>`;
      }} else {{
        kpiRisks.innerHTML = `<span style="color: var(--danger);">${{risks.length}} Detected</span>`;
      }}

      // Tests KPI
      const kpiTests = document.getElementById("kpi-tests");
      if (candidateTest.passed) {{
        kpiTests.innerHTML = `<span style="color: var(--success);">${{candidateTest.passed_count}} Passed</span>`;
      }} else if (candidateTest.status === "failed") {{
        kpiTests.innerHTML = `<span style="color: var(--danger);">${{candidateTest.failed_count}} Failed</span>`;
      }} else {{
        kpiTests.innerHTML = `<span>${{candidateTest.status || 'Not Run'}}</span>`;
      }}

      // Diff View
      const diffContainer = document.getElementById("diff-content");
      const diffs = report.diffs || {{}};
      const firstFile = Object.keys(diffs)[0] || "calculator.py";
      document.getElementById("diff-file-badge").textContent = firstFile;
      if (diffs[firstFile]) {{
        diffContainer.innerHTML = formatDiff(diffs[firstFile]);
      }} else {{
        diffContainer.innerHTML = `<div style="padding: 2.5rem; color: var(--text-dim); text-align: center;">No source differences between ${{report.old_version}} and ${{report.new_version}}.</div>`;
      }}

      // Function changes
      const funcList = document.getElementById("functions-list");
      funcList.innerHTML = "";
      const funcChanges = (report.function_changes && report.function_changes[firstFile]) || {{}};
      
      (funcChanges.modified || []).forEach(f => {{
        funcList.innerHTML += `<span class="func-badge func-modified">&#9679; ${{f}} (modified)</span>`;
      }});
      (funcChanges.added || []).forEach(f => {{
        funcList.innerHTML += `<span class="func-badge func-added">+ ${{f}} (added)</span>`;
      }});
      (funcChanges.removed || []).forEach(f => {{
        funcList.innerHTML += `<span class="func-badge func-removed">- ${{f}} (removed)</span>`;
      }});
      (funcChanges.unchanged || []).forEach(f => {{
        funcList.innerHTML += `<span class="func-badge func-unchanged">&#10003; ${{f}} (unchanged)</span>`;
      }});

      // Risks View
      const risksContainer = document.getElementById("risks-container");
      const riskBadge = document.getElementById("risk-count-badge");
      if (risks.length === 0) {{
        riskBadge.style.display = "none";
        risksContainer.innerHTML = `
          <div class="clean-state">
            <div class="clean-icon">&#10003;</div>
            <div>
              <h4 style="font-size: 1.05rem; font-weight: 700; color: white;">All Safety Checks Clean</h4>
              <p style="font-size: 0.85rem; color: var(--text-muted); margin-top: 0.35rem; max-width: 440px;">
                No input validation removal, test regression, dropped intermediate logic, or function thrashing detected.
              </p>
            </div>
          </div>
        `;
      }} else {{
        riskBadge.style.display = "inline-block";
        riskBadge.textContent = `${{risks.length}} Warnings`;
        risksContainer.innerHTML = "";
        risks.forEach(r => {{
          const isHigh = r.severity === "high";
          const loc = r.file ? `${{r.file}}::${{r.function || '*'}}` : (r.category || 'System');
          
          let llmHtml = "";
          if (r.explanation_status === "validated" && r.explanation) {{
            llmHtml = `
              <div class="risk-llm-explanation">
                <span class="risk-llm-title">&#10024; Nemotron Advisory Explanation</span>
                <div>${{escapeHtml(r.explanation)}}</div>
                <div style="font-size: 0.72rem; color: var(--text-dim); margin-top: 0.25rem;">
                  Evidence verified: ${{ (r.explanation_refs || []).join(", ") }}
                </div>
              </div>
            `;
          }}

          risksContainer.innerHTML += `
            <div class="risk-card ${{isHigh ? 'high' : 'medium'}}">
              <div class="risk-header">
                <span class="kpi-badge ${{isHigh ? 'badge-danger' : 'badge-warning'}}">
                  [${{r.severity.toUpperCase()}}] ${{r.category}}
                </span>
                <span class="risk-location">${{loc}}</span>
              </div>
              <div class="risk-msg">${{escapeHtml(r.message)}}</div>
              ${{r.evidence ? `<div class="risk-evidence-block">${{escapeHtml(r.evidence)}}</div>` : ''}}
              ${{r.diff_evidence ? `<div class="risk-evidence-block" style="color:#A7F3D0;">${{escapeHtml(r.diff_evidence)}}</div>` : ''}}
              ${{llmHtml}}
            </div>
          `;
        }});
      }}

      // Test results output
      renderTestBox(candidateTest, testReport.suite);
    }}

    function renderTestBox(testResult, suiteName) {{
      const pill = document.getElementById("test-status-pill");
      const countsBar = document.getElementById("test-counts-bar");
      const consoleOut = document.getElementById("test-console-output");

      const status = testResult?.status || "not_run";
      if (status === "passed") {{
        pill.className = "kpi-badge badge-success";
        pill.textContent = "PASSED";
      }} else if (status === "failed") {{
        pill.className = "kpi-badge badge-danger";
        pill.textContent = "FAILED";
      }} else {{
        pill.className = "kpi-badge badge-info";
        pill.textContent = status.toUpperCase();
      }}

      const passed = testResult?.passed_count ?? 0;
      const failed = testResult?.failed_count ?? 0;
      const errors = testResult?.error_count ?? 0;

      countsBar.innerHTML = `
        <div style="display:flex; align-items:center; gap:0.65rem;">
          <span style="font-weight:600; color:var(--text-main);">Suite: ${{suiteName || testResult?.suite_id || 'calculator_behavior_v1'}}</span>
          <span>&bull;</span>
          <span style="color:var(--success); font-weight:600;">Passed: ${{passed}}</span>
          <span>&bull;</span>
          <span style="color:var(--danger); font-weight:600;">Failed: ${{failed}}</span>
          <span>&bull;</span>
          <span>Errors: ${{errors}}</span>
        </div>
      `;

      let output = (testResult?.stdout || "") + (testResult?.stderr ? "\\n" + testResult.stderr : "");
      if (!output) output = "No test run recorded for this version yet. Click 'Run Tests Now' above to execute.";
      consoleOut.textContent = output;
    }}

    function formatDiff(diffText) {{
      if (!diffText) return "";
      const lines = diffText.split("\\n");
      return lines.map(line => {{
        let cls = "";
        if (line.startsWith("+") && !line.startsWith("+++")) cls = "diff-add";
        else if (line.startsWith("-") && !line.startsWith("---")) cls = "diff-sub";
        else if (line.startsWith("@@")) cls = "diff-hdr";
        return `<div class="diff-line ${{cls}}">${{escapeHtml(line)}}</div>`;
      }}).join("");
    }}

    function escapeHtml(str) {{
      if (!str) return "";
      return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }}

    // RUN TESTS INTERACTIVE ACTION
    async function runCandidateTests(targetVersion) {{
      const ver = targetVersion || document.getElementById("new-version-select").value;
      if (!activePatchId || !ver) {{
        showToast("Please select a patch and version first.", "warning");
        return;
      }}

      showToast(`Running behavioral tests for ${{activePatchId}} (${{ver}})...`, "info");
      
      const consoleOut = document.getElementById("test-console-output");
      const testPill = document.getElementById("test-status-pill");
      testPill.className = "kpi-badge badge-warning";
      testPill.textContent = "RUNNING...";
      consoleOut.textContent = `======================= RUNNING PYTEST ISOLATED SUITE =======================\\nTarget  : ${{activePatchId}} :: ${{ver}}\\nSuite   : calculator_behavior_v1\\nSandbox : /tmp/pytest-patchguard-sandbox/...\\nCommand : pytest -q test_behavior.py --tb=short\\n-----------------------------------------------------------------------------\\nSpawning pytest subprocess...\\n`;

      const btn = document.getElementById("run-tests-btn");
      const cardBtn = document.getElementById("card-run-test-btn");
      if (btn) {{ btn.innerHTML = `<span class="spinner"></span> Running...`; btn.disabled = true; }}
      if (cardBtn) {{ cardBtn.innerHTML = `<span class="spinner"></span> Running...`; cardBtn.disabled = true; }}

      try {{
        if (isOfflineMode) {{
          // Simulated execution in offline view
          await new Promise(r => setTimeout(r, 600));
          const pObj = currentPatches.find(p => p.patch_id === activePatchId);
          const runs = pObj?.metadata?.versions?.[ver]?.test_runs?.calculator_behavior_v1 || [];
          const lastRun = runs[runs.length - 1] || {{
            status: ver === "v4" || activePatchId.includes("regression") || activePatchId.includes("removal") ? "failed" : "passed",
            passed_count: ver === "v4" ? 2 : 3,
            failed_count: ver === "v4" ? 1 : 0,
            stdout: ver === "v4" ? "F..\\n1 failed, 2 passed in 0.01s" : "...\\n3 passed in 0.01s",
            stderr: ""
          }};
          renderTestBox(lastRun, "calculator_behavior_v1");
          showToast(`Tests finished: ${{lastRun.status.toUpperCase()}} (${{lastRun.passed_count}} passed, ${{lastRun.failed_count}} failed)`, lastRun.status === "passed" ? "success" : "error");
        }} else {{
          const res = await fetch("/api/test", {{
            method: "POST",
            headers: {{"Content-Type": "application/json"}},
            body: JSON.stringify({{patch_id: activePatchId, version_id: ver}})
          }});
          const data = await res.json();
          if (data.error) throw new Error(data.error);

          activePatchMeta = data.metadata;
          renderTestBox(data.result, "calculator_behavior_v1");
          
          // Re-render timeline to display updated badge and best-checkpoint star
          const patchData = currentPatches.find(p => p.patch_id === activePatchId);
          const versions = patchData ? patchData.versions : Object.keys(activePatchMeta.versions || {{}});
          renderTimeline(versions, activePatchMeta);

          const isPass = data.result.status === "passed";
          showToast(
            `Tests finished: ${{data.result.passed_count}} passed, ${{data.result.failed_count}} failed (${{data.result.status.toUpperCase()}})`,
            isPass ? "success" : "error"
          );

          // Update comparison with preserved version
          const oldVer = document.getElementById("old-version-select").value;
          await triggerComparison();
        }}
      }} catch (err) {{
        showToast(`Test error: ${{err.message || err}}`, "error");
        consoleOut.textContent += `\\nError executing tests: ${{err.message || err}}`;
        testPill.className = "kpi-badge badge-danger";
        testPill.textContent = "ERROR";
      }} finally {{
        updateCandidateButtonLabel();
        if (btn) btn.disabled = false;
        if (cardBtn) cardBtn.disabled = false;
      }}
    }}

    // RUN ALL TESTS INTERACTIVE ACTION
    async function testAllVersions() {{
      if (!activePatchId) return;
      const patchData = currentPatches.find(p => p.patch_id === activePatchId);
      const versions = patchData ? patchData.versions : Object.keys(activePatchMeta?.versions || {{}});
      if (!versions.length) return;

      showToast(`Testing all ${{versions.length}} versions for ${{activePatchId}}...`, "info");
      const btn = document.getElementById("test-all-btn");
      if (btn) {{ btn.innerHTML = `<span class="spinner"></span> Testing All...`; btn.disabled = true; }}

      for (const ver of versions) {{
        await runCandidateTests(ver);
        await new Promise(r => setTimeout(r, 200));
      }}

      if (btn) {{ btn.innerHTML = `<span>&#9654;&#9654; Test All Checkpoints</span>`; btn.disabled = false; }}
      showToast(`Completed test suite across all versions of ${{activePatchId}}!`, "success");
    }}

    function refreshCurrentPatch() {{
      const oldV = document.getElementById("old-version-select")?.value;
      const newV = document.getElementById("new-version-select")?.value;
      onPatchChange(oldV, newV);
      showToast("Refreshed patch data.", "info");
    }}

    // Modal Restore
    function openRestoreModal() {{
      const modal = document.getElementById("restore-modal");
      const sel = document.getElementById("modal-restore-version");
      sel.innerHTML = "";

      const versions = activePatchMeta ? Object.keys(activePatchMeta.versions || {{}}) : [];
      const bestObj = Object.values(activePatchMeta?.best_checkpoints || {{}})[0];
      const bestId = bestObj?.version_id;

      versions.forEach(v => {{
        const opt = document.createElement("option");
        opt.value = v;
        opt.textContent = v === bestId ? `${{v}} ★ (Recommended Best Checkpoint)` : v;
        sel.appendChild(opt);
      }});

      if (bestId) sel.value = bestId;

      document.getElementById("restore-feedback-box").style.display = "none";
      modal.classList.add("show");
    }}

    function closeRestoreModal() {{
      document.getElementById("restore-modal").classList.remove("show");
    }}

    async function executeRestore(confirm) {{
      const ver = document.getElementById("modal-restore-version").value;
      const targetDir = document.getElementById("modal-restore-target").value;
      const box = document.getElementById("restore-feedback-box");

      try {{
        if (isOfflineMode) {{
          box.style.display = "block";
          if (!confirm) {{
            box.style.background = "rgba(245, 158, 11, 0.15)";
            box.style.border = "1px solid var(--warning-border)";
            box.style.color = "#FCD34D";
            box.innerHTML = `<strong>Dry-Run Passed (Offline Simulation)</strong><br>Target: ${{targetDir}}<br>Approval required. Click '2. Confirm &amp; Restore' to proceed.`;
          }} else {{
            box.style.background = "rgba(16, 185, 129, 0.15)";
            box.style.border = "1px solid var(--success-border)";
            box.style.color = "#6EE7B7";
            box.innerHTML = `<strong>&#10003; Simulated Restore of ${{ver}}!</strong><br>Target: ${{targetDir}}<br>Backup: .patchguard-backups/timestamped`;
            showToast(`Simulated restore of ${{ver}} succeeded.`, "success");
          }}
          return;
        }}

        const res = await fetch("/api/restore", {{
          method: "POST",
          headers: {{"Content-Type": "application/json"}},
          body: JSON.stringify({{
            patch_id: activePatchId,
            version_id: ver,
            target_dir: targetDir,
            confirm: confirm
          }})
        }});
        const result = await res.json();
        box.style.display = "block";

        if (!result.restored) {{
          box.style.background = "rgba(245, 158, 11, 0.15)";
          box.style.border = "1px solid var(--warning-border)";
          box.style.color = "#FCD34D";
          box.innerHTML = `
            <strong>Preview Verification Passed!</strong><br>
            Target directory: ${{targetDir}}<br>
            Approval required. Click '2. Confirm &amp; Restore' to write files.
          `;
          showToast("Dry-run preview passed. Approval required.", "warning");
        }} else {{
          box.style.background = "rgba(16, 185, 129, 0.15)";
          box.style.border = "1px solid var(--success-border)";
          box.style.color = "#6EE7B7";
          box.innerHTML = `
            <strong>&#10003; Successfully Restored ${{ver}}!</strong><br>
            Paths: ${{result.paths.join("<br>")}}<br>
            Backup: ${{result.backup_dir || 'None needed (no existing files overwritten)'}}
          `;
          showToast(`Successfully restored ${{ver}} to ${{targetDir}}!`, "success");
        }}
      }} catch (err) {{
        box.style.display = "block";
        box.style.background = "rgba(239, 68, 68, 0.15)";
        box.style.border = "1px solid var(--danger-border)";
        box.style.color = "#FCA5A5";
        box.textContent = `Restore error: ${{err}}`;
        showToast("Restore failed: " + err, "error");
      }}
    }}

    // Evaluation Tab
    async function loadEvaluationBenchmarks() {{
      const btn = document.getElementById("run-eval-btn");
      if (btn) {{
        btn.innerHTML = `<span class="spinner"></span> Running Evaluation...`;
        btn.disabled = true;
      }}

      try {{
        let data;
        if (isOfflineMode) {{
          await new Promise(r => setTimeout(r, 400));
          data = OFFLINE_DATA.evaluation;
        }} else {{
          const res = await fetch("/api/evaluation");
          data = await res.json();
        }}
        
        document.getElementById("eval-kpi-detect").textContent = `${{data.summary.detection_rate_pct.toFixed(1)}}%`;
        document.getElementById("eval-kpi-fpr").textContent = `${{data.summary.false_positive_rate_pct.toFixed(1)}}%`;

        const grid = document.getElementById("eval-cases-grid");
        grid.innerHTML = "";

        data.cases.forEach(c => {{
          const cardCls = c.passed ? 'success-border' : 'fail-border';

          grid.innerHTML += `
            <div class="eval-card ${{cardCls}}">
              <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div>
                  <span class="kpi-badge ${{c.passed ? 'badge-success' : 'badge-danger'}}" style="margin-bottom:0.4rem; display:inline-block;">
                    ${{c.passed ? 'BENCHMARK PASSED' : 'BENCHMARK FAILED'}}
                  </span>
                  <h4 style="font-size:1.05rem; font-weight:700; color:white;">${{escapeHtml(c.title)}}</h4>
                </div>
                <button class="btn btn-secondary btn-sm" onclick="inspectEvalPatch('${{c.patch_id}}')">
                  Inspect in Explorer &rarr;
                </button>
              </div>

              <p style="font-size:0.84rem; color:var(--text-muted);">${{escapeHtml(c.description)}}</p>

              <div style="background:rgba(11, 15, 23, 0.6); padding:0.85rem; border-radius:var(--radius-sm); font-size:0.8rem; display:flex; flex-direction:column; gap:0.4rem;">
                <div><strong>Target:</strong> <code>${{c.target_file}}::${{c.target_function}}</code></div>
                <div><strong>Comparison:</strong> <code>${{c.comparison}}</code></div>
                <div><strong>Expected Risks:</strong> <code>${{c.expected_risks.length ? c.expected_risks.join(', ') : 'None (0 false-positives)'}}</code></div>
                <div><strong>Detected Risks:</strong> <code>${{c.detected_risks.length ? c.detected_risks.join(', ') : 'None'}}</code></div>
                <div><strong>Behavioral Test Status:</strong> <code style="color:${{c.actual_test_status === 'passed' ? 'var(--success)' : 'var(--danger)'}}">${{c.actual_test_status.toUpperCase()}}</code></div>
                <div><strong>Candidate Accepted:</strong> <code>${{c.actual_accepted}}</code></div>
              </div>
            </div>
          `;
        }});
        showToast("Evaluation benchmarks loaded.", "success");
      }} catch (err) {{
        console.error("Failed to load evaluation benchmarks:", err);
        showToast("Failed to load benchmarks: " + err, "error");
      }} finally {{
        if (btn) {{
          btn.innerHTML = `<span>&#8635; Re-run Evaluation Suite</span>`;
          btn.disabled = false;
        }}
      }}
    }}

    function inspectEvalPatch(patchId) {{
      switchTab('explorer');
      const sel = document.getElementById("patch-select");
      sel.value = patchId;
      onPatchChange();
    }}
  </script>
</body>
</html>
"""


def export_static_dashboard(output_path: Path):
    """Generate and write a standalone static HTML dashboard."""
    seed_evaluation_fixtures()
    html = get_dashboard_html()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def run_dashboard(
    port: int = 8000,
    host: str = "127.0.0.1",
    open_browser: bool = True,
    export_path: str = None,
):
    """Run the local PatchGuard HTTP dashboard server or export static file."""
    if export_path:
        out = export_static_dashboard(Path(export_path))
        print(f"Standalone PatchGuard dashboard exported to: {out}")
        return 0

    seed_evaluation_fixtures()
    server_address = (host, port)
    httpd = ThreadedHTTPServer(server_address, PatchGuardRequestHandler)
    url = f"http://{host}:{port}"

    print("=" * 65)
    print("           PATCHGUARD INTERACTIVE DASHBOARD & API           ")
    print("=" * 65)
    print(f"Server running at: {url}")
    print("Press Ctrl+C to stop the dashboard.")
    print("=" * 65)

    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping PatchGuard dashboard server...")
        httpd.server_close()
        return 0
    return 0
