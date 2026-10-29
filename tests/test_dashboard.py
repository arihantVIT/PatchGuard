"""Tests for PatchGuard Dashboard API server and static export."""

import json
import threading
import time
from urllib.request import urlopen, Request

import pytest
import patchguard.storage as storage
from patchguard.dashboard import (
    ThreadedHTTPServer,
    PatchGuardRequestHandler,
    export_static_dashboard,
)


@pytest.fixture(scope="module")
def dashboard_server(tmp_path_factory):
    storage_dir = tmp_path_factory.mktemp("dashboard_patches")
    old_storage = storage.STORAGE_DIR
    storage.STORAGE_DIR = storage_dir

    # Seed demo patch
    storage.save_version(
        "demo_patch",
        "v1",
        {
            "calculator.py": (
                "def calculate_total(price, discount):\n"
                "    if discount < 0 or discount > price:\n"
                "        raise ValueError()\n"
                "    return price - discount\n"
            )
        },
    )
    storage.save_version(
        "demo_patch",
        "v2",
        {
            "calculator.py": (
                "def calculate_total(price, discount):\n"
                "    return price - discount\n"
            )
        },
    )

    server = ThreadedHTTPServer(("127.0.0.1", 0), PatchGuardRequestHandler)
    host, port = server.server_address
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    yield f"http://{host}:{port}"

    server.shutdown()
    server.server_close()
    storage.STORAGE_DIR = old_storage


def test_dashboard_index_html(dashboard_server):
    with urlopen(dashboard_server + "/") as resp:
        assert resp.status == 200
        html = resp.read().decode("utf-8")
        assert "PatchGuard" in html
        assert "Checkpoint &amp; Version Timeline" in html
        assert "Deterministic Risk Signals" in html


def test_api_patches_endpoint(dashboard_server):
    with urlopen(dashboard_server + "/api/patches") as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "patches" in data
        patch_ids = [p["patch_id"] for p in data["patches"]]
        assert "demo_patch" in patch_ids


def test_api_patch_details(dashboard_server):
    with urlopen(dashboard_server + "/api/patch/demo_patch") as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["patch_id"] == "demo_patch"
        assert "v1" in data["versions"]
        assert "v2" in data["versions"]


def test_api_compare_endpoint(dashboard_server):
    url = f"{dashboard_server}/api/compare?patch_id=demo_patch&old=v1&new=v2&llm=false"
    with urlopen(url) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["patch_id"] == "demo_patch"
        assert "calculator.py" in data["diffs"]
        assert "risks" in data


def test_api_evaluation_endpoint(dashboard_server):
    with urlopen(dashboard_server + "/api/evaluation") as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["summary"]["all_passed"] is True
        assert data["summary"]["total_cases"] == 4


def test_api_restore_endpoint_dry_run_and_confirm(dashboard_server, tmp_path):
    target_dir = tmp_path / "restored"

    # 1. Dry run
    req_data = json.dumps({
        "patch_id": "demo_patch",
        "version_id": "v1",
        "target_dir": str(target_dir),
        "confirm": False,
    }).encode("utf-8")
    req = Request(
        dashboard_server + "/api/restore",
        data=req_data,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req) as resp:
        assert resp.status == 200
        res = json.loads(resp.read().decode("utf-8"))
        assert res["restored"] is False
        assert not target_dir.exists()

    # 2. Confirm
    req_data_confirmed = json.dumps({
        "patch_id": "demo_patch",
        "version_id": "v1",
        "target_dir": str(target_dir),
        "confirm": True,
    }).encode("utf-8")
    req_confirmed = Request(
        dashboard_server + "/api/restore",
        data=req_data_confirmed,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req_confirmed) as resp:
        assert resp.status == 200
        res = json.loads(resp.read().decode("utf-8"))
        assert res["restored"] is True
        assert (target_dir / "calculator.py").exists()


def _post_json(url, payload):
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    return urlopen(request)


def test_agent_run_requires_task_and_valid_source(dashboard_server, monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")

    with pytest.raises(Exception) as exc:
        _post_json(
            dashboard_server + "/api/agent/run",
            {"source_code": "def f():\n    return 1\n", "task": "  "},
        )
    assert exc.value.code == 400

    with pytest.raises(Exception) as exc:
        _post_json(
            dashboard_server + "/api/agent/run",
            {"source_code": "def f(:\n", "task": "Add g"},
        )
    assert exc.value.code == 400


def test_agent_run_reports_missing_api_key(dashboard_server, monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

    with pytest.raises(Exception) as exc:
        _post_json(
            dashboard_server + "/api/agent/run",
            {"source_code": "def f():\n    return 1\n", "task": "Add g"},
        )
    assert exc.value.code == 400


def test_agent_job_streams_progress_and_saves_accepted_candidate(
    dashboard_server, monkeypatch
):
    import patchguard.dashboard as dashboard

    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    source = "def normalize_spaces(text):\n    return \" \".join(text.split())\n"
    regression = (
        "import module\n\n\ndef test_spaces():\n"
        "    assert module.normalize_spaces(' a  b ') == 'a b'\n"
    )

    def fake_workflow(source_path, task, output_path, max_attempts=3, progress=None):
        progress("regression_tests_ready", {"passed_count": 1, "status": "passed"})
        progress("attempt_started", {"attempt": 1})
        progress(
            "attempt_finished",
            {
                "attempt": 1,
                "accepted": True,
                "regression": {"status": "passed", "passed_count": 1, "failed_count": 0},
                "acceptance": {"status": "passed", "passed_count": 2, "failed_count": 0},
            },
        )
        progress("accepted", {"attempt": 1})
        output_path.write_text(
            source_path.read_text(encoding="utf-8")
            + "\n\ndef slugify(text):\n    return text.lower()\n",
            encoding="utf-8",
        )
        return {
            "accepted": True,
            "output_path": str(output_path),
            "attempts": [],
            "regression_test_count": 1,
            "acceptance_test_count": 2,
            "initial_acceptance_status": "failed",
            "regression_tests": regression,
            "acceptance_tests": "",
        }

    monkeypatch.setattr(dashboard, "run_patch_test_revise", fake_workflow)

    with _post_json(
        dashboard_server + "/api/agent/run",
        {
            "source_code": source,
            "filename": "module.py",
            "task": "Add slugify",
            "max_attempts": 1,
        },
    ) as resp:
        assert resp.status == 202
        job_id = json.loads(resp.read().decode("utf-8"))["job_id"]

    job = None
    deadline = time.time() + 20
    while time.time() < deadline:
        with urlopen(f"{dashboard_server}/api/agent/status?job={job_id}") as resp:
            job = json.loads(resp.read().decode("utf-8"))["job"]
        if job["status"] in ("accepted", "rejected", "error"):
            break
        time.sleep(0.2)

    assert job["status"] == "accepted", job
    assert any(stage["event"] == "attempt_finished" for stage in job["stages"])
    assert job["attempts"][0]["attempt"] == 1
    assert "slugify" in job["candidate_code"]
    assert "+def slugify" in job["diff"]
    assert job["patch_id"]

    with urlopen(dashboard_server + "/api/patches") as resp:
        patch_ids = [p["patch_id"] for p in json.loads(resp.read().decode("utf-8"))["patches"]]
    assert job["patch_id"] in patch_ids


def test_agent_status_rejects_unknown_job(dashboard_server):
    with pytest.raises(Exception) as exc:
        urlopen(dashboard_server + "/api/agent/status?job=missing")
    assert exc.value.code == 404


def test_export_static_dashboard(tmp_path):
    out_file = tmp_path / "exported_dashboard.html"
    res = export_static_dashboard(out_file)
    assert res.exists()
    content = res.read_text(encoding="utf-8")
    assert "PatchGuard" in content
    assert "<!DOCTYPE html>" in content
