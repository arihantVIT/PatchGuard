import pytest
import patchguard.storage as storage
from patchguard.storage import (
    load_checkpoint_metadata,
    record_test_result,
)

from patchguard.storage import (
    save_version,
    load_version,
    list_versions,
)


def test_record_results_and_keep_newest_passing_checkpoint(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(storage, "STORAGE_DIR", tmp_path)

    for version_id in ("v1", "v2", "v3"):
        storage.save_version("calculator", version_id, {"calc.py": "x = 1"})

    record_test_result(
        "calculator", "v1", "calculator_behavior_v1",
        {"status": "passed", "passed": 2, "failed": 0, "stdout": "2 passed"},
    )
    record_test_result(
        "calculator", "v2", "calculator_behavior_v1",
        {"status": "passed", "passed": 3, "failed": 0, "stdout": "3 passed"},
    )
    record_test_result(
        "calculator", "v3", "calculator_behavior_v1",
        {"status": "failed", "passed": 2, "failed": 1, "stderr": "1 failed"},
    )
    record_test_result(
        "calculator", "v3", "calculator_behavior_v1",
        {"status": "error", "passed": 0, "failed": 0, "stderr": "collection error"},
    )
    record_test_result(
        "calculator", "v1", "alternate_suite_v1",
        {"status": "passed", "passed": 1, "failed": 0, "stdout": "1 passed"},
    )

    metadata = load_checkpoint_metadata("calculator")

    assert metadata["best_checkpoints"]["calculator_behavior_v1"]["version_id"] == "v2"
    assert metadata["best_checkpoints"]["alternate_suite_v1"]["version_id"] == "v1"
    runs = metadata["versions"]["v3"]["test_runs"]["calculator_behavior_v1"]
    assert [run["status"] for run in runs] == ["failed", "error"]
    assert runs[0]["passed"] == 2
    assert runs[0]["failed"] == 1
    assert runs[0]["stderr"] == "1 failed"


def test_record_test_result_requires_saved_version_and_valid_status(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(storage, "STORAGE_DIR", tmp_path)

    with pytest.raises(FileNotFoundError):
        record_test_result(
            "calculator",
            "missing",
            "calculator_behavior_v1",
            {"status": "passed"},
        )

    storage.save_version("calculator", "v1", {"calc.py": "x = 1"})

    for invalid_status in ("skipped", [], None):
        with pytest.raises(ValueError, match="result status"):
            record_test_result(
                "calculator",
                "v1",
                "calculator_behavior_v1",
                {"status": invalid_status},
            )


def test_checkpoint_manifest_write_is_atomic_on_replace_failure(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(storage, "STORAGE_DIR", tmp_path)
    storage.save_version("calculator", "v1", {"calc.py": "x = 1"})
    path = storage.checkpoint_path("calculator")
    original_content = path.read_text(encoding="utf-8")

    def fail_replace(_source, _destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(storage.os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        storage._write_checkpoint_metadata(
            "calculator",
            {"schema_version": 1, "patch_id": "calculator", "versions": {}, "best_checkpoints": {}},
        )

    assert path.read_text(encoding="utf-8") == original_content
    assert list(path.parent.glob(".checkpoint-*.tmp")) == []

def test_save_and_load_version(tmp_path, monkeypatch):
    import patchguard.storage as storage

    monkeypatch.setattr(storage, "STORAGE_DIR", tmp_path)

    files = {
        "calculator.py": (
            "def add(a, b):\n"
            "    return a + b\n"
        )
    }

    save_version("calculator", "v0", files)

    loaded = load_version("calculator", "v0")

    assert loaded["patch_id"] == "calculator"
    assert loaded["version_id"] == "v0"
    assert loaded["files"]["calculator.py"] == files["calculator.py"]


def test_versions_remain_independent(tmp_path, monkeypatch):
    import patchguard.storage as storage

    monkeypatch.setattr(storage, "STORAGE_DIR", tmp_path)

    files_v0 = {"calculator.py": "return a + b"}
    files_v1 = {"calculator.py": "return a * b"}

    save_version("calculator", "v0", files_v0)
    save_version("calculator", "v1", files_v1)

    assert load_version("calculator", "v0")["files"] == files_v0
    assert load_version("calculator", "v1")["files"] == files_v1


def test_list_versions(tmp_path, monkeypatch):
    import patchguard.storage as storage

    monkeypatch.setattr(storage, "STORAGE_DIR", tmp_path)

    save_version("calculator", "v0", {"a.py": "x = 1"})
    save_version("calculator", "v1", {"a.py": "x = 2"})

    assert list_versions("calculator") == ["v0", "v1"]


def test_missing_version_raises_error(tmp_path, monkeypatch):
    import patchguard.storage as storage

    monkeypatch.setattr(storage, "STORAGE_DIR", tmp_path)

    with pytest.raises(FileNotFoundError):
        load_version("calculator", "v99")