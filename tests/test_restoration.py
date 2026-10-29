import pytest

import patchguard.storage as storage
from patchguard.restoration import restore_version


def test_restore_requires_confirmation_without_writing(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(storage, "STORAGE_DIR", tmp_path / "snapshots")
    storage.save_version("demo", "v1", {"src/app.py": "value = 1\n"})
    target = tmp_path / "working-copy"

    result = restore_version("demo", "v1", target)

    assert result["restored"] is False
    assert result["reason"] == "confirmation_required"
    assert not target.exists()


@pytest.mark.parametrize(
    "unsafe_path",
    ["../escape.py", "/tmp/escape.py", "C:\\escape.py", "src\\..\\escape.py"],
)
def test_restore_rejects_unsafe_paths(tmp_path, monkeypatch, unsafe_path):
    monkeypatch.setattr(storage, "STORAGE_DIR", tmp_path / "snapshots")
    storage.save_version("demo", "v1", {unsafe_path: "value = 1\n"})
    target = tmp_path / "working-copy"

    with pytest.raises(ValueError, match="Unsafe snapshot file path"):
        restore_version("demo", "v1", target, confirmed=True)

    assert not target.exists()


def test_restore_backs_up_overwritten_file_and_keeps_unrelated_files(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(storage, "STORAGE_DIR", tmp_path / "snapshots")
    storage.save_version(
        "demo", "v1", {"src/app.py": "restored = True\n"}
    )
    storage.record_test_result(
        "demo",
        "v1",
        "demo_suite_v1",
        {"status": "passed", "passed_count": 1, "failed_count": 0},
    )
    stored_version_before = storage.load_version("demo", "v1")
    metadata_before = storage.load_checkpoint_metadata("demo")
    target = tmp_path / "working-copy"
    existing = target / "src/app.py"
    existing.parent.mkdir(parents=True)
    existing.write_text("old = True\n", encoding="utf-8")
    unrelated = target / "notes.txt"
    unrelated.write_text("leave this alone\n", encoding="utf-8")

    result = restore_version("demo", "v1", target, confirmed=True)

    assert result["restored"] is True
    assert existing.read_text(encoding="utf-8") == "restored = True\n"
    assert unrelated.read_text(encoding="utf-8") == "leave this alone\n"
    backup = tmp_path / ".working-copy.patchguard-backups"
    backup_file = backup / existing.relative_to(target)
    backups = list(backup.rglob("app.py"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == "old = True\n"
    assert result["backup_dir"]
    assert storage.load_version("demo", "v1") == stored_version_before
    assert storage.load_checkpoint_metadata("demo") == metadata_before


def test_restore_rejects_non_string_contents_before_writing(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(storage, "STORAGE_DIR", tmp_path / "snapshots")
    storage.save_version("demo", "v1", {"src/app.py": 10})
    target = tmp_path / "working-copy"

    with pytest.raises(ValueError, match="must be a string"):
        restore_version("demo", "v1", target, confirmed=True)

    assert not target.exists()


def test_restore_rejects_symlink_escape(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "STORAGE_DIR", tmp_path / "snapshots")
    storage.save_version("demo", "v1", {"linked/app.py": "value = 1"})
    target = tmp_path / "working-copy"
    target.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (target / "linked").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="escapes|symbolic link"):
        restore_version("demo", "v1", target, confirmed=True)

    assert list(outside.iterdir()) == []