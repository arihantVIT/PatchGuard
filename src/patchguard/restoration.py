"""Restore stored snapshots into an explicitly selected directory."""

from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from shutil import copy2
from uuid import uuid4

from .storage import load_version


def _validated_destination(root, relative_name):
    if not isinstance(relative_name, str) or not relative_name:
        raise ValueError("Snapshot file paths must be non-empty strings.")
    if "\\" in relative_name or "\x00" in relative_name:
        raise ValueError(f"Unsafe snapshot file path: {relative_name!r}")

    relative_path = PurePosixPath(relative_name)
    windows_path = PureWindowsPath(relative_name)
    if (
        relative_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or ".." in relative_path.parts
    ):
        raise ValueError(f"Unsafe snapshot file path: {relative_name!r}")

    destination = root.joinpath(*relative_path.parts)
    resolved_root = root.resolve()
    resolved_destination = destination.resolve(strict=False)
    try:
        resolved_destination.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(
            f"Snapshot file escapes the restore target: {relative_name!r}"
        ) from exc

    current = root
    for part in relative_path.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(
                f"Restore destination contains a symbolic link: {current}"
            )
    return destination


def restore_version(patch_id, version_id, target_dir, *, confirmed=False):
    """Restore snapshot files, requiring explicit approval before any writes."""
    version = load_version(patch_id, version_id)
    files = version.get("files")
    if not isinstance(files, dict):
        raise ValueError("Stored version must contain a files dictionary.")

    if not confirmed:
        return {
            "restored": False,
            "reason": "confirmation_required",
            "patch_id": patch_id,
            "version_id": version_id,
            "paths": [],
            "backup_dir": None,
        }

    root = Path(target_dir).expanduser().resolve(strict=False)
    destinations = {}
    for relative_name, content in files.items():
        if not isinstance(content, str):
            raise ValueError(
                f"Source content for {relative_name!r} must be a string."
            )
        destination = _validated_destination(root, relative_name)
        if destination.exists() and not destination.is_file():
            raise ValueError(
                f"Restore destination is not a regular file: {destination}"
            )
        destinations[relative_name] = destination

    existing_files = {
        relative_name: destination
        for relative_name, destination in destinations.items()
        if destination.is_file()
    }

    backup_dir = None
    if existing_files:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup_dir = (
            root.parent
            / f".{root.name}.patchguard-backups"
            / f"{timestamp}-{uuid4().hex[:8]}"
        )
        for relative_name, source in existing_files.items():
            backup_path = backup_dir / PurePosixPath(relative_name)
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            copy2(source, backup_path)

    for relative_name, destination in destinations.items():
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(files[relative_name], encoding="utf-8")

    return {
        "restored": True,
        "reason": None,
        "patch_id": patch_id,
        "version_id": version_id,
        "paths": [str(path) for path in destinations.values()],
        "backup_dir": str(backup_dir) if backup_dir else None,
    }