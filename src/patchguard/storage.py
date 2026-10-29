import json
from pathlib import Path
import os
import tempfile
from datetime import datetime, timezone
from uuid import uuid4



# Directory where all patch versions will be stored
STORAGE_DIR = Path(__file__).resolve().parents[2] / "data" / "patches"

#Construct the path where a patch's checkpoint metadata will be stored.
def checkpoint_path(patch_id):
    return STORAGE_DIR / patch_id / "checkpoint.json"



#Create a default Python dictionary representing a patch that has no recorded checkpoint metadata yet.
def _empty_checkpoint_metadata(patch_id):  
    return {
        "schema_version" : 1,
        "patch_id" : patch_id,
        "versions" : {},
        "best_checkpoints" : {},
    }

#this function loads the checkpoint metadata for a particular patch if it exists. If it doesn't exist, it returns an empty metadata structure.
def load_checkpoint_metadata(patch_id):
    path = checkpoint_path(patch_id)
    if not path.exists():
        return _empty_checkpoint_metadata(patch_id)

    with path.open(encoding = "utf-8") as file:
        metadata = json.load(file)

    if metadata.get("patch_id") != patch_id:
        raise ValueError(f"Checkpoint metadata belongs to another patch: {path}")

    return metadata

#saves checkpoint metadata safely without risking corruption if the program crashes while writing the file.
def _write_checkpoint_metadata(patch_id, metadata):
    path = checkpoint_path(patch_id)
    path.parent.mkdir(parents = True, exist_ok = True)

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir = path.parent,
            prefix=".checkpoint-",
            suffix=".tmp",
            delete=False
        ) as file:
            temp_path = Path(file.name)
            json.dump(metadata, file, indent=2)
            file.write("\n")

        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

#recording api
def record_test_result(patch_id, version_id, suite_id, result):
    if not suite_id or not isinstance(suite_id, str):
        raise ValueError("suite_id must be a non-empty string")
    if not isinstance(result, dict):
        raise ValueError("result must be a dictionary")

    status = result.get("status")
    if not isinstance(status, str) or status not in {
        "passed",
        "failed",
        "error",
    }:
        raise ValueError("result status must be passed, failed, or error")

    # Verify that results are only recorded for saved snapshots.
    load_version(patch_id, version_id)

    metadata = load_checkpoint_metadata(patch_id)
    version_metadata = metadata["versions"].setdefault(
        version_id,
        {"test_runs": {}},
    )

    run = {
        **result,
        "run_id": str(uuid4()),
        "suite_id": suite_id,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }

    suite_runs = version_metadata["test_runs"].setdefault(suite_id, [])
    suite_runs.append(run)

    sequence = version_metadata.get("sequence")
    if status == "passed" and sequence is not None:
        best = metadata["best_checkpoints"].get(suite_id)
        if best is None or sequence > best["sequence"]:
            metadata["best_checkpoints"][suite_id] = {
                "version_id": version_id,
                "sequence": sequence,
                "run_id": run["run_id"],
            }
    elif status == "passed":
        run["checkpoint_eligible"] = False

    _write_checkpoint_metadata(patch_id, metadata)
    return run

def save_version(patch_id, version_id, files):
    """Save a snapshot and record its save-order sequence."""
    patch_dir = STORAGE_DIR / patch_id
    patch_dir.mkdir(parents=True, exist_ok=True)

    version_file = patch_dir / f"{version_id}.json"
    if version_file.exists():
        raise FileExistsError(
            f"Version '{version_id}' already exists for patch '{patch_id}'"
        )

    metadata = load_checkpoint_metadata(patch_id)
    version_metadata = metadata["versions"]

    # Do not guess the age of snapshots saved before sequence tracking existed.
    existing_snapshots = {
        path.stem
        for path in patch_dir.glob("*.json")
        if path.name != "checkpoint.json"
    }
    untracked_snapshots = existing_snapshots - version_metadata.keys()
    if untracked_snapshots:
        raise ValueError(
            "Existing snapshots need explicit sequence backfill first: "
            + ", ".join(sorted(untracked_snapshots))
        )

    sequences = [
        entry["sequence"]
        for entry in version_metadata.values()
        if "sequence" in entry
    ]
    sequence = max(sequences, default=0) + 1

    data = {
        "patch_id": patch_id,
        "version_id": version_id,
        "files": files,
    }

    with version_file.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

    version_metadata[version_id] = {
        "sequence": sequence,
        "test_runs": {},
    }
    _write_checkpoint_metadata(patch_id, metadata)

    print(f"Saved {patch_id} - {version_id} (sequence {sequence})")


def load_version(patch_id, version_id):
    """Load a previously saved version of a patch."""

    version_file = STORAGE_DIR / patch_id / f"{version_id}.json"

    # Check whether the version exists
    if not version_file.exists():
        raise FileNotFoundError(
            f"Version '{version_id}' not found for patch '{patch_id}'"
        )

    # Read the saved version
    with open(version_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def list_versions(patch_id):
    """List all versions of a patch."""

    patch_dir = STORAGE_DIR / patch_id

    # Check whether the patch exists
    if not patch_dir.exists():
        return []

    # Find all saved versions
    versions = [
        file.stem
        for file in patch_dir.glob("*.json")
        if file.is_file() and file.name != "checkpoint.json"
    ]

    return sorted(versions)
