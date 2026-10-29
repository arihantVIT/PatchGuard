import json
from types import SimpleNamespace

import pytest

import patchguard.diagnosis as diagnosis
import patchguard.storage as storage
from patchguard.pipeline import compare_versions


def _response(content):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
            )
        ]
    )


class _FakeCompletions:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


class _FakeOpenAI:
    def __init__(self, response, **init_kwargs):
        self.init_kwargs = init_kwargs
        self.completions = _FakeCompletions(response)
        self.chat = SimpleNamespace(completions=self.completions)


def test_explain_risks_sends_minimal_evidence_and_validates_response(
    monkeypatch,
):
    warning = {
        "warning_id": "W001",
        "category": "validation_removed",
        "severity": "high",
        "file": "calculator.py",
        "function": "calculate_total",
        "version_ids": ["v1", "v2"],
        "evidence": "raise ValueError('negative')",
        "diff_evidence": "- raise ValueError('negative')",
        "pytest_output": "must not be sent",
    }
    response_content = json.dumps({
        "explanations": [{
            "warning_id": "W001",
            "explanation": "The input guard was removed, so negative discounts may be accepted.",
            "evidence_refs": ["W001-E1", "W001-D1"],
            "quotes": [{
                "evidence_id": "W001-E1",
                "text": "raise ValueError('negative')",
            }],
        }]
    })
    fake_client = _FakeOpenAI(_response(response_content))
    client_kwargs = {}

    def create_client(**kwargs):
        client_kwargs.update(kwargs)
        return fake_client

    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    monkeypatch.setattr(diagnosis, "OpenAI", create_client)

    explanations = diagnosis.explain_risks([warning])

    assert client_kwargs["api_key"] == "test-key"
    assert client_kwargs["base_url"] == diagnosis.NVIDIA_BASE_URL
    request = fake_client.completions.kwargs
    assert request["model"] == diagnosis.NVIDIA_MODEL
    assert request["response_format"] == {"type": "json_object"}
    request_text = request["messages"][1]["content"]
    assert "must not be sent" not in request_text
    assert "original_code" not in request_text
    assert explanations["W001"]["validation_status"] == "validated"
    assert explanations["W001"]["evidence_refs"] == ["W001-E1", "W001-D1"]


def test_explanation_rejects_unknown_evidence_reference():
    content = json.dumps({
        "explanations": [{
            "warning_id": "W001",
            "explanation": "Unsupported explanation.",
            "evidence_refs": ["invented"],
            "quotes": [],
        }]
    })

    with pytest.raises(ValueError, match="unknown or missing evidence"):
        diagnosis.validate_explanations(
            content,
            {"W001-E1": "real evidence"},
            ["W001"],
        )


def test_explanation_rejects_quote_not_in_evidence():
    content = json.dumps({
        "explanations": [{
            "warning_id": "W001",
            "explanation": "Unsupported quote.",
            "evidence_refs": ["W001-E1"],
            "quotes": [{
                "evidence_id": "W001-E1",
                "text": "not present",
            }],
        }]
    })

    with pytest.raises(ValueError, match="not present in submitted evidence"):
        diagnosis.validate_explanations(
            content,
            {"W001-E1": "actual evidence"},
            ["W001"],
        )


@pytest.mark.parametrize("content", ["not-json", "{\"explanations\": []}"])
def test_explanation_rejects_malformed_or_incomplete_response(content):
    with pytest.raises(ValueError):
        diagnosis.validate_explanations(
            content,
            {"W001-E1": "evidence"},
            ["W001"],
        )


def test_missing_nemotron_key_fails_before_client_creation(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setattr(
        diagnosis,
        "OpenAI",
        lambda **kwargs: pytest.fail("client must not be created"),
    )

    with pytest.raises(RuntimeError, match="NVIDIA_API_KEY"):
        diagnosis.explain_risks([{"evidence": "source evidence"}])


def test_pipeline_preserves_risks_when_nemotron_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "STORAGE_DIR", tmp_path)
    storage.save_version(
        "fallback",
        "v1",
        {
            "module.py": (
                "def process(value):\n"
                "    if value < 0:\n"
                "        raise ValueError('negative')\n"
                "    return value\n"
            )
        },
    )
    storage.save_version(
        "fallback",
        "v2",
        {"module.py": "def process(value):\n    return value\n"},
    )
    monkeypatch.setattr(
        diagnosis,
        "explain_risks",
        lambda risks: (_ for _ in ()).throw(ValueError("bad response")),
    )

    report = compare_versions("fallback", "v1", "v2", run_llm=True)

    assert any(risk["category"] == "validation_removed" for risk in report["risks"])
    assert report["risks"][0]["explanation"] is None
    assert report["risks"][0]["explanation_status"] == "unavailable_or_rejected"
    assert report["diagnosis_error"] == "ValueError"


def test_pipeline_no_llm_does_not_request_explanations(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "STORAGE_DIR", tmp_path)
    storage.save_version(
        "offline",
        "v1",
        {"module.py": "def f(x):\n    if x < 0:\n        raise ValueError()\n    return x\n"},
    )
    storage.save_version("offline", "v2", {"module.py": "def f(x):\n    return x\n"})
    monkeypatch.setattr(
        diagnosis,
        "explain_risks",
        lambda risks: pytest.fail("Nemotron must not be called"),
    )

    report = compare_versions("offline", "v1", "v2", run_llm=False)

    assert report["risks"][0]["explanation_status"] == "not_requested"
    assert report["risks"][0]["explanation"] is None


def test_pipeline_attaches_validated_explanations(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "STORAGE_DIR", tmp_path)
    storage.save_version(
        "explained",
        "v1",
        {"module.py": "def f(x):\n    if x < 0:\n        raise ValueError()\n    return x\n"},
    )
    storage.save_version("explained", "v2", {"module.py": "def f(x):\n    return x\n"})
    monkeypatch.setattr(
        diagnosis,
        "explain_risks",
        lambda risks: {
            "W001": {
                "explanation": "The negative-value guard was removed.",
                "evidence_refs": ["W001-E1", "W001-D1"],
                "validation_status": "validated",
            }
        },
    )

    report = compare_versions("explained", "v1", "v2", run_llm=True)
    warning = next(
        risk for risk in report["risks"]
        if risk["category"] == "validation_removed"
    )

    assert warning["warning_id"] == "W001"
    assert warning["explanation"] == "The negative-value guard was removed."
    assert warning["explanation_refs"] == ["W001-E1", "W001-D1"]
    assert warning["explanation_status"] == "validated"


def test_cli_prints_validated_explanation(monkeypatch, capsys):
    from patchguard.__main__ import main

    monkeypatch.setattr(
        "sys.argv",
        ["patchguard", "compare", "demo", "v1", "v2", "--no-llm"],
    )
    monkeypatch.setattr(
        "patchguard.__main__.compare_versions",
        lambda **kwargs: {
            "patch_id": "demo",
            "old_version": "v1",
            "new_version": "v2",
            "added_files": [],
            "removed_files": [],
            "modified_files": [],
            "diffs": {},
            "function_changes": {},
            "tests": {},
            "risks": [{
                "warning_id": "W001",
                "category": "validation_removed",
                "severity": "high",
                "file": "module.py",
                "function": "f",
                "message": "Guard removed",
                "evidence": "raise ValueError()",
                "diff_evidence": "- raise ValueError()",
                "version_ids": ["v1", "v2"],
                "explanation": "The guard is absent in v2.",
                "explanation_refs": ["W001-E1"],
                "explanation_status": "validated",
            }],
            "diagnosis": None,
            "diagnosis_error": None,
        },
    )

    assert main() == 0
    output = capsys.readouterr().out
    assert "Explanation: The guard is absent in v2." in output
    assert "Explanation evidence: W001-E1" in output
