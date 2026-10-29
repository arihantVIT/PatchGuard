import pytest

from patchguard import agent


REGRESSION_TESTS = """
import module


def test_existing_double_behavior():
    assert module.double(4) == 8
"""

ACCEPTANCE_TESTS = """
import module


def test_duration_parser():
    assert module.parse_duration("2m") == 120
"""


def _source_file(tmp_path):
    path = tmp_path / "module.py"
    path.write_text("def double(value):\n    return value * 2\n", encoding="utf-8")
    return path


def test_generate_workflow_tests_requests_specific_behavior(monkeypatch):
    requests = []

    def request(system_prompt, user_prompt):
        requests.append((system_prompt, user_prompt))
        return REGRESSION_TESTS

    monkeypatch.setattr(agent, "_request_workflow_code", request)
    result = agent.generate_workflow_tests_with_nemotron(
        "def double(value): return value * 2", "module", "module.py"
    )

    assert "test_existing_double_behavior" in result
    assert "behavior already implemented" in requests[0][1]


def test_nemotron_request_uses_nvidia_endpoint_model_and_api_key(monkeypatch):
    from types import SimpleNamespace

    from patchguard import agent

    monkeypatch.setenv("NVIDIA_API_KEY", "test-nvidia-key")
    captured = {}

    class Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content="```python\nprint('ok')\n```")
                )]
            )

    class Client:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.chat = SimpleNamespace(completions=Completions())

    monkeypatch.setattr(agent, "OpenAI", Client)

    assert agent._request_workflow_code("system", "user") == "print('ok')"
    assert captured["client"] == {
        "api_key": "test-nvidia-key",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "timeout": 60.0,
        "max_retries": 1,
    }
    assert captured["model"] == "nvidia/nemotron-3.5-super-vl-120b-a12b"


def test_generated_suite_must_test_behavior():
    with pytest.raises(ValueError, match="assertions or exception checks"):
        agent._validate_generated_test_suite(
            "def test_only_checks_name():\n    assert True\n"
        )


def test_workflow_retries_candidate_and_writes_only_passing_result(
    tmp_path, monkeypatch
):
    source = _source_file(tmp_path)
    output = tmp_path / "accepted.py"
    generated_suites = []
    candidates = iter([
        "def double(value):\n    return value + 2\n\n"
        "def parse_duration(text):\n    return int(text[:-1]) * 60\n",
        "def double(value):\n    return value * 2\n\n"
        "def parse_duration(text):\n    return int(text[:-1]) * 60\n",
    ])

    def generate(code, module_name, target_filename, task=None):
        generated_suites.append(task)
        return ACCEPTANCE_TESTS if task else REGRESSION_TESTS

    monkeypatch.setattr(agent, "generate_workflow_tests_with_nemotron", generate)
    monkeypatch.setattr(agent, "_request_workflow_code", lambda *_: next(candidates))

    result = agent.run_patch_test_revise(
        source, "Add parse_duration for minutes", output, max_attempts=2
    )

    assert generated_suites == [None, "Add parse_duration for minutes"]
    assert result["accepted"] is True
    assert len(result["attempts"]) == 2
    assert result["attempts"][0]["regression"]["status"] == "failed"
    assert result["attempts"][1]["regression"]["status"] == "passed"
    assert result["attempts"][1]["acceptance"]["status"] == "passed"
    assert source.read_text(encoding="utf-8") == (
        "def double(value):\n    return value * 2\n"
    )
    assert "parse_duration" in output.read_text(encoding="utf-8")


def test_failed_attempts_never_write_candidate(tmp_path, monkeypatch):
    source = _source_file(tmp_path)
    output = tmp_path / "rejected.py"
    monkeypatch.setattr(agent, "generate_workflow_tests_with_nemotron", lambda *a, **k: (
        ACCEPTANCE_TESTS if k.get("task") else REGRESSION_TESTS
    ))
    monkeypatch.setattr(
        agent,
        "_request_workflow_code",
        lambda *_: "def double(value):\n    return value + 1\n\n"
        "def parse_duration(text):\n    return int(text[:-1]) * 60\n",
    )

    result = agent.run_patch_test_revise(
        source, "Add parse_duration for minutes", output, max_attempts=2
    )

    assert result["accepted"] is False
    assert len(result["attempts"]) == 2
    assert not output.exists()
    assert source.read_text(encoding="utf-8") == (
        "def double(value):\n    return value * 2\n"
    )


def test_workflow_stops_if_generated_regression_tests_fail_baseline(
    tmp_path, monkeypatch
):
    source = _source_file(tmp_path)
    output = tmp_path / "never-written.py"
    monkeypatch.setattr(agent, "generate_workflow_tests_with_nemotron", lambda *a, **k: (
        "def test_baseline_is_wrong():\n    assert False\n"
    ))

    with pytest.raises(RuntimeError, match="no patch was attempted"):
        agent.run_patch_test_revise(source, "Add a feature", output)

    assert not output.exists()
