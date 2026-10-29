from patchguard.diff_engine import generate_diff


def test_changed_line_appears_in_diff():
    old_code = "def add(a, b):\n    return a + b\n"
    new_code = "def add(a, b):\n    return a - b\n"

    diff = generate_diff(old_code, new_code, "v0.py", "v1.py")

    assert "-    return a + b" in diff
    assert "+    return a - b" in diff


def test_identical_versions_produce_empty_diff():
    code = "def add(a, b):\n    return a + b\n"

    diff = generate_diff(code, code, "v0.py", "v1.py")

    assert diff == ""


def test_added_function_appears_in_diff():
    old_code = "def add(a, b):\n    return a + b\n"

    new_code = (
        "def add(a, b):\n"
        "    return a + b\n\n"
        "def subtract(a, b):\n"
        "    return a - b\n"
    )

    diff = generate_diff(old_code, new_code, "v0.py", "v1.py")

    assert "+def subtract(a, b):" in diff