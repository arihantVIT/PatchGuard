from patchguard.function_tracker import (
    extract_functions,
    track_changed_functions,
)


def test_extract_functions():
    code = """
def add(a, b):
    return a + b


def subtract(a, b):
    return a - b
"""

    functions = extract_functions(code)

    assert set(functions.keys()) == {"add", "subtract"}
    assert functions["add"]["start_line"] == 2
    assert functions["add"]["end_line"] == 3


def test_added_function():
    old_code = """
def add(a, b):
    return a + b
"""

    new_code = """
def add(a, b):
    return a + b


def subtract(a, b):
    return a - b
"""

    result = track_changed_functions(old_code, new_code)

    assert result["added"] == ["subtract"]
    assert result["removed"] == []
    assert result["modified"] == []
    assert result["unchanged"] == ["add"]


def test_removed_function():
    old_code = """
def add(a, b):
    return a + b


def subtract(a, b):
    return a - b
"""

    new_code = """
def add(a, b):
    return a + b
"""

    result = track_changed_functions(old_code, new_code)

    assert result["added"] == []
    assert result["removed"] == ["subtract"]
    assert result["modified"] == []
    assert result["unchanged"] == ["add"]


def test_modified_function():
    old_code = """
def withdraw(balance, amount):
    return balance - amount
"""

    new_code = """
def withdraw(balance, amount):
    return balance + amount
"""

    result = track_changed_functions(old_code, new_code)

    assert result["added"] == []
    assert result["removed"] == []
    assert result["modified"] == ["withdraw"]
    assert result["unchanged"] == []


def test_unchanged_function():
    code = """
def add(a, b):
    return a + b
"""

    result = track_changed_functions(code, code)

    assert result["unchanged"] == ["add"]
    assert result["modified"] == []