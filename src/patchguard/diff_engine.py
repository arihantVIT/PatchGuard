import difflib


def generate_diff(old_code, new_code, old_name, new_name):
    """Generate a unified diff between two code versions."""

    # Convert source code strings into lists of lines
    old_lines = old_code.splitlines(keepends=True)
    new_lines = new_code.splitlines(keepends=True)

    # Generate the unified diff
    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=old_name,
        tofile=new_name
    )

    # Convert the diff iterator into a single string
    return "".join(diff)



