import ast


def extract_functions(code):
    """Extract function names and their source locations."""

    tree = ast.parse(code)

    functions = {}

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions[node.name] = {
                "start_line": node.lineno,
                "end_line": node.end_lineno,
                "ast": ast.dump(node, include_attributes=False)
            }

    return functions


def track_changed_functions(old_code, new_code):
    """Identify added, removed, modified and unchanged functions."""

    old_functions = extract_functions(old_code)
    new_functions = extract_functions(new_code)

    old_names = set(old_functions.keys())
    new_names = set(new_functions.keys())

    added = new_names - old_names
    removed = old_names - new_names
    common = old_names & new_names

    modified = set()
    unchanged = set()

    for name in common:
        old_ast = old_functions[name]["ast"]
        new_ast = new_functions[name]["ast"]

        if old_ast != new_ast:
            modified.add(name)
        else:
            unchanged.add(name)

    return {
        "added": sorted(added),
        "removed": sorted(removed),
        "modified": sorted(modified),
        "unchanged": sorted(unchanged)
    }