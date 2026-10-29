import ast


"""
it should:
1. Parse both versions.
2. Find the withdraw function in each AST.
3. Compare their structures.
4. Report whether the function changed.
5. If it changed, identify the old and new arithmetic operators.
"""



old_code = """
def withdraw(balance, amount):
    if amount <= balance:
        return balance - amount

    return balance
"""

new_code = """
def withdraw(balance, amount):
    if amount <= balance:
        return balance + amount

    return balance
"""


def extract_functions(code):
    tree = ast.parse(code)

    functions = {}

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions[node.name] = node

    return functions


def compare_versions(old_code, new_code):
    old_functions = extract_functions(old_code)
    new_functions = extract_functions(new_code)

    old_names = set(old_functions)
    new_names = set(new_functions)

    added = new_names - old_names
    removed = old_names - new_names

    modified = set()
    unchanged = set()

    for name in old_names & new_names:
        old_structure = ast.dump(
            old_functions[name],
            include_attributes=False
        )

        new_structure = ast.dump(
            new_functions[name],
            include_attributes=False
        )

        if old_structure != new_structure:
            modified.add(name)
        else:
            unchanged.add(name)

    return {
        "added": sorted(added),
        "removed": sorted(removed),
        "modified": sorted(modified),
        "unchanged": sorted(unchanged),
    }


print(compare_versions(old_code, new_code))
