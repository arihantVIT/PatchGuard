from .diagnosis import diagnose_patch


original_code = """
def calculate_total(price, discount):
    return price - discount
"""

modified_code = """
def calculate_total(price, discount):
    if discount < 0:
        raise ValueError("Discount cannot be negative")

    return price - discount
"""

diff = """
 def calculate_total(price, discount):
+    if discount < 0:
+        raise ValueError("Discount cannot be negative")
+
     return price - discount
"""

changed_functions = {
    "modified": ["calculate_total"],
    "added": [],
    "removed": [],
    "unchanged": [],
}

test_results = {
    "status": "passed",
    "passed": 2,
    "failed": 0,
    "details": [
        "test_valid_discount: PASSED",
        "test_negative_discount: PASSED",
    ],
}


if __name__ == "__main__":
    report = diagnose_patch(
        original_code=original_code,
        modified_code=modified_code,
        diff=diff,
        changed_functions=changed_functions,
        test_results=test_results,
    )

    print("\n========== PATCHGUARD DIAGNOSIS ==========\n")
    print(report)