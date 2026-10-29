EXAMPLES = {
    "division_by_zero": {
        "description": "Handle division by zero safely",

        "original": '''
def divide(a, b):
    return a / b
''',

        "candidate": '''
def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
''',

        "tests": '''
import pytest
from implementation import divide


def test_normal_division():
    assert divide(10, 2) == 5


def test_division_by_zero():
    with pytest.raises(ValueError):
        divide(10, 0)
'''
    },

    "incorrect_discount": {
        "description": "Calculate percentage discounts correctly",

        "original": '''
def apply_discount(price, discount):
    return price - discount
''',

        "candidate": '''
def apply_discount(price, discount):
    return price * (1 - discount / 100)
''',

        "tests": '''
from implementation import apply_discount


def test_discount():
    assert apply_discount(100, 20) == 80
    assert apply_discount(200, 10) == 180
'''
    },

    "total_regression": {
        "description": "Detect a regression in total calculation",

        "original": '''
def calculate_total(price, quantity):
    return price * quantity
''',

        "candidate": '''
def calculate_total(price, quantity):
    return price + quantity
''',

        "tests": '''
from implementation import calculate_total


def test_calculate_total():
    assert calculate_total(100, 3) == 300
'''
    }
}