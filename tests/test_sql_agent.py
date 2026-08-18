import pytest
from pydantic import ValidationError

from agents.sql_agent import SQLGeneration


def test_sql_generation_accepts_valid_data():
    sql_generation = SQLGeneration(
        sql="SELECT * FROM products",
        explanation="query to gather all products",
        confidence=1.0,
    )

    assert sql_generation.sql == "SELECT * FROM products"
    assert sql_generation.explanation == "query to gather all products"
    assert sql_generation.confidence == 1


def test_sql_generation_accepts_zero_confidence():
    sql_generation = SQLGeneration(
        sql="SELECT ProductName FROM Products",
        explanation="Returns product names",
        confidence=0.0,
    )

    assert sql_generation.confidence == 0.0


def test_sql_generation_rejects_confidence_below_zero():
    with pytest.raises(ValidationError):
        SQLGeneration(
            sql="SELECT * FROM products",
            explanation="query to gather all products",
            confidence=-0.1,
        )


def test_sql_generation_rejects_confidence_above_one():
    with pytest.raises(ValidationError):
        SQLGeneration(
            sql="SELECT * FROM products",
            explanation="query to gather all products",
            confidence=1.1,
        )


def test_sql_generation_rejects_empty_sql():
    with pytest.raises(ValidationError):
        SQLGeneration(
            sql="",
            explanation="query to gather all products",
            confidence=0.5,
        )


def test_sql_generation_rejects_missing_sql():
    with pytest.raises(ValidationError):
        SQLGeneration(
            explanation="query to gather all products",
            confidence=0.5,
        )
