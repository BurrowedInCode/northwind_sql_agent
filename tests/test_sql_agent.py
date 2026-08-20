from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from agents.sql_agent import SQLGeneration, build_system_prompt, generate_sql


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
        SQLGeneration.model_validate(
            {
                "explanation": "query to gather all products",
                "confidence": 0.5,
            }
        )


def test_sql_generation_rejects_extra_fields():
    payload = {
        "sql": "SELECT * FROM products",
        "explanation": "query to gather all products",
        "confidence": 0.5,
        "table": "products",
    }
    with pytest.raises(ValidationError):
        SQLGeneration.model_validate(payload)


def test_sql_generation_schema_forbids_extra_fields():
    schema = SQLGeneration.model_json_schema()
    assert schema["additionalProperties"] is False


def test_build_system_prompt(tmp_path):
    prompt_path = tmp_path / "prompt.md"
    prompt_text = "# Context\n\n you are an expert at generating SQL"
    prompt_path.write_text(prompt_text, encoding="utf-8")
    schema = """
        CREATE TABLE products (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL);
    """

    system_prompt = build_system_prompt(prompt_path, schema)

    assert "# Database Schema" in system_prompt
    assert prompt_text in system_prompt
    assert schema in system_prompt
    assert system_prompt.index(prompt_text) < system_prompt.index(schema)


def test_generate_sql():
    client = Mock()
    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message.content = """
    {
    "sql": "SELECT ProductName FROM Products",
    "explanation": "Returns product names.",
    "confidence": 0.9
    }
    """
    client.chat.send.return_value = response

    result = generate_sql(
        client=client,
        model="test-model",
        system_prompt="system instructions",
        user_query="List product names",
    )

    assert isinstance(result, SQLGeneration)
    assert result.sql == "SELECT ProductName FROM Products"
    assert result.explanation == "Returns product names."
    assert result.confidence == 0.9

    client.chat.send.assert_called_once()

    request = client.chat.send.call_args.kwargs
    assert request["model"] == "test-model"

    assert request["messages"][0] == {
        "role": "system",
        "content": "system instructions",
    }
    assert request["messages"][1] == {
        "role": "user",
        "content": "List product names",
    }
    assert (
        request["response_format"]["json_schema"]["schema_"]
        == SQLGeneration.model_json_schema()
    )
