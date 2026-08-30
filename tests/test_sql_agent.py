from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from agents.sql_agent import (
    SQLGeneration,
    SQLGenerationResult,
    TokenUsage,
    build_system_prompt,
    generate_sql,
)


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

    response.usage = Mock(
        prompt_tokens=100,
        completion_tokens=25,
        total_tokens=125,
        cost=0.01,
    )

    client.chat.send.return_value = response

    result = generate_sql(
        client=client,
        model="test-model",
        system_prompt="system instructions",
        user_query="List product names",
    )

    assert isinstance(result, SQLGenerationResult)
    assert result.generation.sql == "SELECT ProductName FROM Products"
    assert result.generation.explanation == "Returns product names."
    assert result.generation.confidence == 0.9
    assert result.usage.prompt_tokens == 100
    assert result.usage.completion_tokens == 25
    assert result.usage.total_tokens == 125
    assert result.usage.cost == 0.01

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

    assert request["provider"] == {"require_parameters": True}


def test_generate_sql_rejects_missing_usage():
    client = Mock()
    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message.content = """
        {
    "sql": "SELECT 1",
    "explanation": "Returns one.",
    "confidence": 1.0
    }
    """

    response.usage = None

    client.chat.send.return_value = response

    with pytest.raises(ValueError, match="no usage"):
        generate_sql(
            client=client,
            model="test-model",
            system_prompt="system instructions",
            user_query="List product names",
        )


def test_token_usage_valid():
    token_usage = TokenUsage(
        prompt_tokens=100, completion_tokens=25, total_tokens=125, cost=0.01
    )

    assert token_usage.prompt_tokens == 100
    assert token_usage.completion_tokens == 25
    assert token_usage.total_tokens == 125
    assert token_usage.cost == 0.01


def test_token_usage_omitted_cost():
    token_usage = TokenUsage(prompt_tokens=100, completion_tokens=25, total_tokens=125)

    assert token_usage.cost is None


def test_token_usage_negative_token():
    with pytest.raises(ValidationError):
        TokenUsage(prompt_tokens=-1, completion_tokens=25, total_tokens=125, cost=0.01)


def test_sql_generation_result():
    generation = SQLGeneration(
        sql="SELECT 1",
        explanation="Returns one.",
        confidence=1.0,
    )

    usage = TokenUsage(
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
        cost=0.001,
    )

    result = SQLGenerationResult(
        generation=generation,
        usage=usage,
    )
    assert result.generation == generation
    assert result.usage == usage
    assert result.generation.sql == "SELECT 1"
    assert result.usage.total_tokens == 120
