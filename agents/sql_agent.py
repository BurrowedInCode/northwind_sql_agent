from pathlib import Path

from pydantic import BaseModel, Field, ConfigDict
from openrouter import OpenRouter


class SQLGeneration(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sql: str = Field(min_length=1)
    explanation: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class TokenUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    cost: float | None = Field(default=None, ge=0)


class SQLGenerationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    generation: SQLGeneration
    usage: TokenUsage


def build_system_prompt(prompt_path: str | Path, schema: str) -> str:
    prompt_text: str = Path(prompt_path).read_text(encoding="utf-8")

    return f"{prompt_text}\n\n# Database Schema\n\n{schema}"


def generate_sql(
    client: OpenRouter, model: str, system_prompt: str, user_query: str
) -> SQLGenerationResult:
    response = client.chat.send(
        provider={"require_parameters": True},
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "sql_generation",
                "strict": True,
                "schema_": SQLGeneration.model_json_schema(),
            },
        },
    )
    content = response.choices[0].message.content
    if not isinstance(content, str):
        raise ValueError("OpenRouter returned non-text content")

    generation = SQLGeneration.model_validate_json(json_data=content)

    usage = response.usage

    if usage is None:
        raise ValueError("OpenRouter returned no usage.")

    cost = usage.cost
    if isinstance(cost, (int, float)):
        normalized_cost = float(cost)
    else:
        normalized_cost = None

    token_usage = TokenUsage(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        cost=normalized_cost,
    )

    return SQLGenerationResult(generation=generation, usage=token_usage)
