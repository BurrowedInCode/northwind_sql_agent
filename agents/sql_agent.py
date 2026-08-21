from pathlib import Path

from pydantic import BaseModel, Field, ConfigDict
from openrouter import OpenRouter


class SQLGeneration(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sql: str = Field(min_length=1)
    explanation: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


def build_system_prompt(prompt_path: str | Path, schema: str) -> str:
    prompt_text: str = Path(prompt_path).read_text(encoding="utf-8")

    return f"{prompt_text}\n\n# Database Schema\n\n{schema}"


def generate_sql(
    client: OpenRouter, model: str, system_prompt: str, user_query: str
) -> SQLGeneration:

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

    return SQLGeneration.model_validate_json(json_data=content)
