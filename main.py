import os
from pathlib import Path

from openrouter import OpenRouter

from agents.sql_agent import (
    SQLAgentResult,
    build_system_prompt,
    run_sql_agent,
)
from db.schema import get_schema, DATABASE_PATH

PROJECT_ROOT = Path(__file__).resolve().parent
PROMPT_PATH = PROJECT_ROOT / "prompts" / "sql_agent.v1.md"


def main():
    api_key = os.environ["OPENROUTER_API_KEY"]
    model = os.environ["OPENROUTER_MODEL"]
    northwind_schema = get_schema(database_path=DATABASE_PATH)
    system_prompt = build_system_prompt(PROMPT_PATH, northwind_schema)
    user_query = input("Question: ").strip()

    with OpenRouter(api_key=api_key) as router:
        agent_result: SQLAgentResult = run_sql_agent(
            client=router,
            model=model,
            system_prompt=system_prompt,
            user_query=user_query,
            database_path=DATABASE_PATH,
        )

    generation = agent_result.result.generation
    usage = agent_result.result.usage

    print("SQL: ", generation.sql)
    print("explanation: ", generation.explanation)
    print("confidence: ", generation.confidence)
    print("columns: ", agent_result.columns)
    print("rows: ", agent_result.rows)
    print("prompt tokens:", usage.prompt_tokens)
    print("completion tokens:", usage.completion_tokens)
    print("total tokens:", usage.total_tokens)
    print("cost:", usage.cost)


if __name__ == "__main__":
    main()
