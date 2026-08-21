import os
from pathlib import Path

from openrouter import OpenRouter

from agents.sql_agent import build_system_prompt, generate_sql
from db.execute import execute_query
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
        generation = generate_sql(
            client=router,
            model=model,
            system_prompt=system_prompt,
            user_query=user_query,
        )

        print("SQL: ", generation.sql)
        print("explanation: ", generation.explanation)
        print("confidence: ", generation.confidence)

    columns, rows = execute_query(database_path=DATABASE_PATH, sql=generation.sql)
    print("columns: ", columns)
    print("rows: ", rows)


if __name__ == "__main__":
    main()
