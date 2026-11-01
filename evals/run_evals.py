from pathlib import Path
from math import isclose

from openrouter import OpenRouter
from pydantic import BaseModel, ConfigDict, Field

from agents.sql_agent import SQLAgentResult, TokenUsage, run_sql_agent
from db.execute import execute_query

type CellValue = str | int | float | None
type Row = list[CellValue]
type Rows = list[Row]


class GoldenCase(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    question: str = Field(min_length=1)
    expected_sql: str = Field(min_length=1)
    ordered: bool


class EvalCaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    case_id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    passed: bool
    generated_sql: str | None = Field(default=None, min_length=1)
    confidence: float | None = Field(default=None, ge=0, le=1)
    usage: TokenUsage | None = None
    error: str | None = Field(default=None, min_length=1)


def load_golden_cases(golden_file: str | Path) -> list[GoldenCase]:
    cases: list[GoldenCase] = []
    seen_ids: set[str] = set()
    with open(golden_file, encoding="utf-8") as file:
        for line in file:
            golden_case = GoldenCase.model_validate_json(line)
            if golden_case.id in seen_ids:
                raise ValueError(f"duplicate golden case id: {golden_case.id}")
            seen_ids.add(golden_case.id)
            cases.append(golden_case)
    return cases


def row_matches(expected_row: Row, actual_row: Row) -> bool:
    if len(expected_row) != len(actual_row):
        return False

    for expected_value, actual_value in zip(expected_row, actual_row):
        if isinstance(expected_value, (int, float)) and isinstance(
            actual_value, (int, float)
        ):
            if not isclose(
                expected_value,
                actual_value,
                rel_tol=1e-9,
                abs_tol=1e-6,
            ):
                return False
        elif expected_value != actual_value:
            return False
    return True


def rows_match(expected_rows: Rows, actual_rows: Rows, ordered: bool) -> bool:
    if len(expected_rows) != len(actual_rows):
        return False

    if ordered:
        for expected_row, actual_row in zip(expected_rows, actual_rows):
            if not row_matches(expected_row, actual_row):
                return False
        return True

    unmatched_actual_rows = actual_rows.copy()

    for expected_row in expected_rows:
        matched_index: int | None = None

        for index, actual_row in enumerate(unmatched_actual_rows):
            if row_matches(expected_row, actual_row):
                matched_index = index
                break
        if matched_index is None:
            return False

        del unmatched_actual_rows[matched_index]

    return True


def evaluate_case(
    case: GoldenCase,
    client: OpenRouter,
    model: str,
    system_prompt: str,
    database_path: str | Path,
    row_limit: int = 100,
) -> EvalCaseResult:

    _, expected_row_tuples = execute_query(
        database_path=database_path, sql=case.expected_sql, row_limit=row_limit
    )
    expected_rows: Rows = [list(row) for row in expected_row_tuples]
    agent_result: SQLAgentResult = run_sql_agent(
        client=client,
        model=model,
        system_prompt=system_prompt,
        user_query=case.question,
        database_path=database_path,
        row_limit=row_limit,
    )
    passed = rows_match(
        expected_rows=expected_rows, actual_rows=agent_result.rows, ordered=case.ordered
    )
    eval_case_result = EvalCaseResult(
        case_id=case.id,
        category=case.category,
        passed=passed,
        generated_sql=agent_result.result.generation.sql,
        confidence=agent_result.result.generation.confidence,
        usage=agent_result.result.usage,
    )
    return eval_case_result
