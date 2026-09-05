from pathlib import Path
from math import isclose

from pydantic import BaseModel, ConfigDict, Field

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
