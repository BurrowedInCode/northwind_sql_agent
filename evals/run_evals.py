from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field


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
