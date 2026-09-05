from pathlib import Path

from pydantic import ValidationError
import pytest
from evals.run_evals import GoldenCase, load_golden_cases, rows_match, Rows


def test_golden_case():
    golden_case = GoldenCase(
        id="case_01",
        category="simple_selection",
        question="List first 10 products",
        expected_sql="SELECT * FROM Products ORDER BY ProductID ASC LIMIT 10",
        ordered=True,
    )

    assert golden_case.id == "case_01"
    assert golden_case.category == "simple_selection"
    assert golden_case.question == "List first 10 products"
    assert (
        golden_case.expected_sql
        == "SELECT * FROM Products ORDER BY ProductID ASC LIMIT 10"
    )
    assert golden_case.ordered is True


def test_golden_case_rejects_whitespace_only_question():
    with pytest.raises(ValidationError):
        GoldenCase(
            id="1",
            category="Products",
            question=" ",
            expected_sql="SELECT * FROM Products LIMIT 10 ORDER BY ProductID DESC",
            ordered=True,
        )


def test_load_golden_cases(tmp_path: Path):
    dataset_path = tmp_path / "golden.jsonl"

    dataset_path.write_text(
        """{"id":"case_01","category":"scalar_aggregate","question":"Count all orders.","expected_sql":"SELECT COUNT(*) FROM Orders;","ordered":false}
{"id":"case_02","category":"simple_selection","question":"Return one.","expected_sql":"SELECT 1;","ordered":true}
""",
        encoding="utf-8",
    )

    cases = load_golden_cases(dataset_path)

    assert len(cases) == 2
    assert [case.id for case in cases] == ["case_01", "case_02"]
    assert all(isinstance(case, GoldenCase) for case in cases)
    assert cases[0].expected_sql == "SELECT COUNT(*) FROM Orders;"


def test_load_golden_cases_rejects_duplicate_id(tmp_path: Path):
    dataset_path = tmp_path / "golden.jsonl"

    dataset_path.write_text(
        """{"id":"case_01","category":"scalar_aggregate","question":"Count all orders.","expected_sql":"SELECT COUNT(*) FROM Orders;","ordered":false}
{"id":"case_01","category":"simple_selection","question":"Return one.","expected_sql":"SELECT 1;","ordered":true}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate"):
        load_golden_cases(dataset_path)


def test_load_golden_cases_contains_all_cases():
    golden_root = Path(__file__).resolve().parent.parent
    golden_path = golden_root / "evals" / "golden.jsonl"

    cases = load_golden_cases(golden_path)

    assert len(cases) == 10

    expected_ids = [f"case_{number:02d}" for number in range(1, 11)]

    assert [case.id for case in cases] == expected_ids


def test_rows_match_ordered():
    expected_rows: Rows = [[1, "Chai"], [2, "Chang"]]
    actual_rows: Rows = [[1, "Chai"], [2, "Chang"]]

    assert rows_match(expected_rows, actual_rows, ordered=True)

    reversed_rows = list(reversed(actual_rows))

    assert not rows_match(expected_rows, reversed_rows, ordered=True)


def test_rows_match_order_ignored():
    expected_rows: Rows = [[1, "Chai"], [2, "Chang"]]
    actual_rows: Rows = [[1, "Chai"], [2, "Chang"]]

    assert rows_match(
        expected_rows=expected_rows,
        actual_rows=list(reversed(actual_rows)),
        ordered=False,
    )


def test_rows_match_duplicate_counts_matter():
    expected_rows: Rows = [[1, "Chai"], [1, "Chai"]]
    actual_rows: Rows = [
        [1, "Chai"],
    ]

    assert not rows_match(expected_rows, actual_rows, ordered=False)


def test_rows_match_float_tolerance():
    expected_rows: Rows = [[38, "Cote de Blaye", 53265895.235]]
    nearly_equal_rows: Rows = [[38, "Cote de Blaye", 53265895.23500001]]
    different_rows: Rows = [[38, "Cote de Blaye", 53265896.235]]

    assert rows_match(expected_rows, nearly_equal_rows, True)
    assert not rows_match(expected_rows, different_rows, True)


def test_rows_match_unordered_float_tolerance():
    expected_rows: Rows = [
        [38, "Cote de Blaye", 53265895.235],
        [29, "Thuringer Rostbratwurst", 24623469.232],
    ]
    nearly_equal_rows: Rows = [
        [29, "Thuringer Rostbratwurst", 24623469.23200001],
        [38, "Cote de Blaye", 53265895.23500001],
    ]
    different_rows: Rows = [
        [29, "Thuringer Rostbratwurst", 24623470.232],
        [38, "Cote de Blaye", 53265895.235],
    ]

    assert rows_match(expected_rows, nearly_equal_rows, ordered=False)
    assert not rows_match(expected_rows, different_rows, ordered=False)
