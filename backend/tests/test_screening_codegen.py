"""Tests for LLM screening-code extract / safety / CSV load path."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from screening.codegen import (
    extract_screening_code,
    fixture_screening_code,
    validate_screening_code_safety,
)
from screening.engine import (
    finalize_screening_dict,
    load_screening_csv,
    run_generated_screener,
)


def test_extract_and_safety_ok():
    text = """```python
def build_screening_csv(stocks_dir, start_date, end_date, out_csv):
    pd.DataFrame(columns=['date', 'ticker']).to_csv(out_csv, index=False)
```"""
    code = extract_screening_code(text)
    assert "def build_screening_csv" in code
    validate_screening_code_safety(code)


def test_rejects_imports():
    bad = """
def build_screening_csv(stocks_dir, start_date, end_date, out_csv):
    import os
    pass
"""
    try:
        validate_screening_code_safety(bad)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "import" in str(e).lower()


def test_rejects_eval():
    bad = """
def build_screening_csv(stocks_dir, start_date, end_date, out_csv):
    eval('1')
"""
    try:
        validate_screening_code_safety(bad)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "eval" in str(e).lower() or "Forbidden" in str(e)


def test_load_csv_and_finalize():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "s.csv")
        with open(path, "w") as f:
            f.write("date,ticker\n2015-01-02,aapl\n2015-01-02,msft\n2015-01-05,aapl\n")
        d = load_screening_csv(path)
        assert d["2015-01-02"] == ["AAPL", "MSFT"]
        final, logs = finalize_screening_dict(d, max_unique_tickers=10, max_tickers_per_day=50)
        assert "AAPL" in final["2015-01-02"]
        assert any("result:" in x for x in logs)


def test_run_empty_screener_writes_csv():
    code = """
def build_screening_csv(stocks_dir, start_date, end_date, out_csv):
    pd.DataFrame(columns=['date', 'ticker']).to_csv(out_csv, index=False)
"""
    with tempfile.TemporaryDirectory() as td:
        # empty result is ok — dict empty
        d, ui, csv_path = run_generated_screener(
            code,
            start_date="2015-01-01",
            end_date="2015-01-10",
            dataset_path=td,  # no stock files
            out_dir=td,
            timeout_sec=30,
        )
        assert d == {}
        assert os.path.isfile(csv_path)
        assert "def build_screening_csv" in ui


def test_fixture_code_present():
    code = fixture_screening_code()
    assert "def build_screening_csv" in code
    validate_screening_code_safety(code)


if __name__ == "__main__":
    test_extract_and_safety_ok()
    test_rejects_imports()
    test_rejects_eval()
    test_load_csv_and_finalize()
    test_run_empty_screener_writes_csv()
    test_fixture_code_present()
    print("test_screening_codegen OK")
