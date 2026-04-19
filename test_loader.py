"""tests/test_loader.py — Unit tests for DatasetLoader."""
import pytest
import pandas as pd
import tempfile
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.loader import DatasetLoader


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write_csv(content: str, suffix: str = ".csv") -> Path:
    """Geçici bir CSV dosyası oluşturur ve Path döner."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=suffix, delete=False, encoding="utf-8"
    )
    tmp.write(content)
    tmp.close()
    return Path(tmp.name)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def loader():
    return DatasetLoader()


@pytest.fixture
def valid_csv(tmp_path):
    path = tmp_path / "sample.csv"
    path.write_text("name,age,score\nAlice,30,95.5\nBob,25,88.0\nCarol,35,72.0\n", encoding="utf-8")
    return path


@pytest.fixture
def empty_csv(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("name,age\n", encoding="utf-8")
    return path


@pytest.fixture
def duplicate_cols_csv(tmp_path):
    path = tmp_path / "dupes.csv"
    path.write_text("col,col,other\n1,2,3\n4,5,6\n", encoding="utf-8")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Normal Loading
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalLoading:

    def test_loads_valid_csv(self, loader, valid_csv):
        df = loader.load(valid_csv)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 3)

    def test_columns_correct(self, loader, valid_csv):
        df = loader.load(valid_csv)
        assert list(df.columns) == ["name", "age", "score"]

    def test_data_types_inferred(self, loader, valid_csv):
        df = loader.load(valid_csv)
        assert pd.api.types.is_numeric_dtype(df["age"])
        assert pd.api.types.is_numeric_dtype(df["score"])

    def test_returns_dataframe(self, loader, valid_csv):
        result = loader.load(valid_csv)
        assert isinstance(result, pd.DataFrame)

    def test_accepts_path_object(self, loader, valid_csv):
        """Path nesnesi ile çağrılabilmeli."""
        df = loader.load(Path(valid_csv))
        assert not df.empty

    def test_accepts_string_path(self, loader, valid_csv):
        """String path ile çağrılabilmeli."""
        df = loader.load(str(valid_csv))
        assert not df.empty


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Validation Errors
# ─────────────────────────────────────────────────────────────────────────────

class TestValidationErrors:

    def test_file_not_found_raises(self, loader):
        with pytest.raises(FileNotFoundError):
            loader.load("nonexistent_file_12345.csv")

    def test_unsupported_extension_raises(self, loader, tmp_path):
        path = tmp_path / "data.xlsx"
        path.write_text("x,y\n1,2\n")
        with pytest.raises(ValueError, match="Unsupported file type"):
            loader.load(path)

    def test_txt_extension_raises(self, loader, tmp_path):
        path = tmp_path / "data.txt"
        path.write_text("a,b\n1,2\n")
        with pytest.raises(ValueError):
            loader.load(path)

    def test_empty_csv_raises(self, loader, empty_csv):
        with pytest.raises(ValueError, match="Empty dataset"):
            loader.load(empty_csv)

    def test_duplicate_columns_raises(self, loader, duplicate_cols_csv):
        with pytest.raises(ValueError, match="Duplicate column"):
            loader.load(duplicate_cols_csv)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Edge Cases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_single_row_csv(self, loader, tmp_path):
        path = tmp_path / "single.csv"
        path.write_text("x,y\n1,2\n", encoding="utf-8")
        df = loader.load(path)
        assert df.shape == (1, 2)

    def test_single_column_csv(self, loader, tmp_path):
        path = tmp_path / "onecol.csv"
        path.write_text("value\n10\n20\n30\n", encoding="utf-8")
        df = loader.load(path)
        assert df.shape == (3, 1)

    def test_csv_with_spaces_in_values(self, loader, tmp_path):
        path = tmp_path / "spaces.csv"
        path.write_text("city,pop\n'New York',8000000\n'Los Angeles',4000000\n", encoding="utf-8")
        df = loader.load(path)
        assert not df.empty

    def test_numeric_heavy_csv(self, loader, tmp_path):
        """Tamamen sayısal bir CSV doğru yüklenmeli."""
        path = tmp_path / "nums.csv"
        rows = "\n".join([f"{i},{i*2},{i*3}" for i in range(1, 101)])
        path.write_text(f"a,b,c\n{rows}\n", encoding="utf-8")
        df = loader.load(path)
        assert df.shape == (100, 3)
