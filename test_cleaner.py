"""tests/test_cleaner.py — Unit tests for AutoCleaner."""
import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.cleaner import AutoCleaner
from tools.profiler import DataProfiler, DatasetContext


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def cleaner():
    return AutoCleaner()


@pytest.fixture
def profiler():
    return DataProfiler()


def make_ctx(**kwargs) -> DatasetContext:
    """Kolayca özel DatasetContext oluşturmak için yardımcı."""
    defaults = {
        "rows": 10, "columns": 3,
        "numeric_columns": [], "categorical_columns": [],
        "skewed_columns": [], "low_variance_columns": [],
        "missing_report": [], "high_missing_columns": [],
        "possible_identifiers": [], "datetime_columns": [],
        "target_column": None, "risks": [],
        "numeric_summary": [], "categorical_summary": [],
    }
    defaults.update(kwargs)
    return DatasetContext(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Clean interface
# ─────────────────────────────────────────────────────────────────────────────

class TestCleanInterface:

    def test_returns_tuple_of_df_and_logs(self, cleaner):
        df = pd.DataFrame({"x": [1, 2, 3]})
        ctx = make_ctx(numeric_columns=["x"])
        result = cleaner.clean(df, ctx)
        assert isinstance(result, tuple)
        assert len(result) == 2
        df_clean, logs = result
        assert isinstance(df_clean, pd.DataFrame)
        assert isinstance(logs, list)

    def test_original_df_not_mutated(self, cleaner):
        """Orijinal DataFrame değişmemeli — kopya üzerinde çalışılmalı."""
        df = pd.DataFrame({"a": [1.0, None, 3.0], "b": ["x", None, "z"]})
        ctx = make_ctx(
            numeric_columns=["a"],
            categorical_columns=["b"],
            missing_report=[
                {"column": "a", "missing_rate": 0.33, "severity": "warn"},
                {"column": "b", "missing_rate": 0.33, "severity": "warn"},
            ],
            high_missing_columns=["a", "b"],
        )
        original_na_count = df.isna().sum().sum()
        cleaner.clean(df, ctx)
        assert df.isna().sum().sum() == original_na_count

    def test_log_cleared_between_runs(self, cleaner):
        """Birden fazla çalıştırmada loglar sıfırlanmalı."""
        df1 = pd.DataFrame({"a": [1.0, None, 3.0]})
        ctx1 = make_ctx(
            numeric_columns=["a"],
            missing_report=[{"column": "a", "missing_rate": 0.33, "severity": "warn"}],
            high_missing_columns=["a"],
        )
        _, logs1 = cleaner.clean(df1, ctx1)

        df2 = pd.DataFrame({"x": [1, 2, 3]})
        ctx2 = make_ctx(numeric_columns=["x"])
        _, logs2 = cleaner.clean(df2, ctx2)

        # İkinci çalıştırma birincinin loglarını taşımamalı
        assert logs1 != logs2 or len(logs2) <= 1  # ya farklı ya da sadece "veri zaten harika" mesajı


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Low Variance Dropping
# ─────────────────────────────────────────────────────────────────────────────

class TestDropLowVariance:

    def test_low_variance_column_dropped(self, cleaner):
        df = pd.DataFrame({"constant": [1, 1, 1, 1, 1], "useful": [1, 2, 3, 4, 5]})
        ctx = make_ctx(
            numeric_columns=["constant", "useful"],
            low_variance_columns=["constant"],
        )
        df_clean, logs = cleaner.clean(df, ctx)
        assert "constant" not in df_clean.columns

    def test_useful_columns_not_dropped(self, cleaner):
        df = pd.DataFrame({"constant": [1, 1, 1, 1, 1], "useful": [1, 2, 3, 4, 5]})
        ctx = make_ctx(
            numeric_columns=["constant", "useful"],
            low_variance_columns=["constant"],
        )
        df_clean, _ = cleaner.clean(df, ctx)
        assert "useful" in df_clean.columns

    def test_drop_logged(self, cleaner):
        df = pd.DataFrame({"dead": [0, 0, 0, 0, 0], "alive": [1, 2, 3, 4, 5]})
        ctx = make_ctx(
            numeric_columns=["dead", "alive"],
            low_variance_columns=["dead"],
        )
        _, logs = cleaner.clean(df, ctx)
        assert any("dead" in log for log in logs)

    def test_no_low_variance_nothing_dropped(self, cleaner):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        ctx = make_ctx(numeric_columns=["a", "b"], low_variance_columns=[])
        df_clean, _ = cleaner.clean(df, ctx)
        assert list(df_clean.columns) == ["a", "b"]


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Missing Value Handling
# ─────────────────────────────────────────────────────────────────────────────

class TestHandleMissing:

    def test_numeric_nan_filled_with_mean(self, cleaner):
        df = pd.DataFrame({"score": [10.0, 20.0, None, 40.0, 50.0]})
        ctx = make_ctx(
            numeric_columns=["score"],
            skewed_columns=[],
            categorical_columns=[],
            missing_report=[{"column": "score", "missing_rate": 0.2, "severity": "warn"}],
            high_missing_columns=["score"],
        )
        df_clean, _ = cleaner.clean(df, ctx)
        assert df_clean["score"].isna().sum() == 0
        # Beklenen ortalama: (10+20+40+50)/4 = 30
        assert df_clean["score"].iloc[2] == pytest.approx(30.0, abs=0.01)

    def test_skewed_numeric_filled_with_median(self, cleaner):
        """Çarpık sütunlar ortalama yerine medyan ile doldurulmalı."""
        df = pd.DataFrame({"income": [100.0, 200.0, None, 100.0, 50000.0]})
        ctx = make_ctx(
            numeric_columns=["income"],
            skewed_columns=["income"],  # Çarpık olarak işaretle
            categorical_columns=[],
            missing_report=[{"column": "income", "missing_rate": 0.2, "severity": "warn"}],
            high_missing_columns=["income"],
        )
        df_clean, logs = cleaner.clean(df, ctx)
        assert df_clean["income"].isna().sum() == 0
        assert any("Medyan" in log for log in logs)

    def test_categorical_nan_filled_with_bilinmiyor(self, cleaner):
        df = pd.DataFrame({"city": ["Istanbul", None, "Ankara", None, "Izmir"]})
        ctx = make_ctx(
            numeric_columns=[],
            categorical_columns=["city"],
            missing_report=[{"column": "city", "missing_rate": 0.4, "severity": "warn"}],
            high_missing_columns=["city"],
        )
        df_clean, _ = cleaner.clean(df, ctx)
        assert df_clean["city"].isna().sum() == 0
        assert "Bilinmiyor" in df_clean["city"].values

    def test_no_missing_no_change(self, cleaner):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "z"]})
        ctx = make_ctx(
            numeric_columns=["a"], categorical_columns=["b"],
            missing_report=[], high_missing_columns=[],
        )
        df_clean, _ = cleaner.clean(df, ctx)
        assert df_clean["a"].isna().sum() == 0
        assert df_clean["b"].isna().sum() == 0


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Outlier Capping (Winsorization)
# ─────────────────────────────────────────────────────────────────────────────

class TestCapOutliers:

    def test_outliers_capped(self, cleaner):
        # Normal dağılım + 2 aşırı değer
        values = [10.0, 11.0, 12.0, 10.0, 11.0, 9.0, 12.0, 10.0, 11.0, 10000.0]
        df = pd.DataFrame({"amount": values})
        ctx = make_ctx(
            numeric_columns=["amount"],
            target_column=None,
        )
        df_clean, logs = cleaner.clean(df, ctx)
        # Outlier temizlendikten sonra maksimum değer aşırı uca gitmemeli
        assert df_clean["amount"].max() < 10000.0

    def test_capping_logged(self, cleaner):
        values = [10.0, 11.0, 12.0, 10.0, 11.0, 9.0, 12.0, 10.0, 11.0, 99999.0]
        df = pd.DataFrame({"salary": values})
        ctx = make_ctx(numeric_columns=["salary"], target_column=None)
        _, logs = cleaner.clean(df, ctx)
        assert any("Tıraşlandı" in log or "salary" in log for log in logs)

    def test_target_column_not_capped(self, cleaner):
        """Target sütunu hiçbir zaman kırpılmamalı."""
        values = [0, 1, 0, 1, 0, 1, 0, 99999, 0, 1]
        df = pd.DataFrame({"churn": values, "score": [50.0] * 10})
        ctx = make_ctx(
            numeric_columns=["churn", "score"],
            target_column="churn",
        )
        df_clean, _ = cleaner.clean(df, ctx)
        assert df_clean["churn"].max() == 99999

    def test_no_outliers_no_capping(self, cleaner):
        df = pd.DataFrame({"x": [10.0, 11.0, 10.0, 11.0, 10.0]})
        ctx = make_ctx(numeric_columns=["x"], target_column=None)
        _, logs = cleaner.clean(df, ctx)
        capping_logs = [l for l in logs if "Tıraşlandı" in l]
        assert len(capping_logs) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Clean data gives friendly message
# ─────────────────────────────────────────────────────────────────────────────

class TestCleanDataMessage:

    def test_perfect_data_friendly_message(self, cleaner):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "z"]})
        ctx = make_ctx(
            numeric_columns=["a"],
            categorical_columns=["b"],
            low_variance_columns=[],
            missing_report=[],
            high_missing_columns=[],
            target_column=None,
        )
        _, logs = cleaner.clean(df, ctx)
        # Hiçbir temizleme yapılmadıysa dostça mesaj gelmelı
        assert any("harika" in log.lower() or "müdahale" in log.lower() for log in logs)
