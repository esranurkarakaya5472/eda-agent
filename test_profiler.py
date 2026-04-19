"""tests/test_profiler.py — Unit tests for DataProfiler & DatasetContext."""
import pytest
import pandas as pd
import numpy as np
import sys
import os

# ── Proje kök dizinini Python path'e ekle ─────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.profiler import DataProfiler, DatasetContext


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_df():
    """Basit, temiz, küçük DataFrame."""
    return pd.DataFrame({
        "age":    [25, 30, 35, 40, 45],
        "salary": [3000.0, 4000.0, 5000.0, 6000.0, 7000.0],
        "city":   ["Istanbul", "Ankara", "Izmir", "Istanbul", "Ankara"],
        "churn":  [0, 1, 0, 1, 0],
    })


@pytest.fixture
def df_with_missing():
    """Eksik değerli DataFrame."""
    return pd.DataFrame({
        "a": [1.0, None, 3.0, None, 5.0, None, 7.0, None, 9.0, None],
        "b": ["x", None, "y", "z", None, "x", "y", None, "z", "x"],
    })


@pytest.fixture
def df_skewed():
    """Çarpık dağılımlı DataFrame (outlier ağırlıklı)."""
    vals = [1] * 95 + [10000, 20000, 30000, 40000, 50000]
    return pd.DataFrame({"skew_col": vals})


@pytest.fixture
def profiler():
    return DataProfiler()


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Column Classification
# ─────────────────────────────────────────────────────────────────────────────

class TestColumnClassification:

    def test_numeric_columns_detected(self, profiler, simple_df):
        ctx = profiler.profile(simple_df)
        assert "age" in ctx.numeric_columns
        assert "salary" in ctx.numeric_columns

    def test_categorical_columns_detected(self, profiler, simple_df):
        ctx = profiler.profile(simple_df)
        assert "city" in ctx.categorical_columns

    def test_no_false_categorical(self, profiler, simple_df):
        """Sayısal sütunlar kategorik listesine girmemeli."""
        ctx = profiler.profile(simple_df)
        assert "age" not in ctx.categorical_columns
        assert "salary" not in ctx.categorical_columns

    def test_datetime_detection(self, profiler):
        df = pd.DataFrame({
            "event_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
            "value": [1, 2, 3],
        })
        ctx = profiler.profile(df)
        assert "event_date" in ctx.datetime_columns


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Basic Shape
# ─────────────────────────────────────────────────────────────────────────────

class TestBasicShape:

    def test_rows_and_columns(self, profiler, simple_df):
        ctx = profiler.profile(simple_df)
        assert ctx.rows == 5
        assert ctx.columns == 4

    def test_empty_after_clear(self, profiler):
        df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        ctx = profiler.profile(df)
        assert ctx.rows == 3
        assert ctx.columns == 2


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Missing Value Analysis
# ─────────────────────────────────────────────────────────────────────────────

class TestMissingAnalysis:

    def test_no_missing_clean_df(self, profiler, simple_df):
        ctx = profiler.profile(simple_df)
        assert ctx.missing_report == []

    def test_missing_report_generated(self, profiler, df_with_missing):
        ctx = profiler.profile(df_with_missing)
        missing_cols = [r["column"] for r in ctx.missing_report]
        assert "a" in missing_cols
        assert "b" in missing_cols

    def test_missing_rate_calculated_correctly(self, profiler, df_with_missing):
        ctx = profiler.profile(df_with_missing)
        a_report = next(r for r in ctx.missing_report if r["column"] == "a")
        # 5 eksik / 10 toplam = 0.5
        assert a_report["missing_rate"] == pytest.approx(0.5, abs=0.01)

    def test_critical_severity_assigned(self, profiler, df_with_missing):
        ctx = profiler.profile(df_with_missing)
        a_report = next(r for r in ctx.missing_report if r["column"] == "a")
        # %50 > %30 kritik eşiği → severity = "critical"
        assert a_report["severity"] == "critical"

    def test_warn_severity_assigned(self, profiler):
        """5-29% arası eksik → 'warn'."""
        df = pd.DataFrame({"col": [1.0, None, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]})
        ctx = profiler.profile(df)
        assert len(ctx.missing_report) == 1
        assert ctx.missing_report[0]["severity"] == "warn"


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Numeric Summary
# ─────────────────────────────────────────────────────────────────────────────

class TestNumericSummary:

    def test_summary_contains_all_numeric_cols(self, profiler, simple_df):
        ctx = profiler.profile(simple_df)
        summary_cols = {r["column"] for r in ctx.numeric_summary}
        assert {"age", "salary", "churn"}.issubset(summary_cols)

    def test_mean_is_correct(self, profiler, simple_df):
        ctx = profiler.profile(simple_df)
        age_row = next(r for r in ctx.numeric_summary if r["column"] == "age")
        assert age_row["mean"] == pytest.approx(35.0, abs=0.01)

    def test_median_is_correct(self, profiler, simple_df):
        ctx = profiler.profile(simple_df)
        salary_row = next(r for r in ctx.numeric_summary if r["column"] == "salary")
        assert salary_row["median"] == pytest.approx(5000.0, abs=0.01)

    def test_outlier_count_nonnegative(self, profiler, simple_df):
        ctx = profiler.profile(simple_df)
        for row in ctx.numeric_summary:
            assert row["outlier_count"] >= 0


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Skewness Detection
# ─────────────────────────────────────────────────────────────────────────────

class TestSkewnessDetection:

    def test_heavily_skewed_column_flagged(self, profiler, df_skewed):
        ctx = profiler.profile(df_skewed)
        assert "skew_col" in ctx.skewed_columns

    def test_normal_column_not_flagged(self, profiler):
        df = pd.DataFrame({"normal": [10, 11, 12, 11, 10, 12, 11, 10]})
        ctx = profiler.profile(df)
        assert "normal" not in ctx.skewed_columns


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Target Detection
# ─────────────────────────────────────────────────────────────────────────────

class TestTargetDetection:

    def test_churn_detected_as_target(self, profiler, simple_df):
        ctx = profiler.profile(simple_df)
        assert ctx.target_column == "churn"

    @pytest.mark.parametrize("col_name", ["label", "target", "y", "fraud", "survived"])
    def test_various_target_hints(self, profiler, col_name):
        df = pd.DataFrame({col_name: [0, 1, 0, 1], "feature": [1.0, 2.0, 3.0, 4.0]})
        ctx = profiler.profile(df)
        assert ctx.target_column == col_name

    def test_no_target_when_absent(self, profiler):
        df = pd.DataFrame({"foo": [1, 2, 3], "bar": ["a", "b", "c"]})
        ctx = profiler.profile(df)
        assert ctx.target_column is None


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Risk Compilation
# ─────────────────────────────────────────────────────────────────────────────

class TestRiskCompilation:

    def test_risks_generated_for_missing(self, profiler, df_with_missing):
        ctx = profiler.profile(df_with_missing)
        assert len(ctx.risks) > 0

    def test_risk_message_contains_column_name(self, profiler, df_skewed):
        ctx = profiler.profile(df_skewed)
        skew_risks = [r for r in ctx.risks if "skew_col" in r]
        assert len(skew_risks) > 0

    def test_no_risks_clean_df(self, profiler):
        """Temiz bir DataFrame'in riski minimum olmalı."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        ctx = profiler.profile(df)
        # Saf sayılar, outlier/skew/missing yok
        skew_risks = [r for r in ctx.risks if "skewed" in r.lower()]
        assert len(skew_risks) == 0
