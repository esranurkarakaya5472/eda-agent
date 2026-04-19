"""tests/test_integration.py — End-to-end integration tests for EDA Agent pipeline."""
import pytest
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.loader import DatasetLoader
from tools.profiler import DataProfiler
from tools.cleaner import AutoCleaner


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def realistic_csv(tmp_path) -> Path:
    """Gerçekçi bir müşteri veri seti CSV'i oluşturur."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "customer_id":  [f"C{i:04d}" for i in range(n)],
        "age":          np.random.randint(18, 70, n).astype(float),
        "salary":       np.random.exponential(scale=5000, size=n),   # sağa çarpık
        "tenure_months": np.random.randint(1, 120, n).astype(float),
        "city":         np.random.choice(["Istanbul", "Ankara", "Izmir", "Bursa"], n),
        "churn":        np.random.choice([0, 1], n, p=[0.7, 0.3]),
    })
    # %15 oranında rastgele eksik değer ekle
    for col in ["age", "salary", "city"]:
        mask = np.random.choice([True, False], n, p=[0.15, 0.85])
        df.loc[mask, col] = np.nan
    # Aşırı değer ekle
    df.loc[0, "salary"] = 9_999_999.0

    path = tmp_path / "customers.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def pipeline_result(realistic_csv):
    """Load → Profile → Clean pipeline'ını tek seferde çalıştırır."""
    loader = DatasetLoader()
    profiler = DataProfiler()
    cleaner = AutoCleaner()

    df = loader.load(realistic_csv)
    ctx = profiler.profile(df)
    df_clean, logs = cleaner.clean(df, ctx)
    return df, ctx, df_clean, logs


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Load → Profile Integration
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadProfile:

    def test_profile_has_correct_shape(self, pipeline_result):
        df, ctx, _, _ = pipeline_result
        assert ctx.rows == df.shape[0]
        assert ctx.columns == df.shape[1]

    def test_churn_detected_as_target(self, pipeline_result):
        _, ctx, _, _ = pipeline_result
        assert ctx.target_column == "churn"

    def test_customer_id_flagged_as_identifier(self, pipeline_result):
        _, ctx, _, _ = pipeline_result
        assert "customer_id" in ctx.possible_identifiers

    def test_missing_values_detected(self, pipeline_result):
        _, ctx, _, _ = pipeline_result
        missing_cols = {r["column"] for r in ctx.missing_report}
        # En az age veya salary eksik olmalı (fixture'da %15 eklendi)
        assert len(missing_cols) > 0

    def test_salary_detected_as_skewed(self, pipeline_result):
        """Exponential dağılımlı salary çarpık algılanmalı."""
        _, ctx, _, _ = pipeline_result
        assert "salary" in ctx.skewed_columns

    def test_numeric_summary_populated(self, pipeline_result):
        _, ctx, _, _ = pipeline_result
        numeric_cols_in_summary = {r["column"] for r in ctx.numeric_summary}
        assert "age" in numeric_cols_in_summary
        assert "salary" in numeric_cols_in_summary

    def test_risks_generated(self, pipeline_result):
        """Eksik değerler ve outlier'lar varsa risk listesi dolu olmalı."""
        _, ctx, _, _ = pipeline_result
        assert len(ctx.risks) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Profile → Clean Integration
# ─────────────────────────────────────────────────────────────────────────────

class TestProfileClean:

    def test_cleaned_df_has_no_nulls_in_numeric(self, pipeline_result):
        _, ctx, df_clean, _ = pipeline_result
        for col in ctx.numeric_columns:
            if col in df_clean.columns:
                assert df_clean[col].isna().sum() == 0, f"{col} hâlâ NaN içeriyor!"

    def test_cleaned_df_has_no_nulls_in_categorical(self, pipeline_result):
        _, ctx, df_clean, _ = pipeline_result
        for col in ctx.categorical_columns:
            if col in df_clean.columns:
                assert df_clean[col].isna().sum() == 0, f"{col} hâlâ NaN içeriyor!"

    def test_salary_outlier_capped(self, pipeline_result):
        """Aşırı salary değeri (9.999.999) kırpılmış olmalı."""
        _, _, df_clean, _ = pipeline_result
        assert df_clean["salary"].max() < 9_999_999.0

    def test_same_row_count_after_cleaning(self, pipeline_result):
        """Cleaner satır silmemeli, yalnızca değerleri düzenlemeli."""
        df, _, df_clean, _ = pipeline_result
        assert len(df_clean) == len(df)

    def test_cleaning_logs_not_empty(self, pipeline_result):
        """Gerçekçi veriden sonra log boş olmamalı."""
        _, _, _, logs = pipeline_result
        assert len(logs) > 0

    def test_churn_values_preserved(self, pipeline_result):
        """Target sütunu değiştirilmemeli."""
        df, _, df_clean, _ = pipeline_result
        # Değerlerin toplamı aynı olmalı (capping uygulanmamalı)
        assert df["churn"].sum() == df_clean["churn"].sum()


# ─────────────────────────────────────────────────────────────────────────────
# Tests: DatasetContext Consistency
# ─────────────────────────────────────────────────────────────────────────────

class TestContextConsistency:

    def test_total_columns_match_col_lists(self, pipeline_result):
        """numeric + categorical + datetime sütun sayısı toplam sütun sayısına eşit olmalı."""
        _, ctx, _, _ = pipeline_result
        total = len(ctx.numeric_columns) + len(ctx.categorical_columns) + len(ctx.datetime_columns)
        assert total == ctx.columns

    def test_missing_report_columns_exist_in_df(self, pipeline_result):
        df, ctx, _, _ = pipeline_result
        for rec in ctx.missing_report:
            assert rec["column"] in df.columns

    def test_skewed_columns_subset_of_numeric(self, pipeline_result):
        _, ctx, _, _ = pipeline_result
        for col in ctx.skewed_columns:
            assert col in ctx.numeric_columns

    def test_low_variance_subset_of_numeric(self, pipeline_result):
        _, ctx, _, _ = pipeline_result
        for col in ctx.low_variance_columns:
            assert col in ctx.numeric_columns
