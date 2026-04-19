"""tools/profiler.py — EDA profiling: statistics, missingness, risks."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Thresholds (single source of truth) ─────────────────────────────────────
MISSING_WARN_THRESHOLD = 0.05      # ≥5 % → warn
MISSING_CRITICAL_THRESHOLD = 0.30  # ≥30 % → critical
HIGH_CARDINALITY_RATIO = 0.50      # unique/total ≥50 % → possible ID
SKEW_THRESHOLD = 2.0               # |skew| ≥2 → heavy skew
LOW_VARIANCE_THRESHOLD = 0.01      # std/mean < 1 % → near-constant


@dataclass
class DatasetContext:
    """Structured container for all profiling results."""

    # shape
    rows: int = 0
    columns: int = 0

    # column classification
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    datetime_columns: list[str] = field(default_factory=list)

    # analysis tables (stored as list-of-dicts for JSON serialisation)
    numeric_summary: list[dict[str, Any]] = field(default_factory=list)
    categorical_summary: list[dict[str, Any]] = field(default_factory=list)
    missing_report: list[dict[str, Any]] = field(default_factory=list)

    # detected issues
    possible_identifiers: list[str] = field(default_factory=list)
    high_missing_columns: list[str] = field(default_factory=list)
    skewed_columns: list[str] = field(default_factory=list)
    low_variance_columns: list[str] = field(default_factory=list)
    target_column: str | None = None

    # raw risks as human-readable strings
    risks: list[str] = field(default_factory=list)


class DataProfiler:
    """Produces a DatasetContext from a DataFrame."""

    TARGET_HINTS = {"churn", "target", "label", "y", "default", "fraud",
                    "cancelled", "converted", "survived", "outcome"}

    def profile(self, df: pd.DataFrame) -> DatasetContext:
        ctx = DatasetContext()

        self._classify_columns(df, ctx)
        self._basic_shape(df, ctx)
        self._missing_analysis(df, ctx)
        self._numeric_summary(df, ctx)
        self._categorical_summary(df, ctx)
        self._detect_skew(df, ctx)
        self._detect_identifiers(df, ctx)
        self._detect_low_variance(df, ctx)
        self._detect_target(df, ctx)
        self._compile_risks(ctx)

        return ctx

    # ── Column classification ────────────────────────────────────────────────

    def _classify_columns(self, df: pd.DataFrame, ctx: DatasetContext) -> None:
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                ctx.numeric_columns.append(col)
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                ctx.datetime_columns.append(col)
            else:
                ctx.categorical_columns.append(col)

    # ── Shape ────────────────────────────────────────────────────────────────

    def _basic_shape(self, df: pd.DataFrame, ctx: DatasetContext) -> None:
        ctx.rows, ctx.columns = df.shape

    # ── Missingness ──────────────────────────────────────────────────────────

    def _missing_analysis(self, df: pd.DataFrame, ctx: DatasetContext) -> None:
        total = len(df)
        for col in df.columns:
            n_miss = int(df[col].isna().sum())
            if n_miss == 0:
                continue
            rate = n_miss / total
            ctx.missing_report.append({
                "column": col,
                "missing_count": n_miss,
                "missing_rate": round(rate, 4),
                "severity": "critical" if rate >= MISSING_CRITICAL_THRESHOLD else "warn",
            })
            ctx.high_missing_columns.append(col)

    # ── Numeric summary ──────────────────────────────────────────────────────

    def _numeric_summary(self, df: pd.DataFrame, ctx: DatasetContext) -> None:
        for col in ctx.numeric_columns:
            s = df[col].dropna()
            if s.empty:
                continue
            ctx.numeric_summary.append({
                "column": col,
                "mean": round(float(s.mean()), 4),
                "median": round(float(s.median()), 4),
                "std": round(float(s.std()), 4),
                "min": round(float(s.min()), 4),
                "max": round(float(s.max()), 4),
                "skewness": round(float(s.skew()), 4),
                "q25": round(float(s.quantile(0.25)), 4),
                "q75": round(float(s.quantile(0.75)), 4),
                "outlier_count": int(self._iqr_outliers(s)),
            })

    @staticmethod
    def _iqr_outliers(s: pd.Series) -> int:
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        return int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())

    # ── Categorical summary ──────────────────────────────────────────────────

    def _categorical_summary(self, df: pd.DataFrame, ctx: DatasetContext) -> None:
        for col in ctx.categorical_columns:
            s = df[col].dropna()
            n_unique = int(s.nunique())
            top = s.value_counts().head(3).to_dict()
            ctx.categorical_summary.append({
                "column": col,
                "unique_values": n_unique,
                "top_values": {str(k): int(v) for k, v in top.items()},
                "cardinality_ratio": round(n_unique / len(df), 4),
            })

    # ── Skewness ─────────────────────────────────────────────────────────────

    def _detect_skew(self, df: pd.DataFrame, ctx: DatasetContext) -> None:
        for col in ctx.numeric_columns:
            s = df[col].dropna()
            if abs(s.skew()) >= SKEW_THRESHOLD:
                ctx.skewed_columns.append(col)

    # ── Identifier detection ──────────────────────────────────────────────────

    def _detect_identifiers(self, df: pd.DataFrame, ctx: DatasetContext) -> None:
        for col in df.columns:
            ratio = df[col].nunique() / len(df)
            if ratio < HIGH_CARDINALITY_RATIO:
                continue
            # Skip continuous float columns — they are not identifiers
            if pd.api.types.is_float_dtype(df[col]):
                continue
            ctx.possible_identifiers.append(col)

    # ── Low variance ─────────────────────────────────────────────────────────

    def _detect_low_variance(self, df: pd.DataFrame, ctx: DatasetContext) -> None:
        for col in ctx.numeric_columns:
            s = df[col].dropna()
            if s.mean() != 0 and s.std() / abs(s.mean()) < LOW_VARIANCE_THRESHOLD:
                ctx.low_variance_columns.append(col)

    # ── Target detection ─────────────────────────────────────────────────────

    def _detect_target(self, df: pd.DataFrame, ctx: DatasetContext) -> None:
        for col in df.columns:
            if col.lower() in self.TARGET_HINTS:
                ctx.target_column = col
                return

    # ── Risk compilation ─────────────────────────────────────────────────────

    def _compile_risks(self, ctx: DatasetContext) -> None:
        for col in ctx.high_missing_columns:
            rec = next(r for r in ctx.missing_report if r["column"] == col)
            pct = rec["missing_rate"] * 100
            ctx.risks.append(
                f"{col} has {'critical' if rec['severity'] == 'critical' else 'moderate'} "
                f"missingness ({pct:.1f}%)"
            )
        for col in ctx.possible_identifiers:
            ctx.risks.append(f"{col} may be an identifier column (high cardinality)")
        for col in ctx.skewed_columns:
            ctx.risks.append(f"{col} is heavily skewed — consider transformation")
        for col in ctx.low_variance_columns:
            ctx.risks.append(f"{col} has near-zero variance — likely useless for modeling")
