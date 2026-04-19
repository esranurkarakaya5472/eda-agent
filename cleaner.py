"""tools/cleaner.py — Otomatik Veri Temizleyici."""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import numpy as np

from tools.profiler import DatasetContext

logger = logging.getLogger(__name__)


class AutoCleaner:
    """Intelligently cleans the dataframe based on the profiling context."""

    def __init__(self) -> None:
        self.log: list[str] = []

    def clean(self, df: pd.DataFrame, ctx: DatasetContext) -> tuple[pd.DataFrame, list[str]]:
        """Run all cleaning steps and return the sanitized DataFrame and an action log."""
        self.log.clear()
        df_clean = df.copy()

        # Step 1: Drop purely useless columns (low variance)
        self._drop_low_variance(df_clean, ctx)

        # Step 2: Handle Missing Data
        self._handle_missing(df_clean, ctx)

        # Step 3: Handle Outliers (Capping/Winsorization)
        self._cap_outliers(df_clean, ctx)

        if not self.log:
            self.log.append("Veri zaten harika görünüyordu, müdahaleye gerek kalmadı.")

        return df_clean, self.log.copy()

    def _drop_low_variance(self, df: pd.DataFrame, ctx: DatasetContext) -> None:
        if not ctx.low_variance_columns:
            return
        
        cols_to_drop = [c for c in ctx.low_variance_columns if c in df.columns]
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)
            self.log.append(f"Sabit/Sıfır Varyanslı Sütunlar Çöpe Atıldı: {', '.join(cols_to_drop)}")

    def _handle_missing(self, df: pd.DataFrame, ctx: DatasetContext) -> None:
        if not ctx.missing_report:
            return

        for rec in ctx.missing_report:
            col = rec["column"]
            if col not in df.columns:
                continue

            # If Numeric
            if col in ctx.numeric_columns:
                # If skewed, use median. Else, use mean.
                is_skewed = col in ctx.skewed_columns
                if is_skewed:
                    val_to_fill = df[col].median()
                    strategy = "Medyan"
                else:
                    val_to_fill = df[col].mean()
                    strategy = "Ortalama"

                df[col] = df[col].fillna(val_to_fill)
                self.log.append(f"Sayısal Boşluk Dolduruldu: [{col}] -> {strategy} ile dolduruldu ({val_to_fill:.2f})")

            # If Categorical
            elif col in ctx.categorical_columns:
                val_to_fill = "Bilinmiyor"
                df[col] = df[col].fillna(val_to_fill)
                self.log.append(f"Kategorik Boşluk Dolduruldu: [{col}] -> '{val_to_fill}' etiketi atandı.")

    def _cap_outliers(self, df: pd.DataFrame, ctx: DatasetContext) -> None:
        for col in ctx.numeric_columns:
            if col not in df.columns or col == ctx.target_column:  # Don't cap target
                continue

            # Standard IQR Capping
            s = df[col].dropna()
            if s.empty:
                continue

            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers_count > 0:
                df[col] = np.clip(df[col], lower_bound, upper_bound)
                self.log.append(f"Uç Değer Tıraşlandı (Capping): [{col}] -> {outliers_count} ekstrem satır IQR sınırlarına çekildi.")
