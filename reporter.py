"""tools/reporter.py — Generates Markdown and JSON reports from DatasetContext."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from tools.profiler import DatasetContext

logger = logging.getLogger(__name__)


class ReportWriter:
    """Writes analysis.json and report.md into the reports directory."""

    def __init__(self, reports_dir: str | Path = "reports") -> None:
        self._dir = Path(reports_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def write(
        self,
        ctx: DatasetContext,
        action_plan: list[str],
        dataset_name: str = "dataset",
        llm_summary: str | None = None,
    ) -> dict[str, Path]:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"{Path(dataset_name).stem}_{ts}"

        json_path = self._write_json(ctx, action_plan, stem, llm_summary)
        md_path = self._write_markdown(ctx, action_plan, stem, dataset_name, llm_summary)

        logger.info("Reports written → %s, %s", json_path.name, md_path.name)
        return {"json": json_path, "markdown": md_path}

    # ── JSON ─────────────────────────────────────────────────────────────────

    def _write_json(
        self, ctx: DatasetContext, action_plan: list[str], stem: str, llm_summary: str | None
    ) -> Path:
        payload = {
            "generated_at": datetime.now().isoformat(),
            "shape": {"rows": ctx.rows, "columns": ctx.columns},
            "column_types": {
                "numeric": ctx.numeric_columns,
                "categorical": ctx.categorical_columns,
                "datetime": ctx.datetime_columns,
            },
            "action_plan": action_plan,
            "missing_report": ctx.missing_report,
            "numeric_summary": ctx.numeric_summary,
            "categorical_summary": ctx.categorical_summary,
            "detected_risks": ctx.risks,
            "possible_identifiers": ctx.possible_identifiers,
            "skewed_columns": ctx.skewed_columns,
            "target_column": ctx.target_column,
            "llm_summary": llm_summary,
        }
        path = self._dir / f"{stem}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        return path

    # ── Markdown ─────────────────────────────────────────────────────────────

    def _write_markdown(
        self,
        ctx: DatasetContext,
        action_plan: list[str],
        stem: str,
        dataset_name: str,
        llm_summary: str | None,
    ) -> Path:
        lines: list[str] = []

        # Header
        lines += [
            "# Automated EDA Report",
            f"**Dataset:** `{dataset_name}`  ",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]

        # Dataset overview
        lines += [
            "## Dataset Overview",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Rows | {ctx.rows:,} |",
            f"| Columns | {ctx.columns} |",
            f"| Numeric columns | {len(ctx.numeric_columns)} |",
            f"| Categorical columns | {len(ctx.categorical_columns)} |",
            f"| DateTime columns | {len(ctx.datetime_columns)} |",
            f"| Target column | `{ctx.target_column}` |" if ctx.target_column else "| Target column | *not detected* |",
            "",
        ]

        # Action plan
        lines += ["## Agent Action Plan", ""]
        for i, action in enumerate(action_plan, 1):
            lines.append(f"{i}. {action}")
        lines.append("")

        # Missing value report
        if ctx.missing_report:
            lines += ["## Missing Value Report", ""]
            lines += ["| Column | Missing % | Severity |", "|--------|-----------|----------|"]
            for rec in sorted(ctx.missing_report, key=lambda r: -r["missing_rate"]):
                lines.append(
                    f"| {rec['column']} | {rec['missing_rate']*100:.1f}% | {rec['severity']} |"
                )
            lines.append("")

        # Key risks
        if ctx.risks:
            lines += ["## Key Risks", ""]
            for risk in ctx.risks:
                lines.append(f"- {risk}")
            lines.append("")

        # AI Executive Summary or Rule-Based Fallback
        if llm_summary:
            lines += ["## 🧠 AI Yönetici Özeti (Executive Summary)", ""]
            lines.append(llm_summary)
            lines.append("")
        else:
            lines += ["## Executive Summary", ""]
            lines.append(self._executive_summary(ctx))
            lines.append("")

            lines += ["## Recommended Next Steps", ""]
            for step in self._next_steps(ctx):
                lines.append(f"- {step}")
            lines.append("")

        path = self._dir / f"{stem}.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    # ── Rule-based text generation ────────────────────────────────────────────

    @staticmethod
    def _executive_summary(ctx: DatasetContext) -> str:
        parts: list[str] = [
            f"This dataset contains **{ctx.rows:,} rows** and **{ctx.columns} columns** "
            f"({len(ctx.numeric_columns)} numeric, {len(ctx.categorical_columns)} categorical)."
        ]
        if ctx.high_missing_columns:
            cols = ", ".join(f"`{c}`" for c in ctx.high_missing_columns[:3])
            parts.append(f"Missingness is present in {cols} and requires attention before modeling.")
        if ctx.possible_identifiers:
            ids = ", ".join(f"`{c}`" for c in ctx.possible_identifiers)
            parts.append(f"{ids} appear to be non-informative identifier columns and should be excluded.")
        if ctx.skewed_columns:
            sk = ", ".join(f"`{c}`" for c in ctx.skewed_columns[:3])
            parts.append(f"Heavy right skew detected in {sk}; consider log or power transformations.")
        if ctx.target_column:
            parts.append(
                f"Target column `{ctx.target_column}` detected — apply target-aware preprocessing."
            )
        return " ".join(parts)

    @staticmethod
    def _next_steps(ctx: DatasetContext) -> list[str]:
        steps: list[str] = []
        if ctx.high_missing_columns:
            steps.append("Investigate and impute missing values in flagged columns.")
        if ctx.possible_identifiers:
            steps.append("Exclude high-cardinality identifier columns from feature set.")
        if ctx.skewed_columns:
            steps.append("Apply log1p or Box-Cox transformation to skewed numeric features.")
        if ctx.low_variance_columns:
            steps.append("Review and potentially drop near-constant columns.")
        if ctx.target_column:
            steps.append(f"Analyse class balance in `{ctx.target_column}` before splitting.")
        if not steps:
            steps.append("Dataset looks clean — proceed to feature engineering.")
        return steps
