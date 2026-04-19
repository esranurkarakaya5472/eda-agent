"""agent.py — EDA Agent: rule-based decision engine with rich terminal output."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pandas as pd

from tools.loader import DatasetLoader
from tools.profiler import DataProfiler, DatasetContext
from tools.reporter import ReportWriter
from tools.llm import LLMAnalyst
from tools.cleaner import AutoCleaner
from tools.dashboard import DashboardGenerator

logger = logging.getLogger(__name__)

# ── ANSI colour helpers ───────────────────────────────────────────────────────
_W = "\033[0m"   # reset
_B = "\033[1m"   # bold
_C = "\033[96m"  # cyan
_G = "\033[92m"  # green
_Y = "\033[93m"  # yellow
_R = "\033[91m"  # red
_M = "\033[95m"  # magenta


def _sep(char: str = "─", width: int = 60) -> str:
    return char * width


def _header(title: str) -> None:
    print(f"\n{_B}{_C}{_sep()}{_W}")
    print(f"{_B}{_C}  {title}{_W}")
    print(f"{_B}{_C}{_sep()}{_W}")


def _ok(msg: str) -> None:
    print(f"  {_G}✔{_W}  {msg}")


def _warn(msg: str) -> None:
    print(f"  {_Y}⚠{_W}  {msg}")


def _risk(msg: str) -> None:
    print(f"  {_R}✘{_W}  {msg}")


def _info(msg: str) -> None:
    print(f"  {_M}→{_W}  {msg}")


# ── Agent ─────────────────────────────────────────────────────────────────────

class EDAAgent:
    """
    Autonomous EDA agent.

    Flow:
        load → profile → decide_actions → execute_actions → report
    """

    def __init__(
        self,
        reports_dir: str | Path = "reports",
        processed_dir: str | Path = "processed",
    ) -> None:
        self._loader = DatasetLoader()
        self._profiler = DataProfiler()
        self._reporter = ReportWriter(reports_dir)
        self._llm = LLMAnalyst()
        self._cleaner = AutoCleaner()
        self._dashboard = DashboardGenerator(reports_dir)
        self._processed_dir = Path(processed_dir)
        self._processed_dir.mkdir(parents=True, exist_ok=True)

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self, file_path: str | Path) -> dict:
        file_path = Path(file_path)
        print(f"\n{_B}{'═'*60}{_W}")
        print(f"{_B}  EDA AGENT  ·  {file_path.name}{_W}")
        print(f"{_B}{'═'*60}{_W}")

        # Step 1 — Load
        df = self._load(file_path)

        # Step 2 — Profile
        ctx = self._profile(df)

        # Step 3 — Decide
        action_plan = self._decide_actions(ctx)

        # Step 4 — Execute
        self._execute_actions(df, ctx, action_plan)

        # Step 4.5 — AI Interpretation
        llm_summary = self._run_llm(ctx)

        # Step 5 — Report
        paths = self._reporter.write(ctx, action_plan, str(file_path), llm_summary=llm_summary)
        self._print_report_paths(paths)

        # Step 6 — Auto-Cleaner
        self._run_cleaner(df, ctx, file_path)

        # Step 7 — HTML Dashboard
        dash_path = self._dashboard.generate(df, ctx, file_path, llm_summary, self._cleaner.log)
        if dash_path and dash_path.exists():
            _ok(f"Interactive Dashboard HTML hazırlandı: {dash_path}")

        # Step 8 — Move to processed
        self._move_to_processed(file_path)

        print(f"\n{_B}{_G}{'═'*60}{_W}")
        print(f"{_B}{_G}  AGENT RUN COMPLETE{_W}")
        print(f"{_B}{_G}{'═'*60}{_W}\n")

        return {"context": ctx, "action_plan": action_plan, "reports": paths}

    # ── Step 1: Load ──────────────────────────────────────────────────────────

    def _load(self, file_path: Path) -> pd.DataFrame:
        _header("STEP 1 — Loading Dataset")
        df = self._loader.load(file_path)
        _ok(f"Dataset loaded:  {file_path.name}")
        _ok(f"Rows:            {df.shape[0]:,}")
        _ok(f"Columns:         {df.shape[1]}")
        return df

    # ── Step 2: Profile ───────────────────────────────────────────────────────

    def _profile(self, df: pd.DataFrame) -> DatasetContext:
        _header("STEP 2 — Profiling Dataset")
        ctx = self._profiler.profile(df)

        _ok(f"Numeric columns:     {len(ctx.numeric_columns)}")
        _ok(f"Categorical columns: {len(ctx.categorical_columns)}")

        if ctx.datetime_columns:
            _ok(f"DateTime columns:    {len(ctx.datetime_columns)}")

        if ctx.target_column:
            _info(f"Target detected:     {_B}{ctx.target_column}{_W}")
        else:
            _warn("No target column detected")

        return ctx

    # ── Step 3: Decide ────────────────────────────────────────────────────────

    def _decide_actions(self, ctx: DatasetContext) -> list[str]:
        _header("STEP 3 — Agent Decision Plan")
        plan: list[str] = []

        # Always run
        plan.append("Run numeric summary analysis")
        plan.append("Run categorical summary analysis")

        if ctx.missing_report:
            plan.append("Run missing value analysis")

        if ctx.target_column:
            plan.append(f"Run target distribution analysis  [{ctx.target_column}]")
        else:
            plan.append("Skip target analysis  [no target detected]")

        if ctx.possible_identifiers:
            plan.append(f"Flag possible identifier columns  {ctx.possible_identifiers}")

        if ctx.skewed_columns:
            plan.append(f"Flag heavily skewed columns  {ctx.skewed_columns}")

        if ctx.low_variance_columns:
            plan.append(f"Flag low-variance columns  {ctx.low_variance_columns}")

        plan.append("Compile detected risks and generate report")

        for i, action in enumerate(plan, 1):
            tag = _G if "Skip" not in action else _Y
            print(f"  {tag}[{i}]{_W} {action}")

        return plan

    # ── Step 4: Execute ───────────────────────────────────────────────────────

    def _execute_actions(
        self, df: pd.DataFrame, ctx: DatasetContext, plan: list[str]
    ) -> None:
        _header("STEP 4 — Executing Analysis")

        # Numeric summary
        self._print_numeric_summary(ctx)

        # Categorical summary
        self._print_categorical_summary(ctx)

        # Missing value report
        if ctx.missing_report:
            self._print_missing_report(ctx)

        # Target distribution
        if ctx.target_column and ctx.target_column in df.columns:
            self._print_target_distribution(df, ctx.target_column)

        # Risks
        self._print_risks(ctx)

    # ── Step 4.5: AI Yorumlaması ──────────────────────────────────────────────

    def _run_llm(self, ctx: DatasetContext) -> str | None:
        _header("STEP 4.5 — AI Yorumlaması (LLM Analyst)")
        summary = self._llm.analyze(ctx)
        
        if summary:
            _ok("AI Yüksek Seviye Yönetici Özeti oluşturdu.")
            print(f"\n{_M}── AI ÖZETİ ────────────────────────────────────────────────────────{_W}\n")
            print(summary)
            print(f"\n{_M}────────────────────────────────────────────────────────────────────{_W}\n")
        else:
            _warn("AI çevrimdışı veya eksik yapılandırılmış (GEMINI_API_KEY eksik). Klasik raporlama kullanılıyor.")
            
        return summary

    # ── Step 6: Otomatik Temizleme (Auto-Cleaner) ─────────────────────────────

    def _run_cleaner(self, df: pd.DataFrame, ctx: DatasetContext, file_path: Path) -> None:
        _header("STEP 6 — Auto-Cleaning (Veri Temizliği)")
        df_clean, logs = self._cleaner.clean(df, ctx)
        
        for log in logs:
            if "Çöpe Atıldı" in log or "Tıraşlandı" in log or "Dolduruldu" in log:
                _info(log)
            else:
                _ok(log)

        # Temizlenmiş veriyi kaydet
        safe_name = f"cleaned_{file_path.name}"
        clean_dest = self._processed_dir / safe_name
        
        try:
            df_clean.to_csv(clean_dest, index=False)
            _ok(f"Temizlenmiş pırıl pırıl Veri Seti hazırlandı: {clean_dest}")
        except Exception as e:
            _risk(f"Temizlenmiş veriyi kaydederken hata oluştu: {e}")

    # ── Sub-printers ──────────────────────────────────────────────────────────

    @staticmethod
    def _print_numeric_summary(ctx: DatasetContext) -> None:
        print(f"\n  {_B}Numeric Summary{_W}")
        print(f"  {'Column':<22} {'Mean':>12} {'Median':>12} {'Std':>12} {'Skew':>8} {'Outliers':>9}")
        print(f"  {'-'*79}")
        for row in ctx.numeric_summary:
            print(
                f"  {row['column']:<22} "
                f"{row['mean']:>12,.2f} "
                f"{row['median']:>12,.2f} "
                f"{row['std']:>12,.2f} "
                f"{row['skewness']:>8.2f} "
                f"{row['outlier_count']:>9,}"
            )

    @staticmethod
    def _print_categorical_summary(ctx: DatasetContext) -> None:
        print(f"\n  {_B}Categorical Summary{_W}")
        print(f"  {'Column':<22} {'Unique':>8} {'Cardinality Ratio':>20}")
        print(f"  {'-'*54}")
        for row in ctx.categorical_summary:
            top_str = " | ".join(f"{k}({v})" for k, v in list(row["top_values"].items())[:2])
            print(
                f"  {row['column']:<22} "
                f"{row['unique_values']:>8,} "
                f"{row['cardinality_ratio']:>20.4f}"
                f"   {top_str}"
            )

    @staticmethod
    def _print_missing_report(ctx: DatasetContext) -> None:
        print(f"\n  {_B}Missing Value Report{_W}")
        print(f"  {'Column':<22} {'Missing %':>12} {'Severity':>12}")
        print(f"  {'-'*50}")
        for rec in sorted(ctx.missing_report, key=lambda r: -r["missing_rate"]):
            colour = _R if rec["severity"] == "critical" else _Y
            pct = f"{rec['missing_rate']*100:.1f}%"
            print(f"  {colour}{rec['column']:<22} {pct:>12} {rec['severity']:>12}{_W}")

    @staticmethod
    def _print_target_distribution(df: pd.DataFrame, target: str) -> None:
        print(f"\n  {_B}Target Distribution  [{target}]{_W}")
        vc = df[target].value_counts(normalize=True)
        for label, ratio in vc.items():
            bar = "█" * int(ratio * 40)
            print(f"  {str(label):<10} {bar:<40}  {ratio*100:.1f}%")

    @staticmethod
    def _print_risks(ctx: DatasetContext) -> None:
        print(f"\n  {_B}Detected Data Risks{_W}")
        if not ctx.risks:
            _ok("No significant risks detected.")
            return
        for risk in ctx.risks:
            _risk(risk)

    @staticmethod
    def _print_report_paths(paths: dict) -> None:
        _header("STEP 5 — Reports Written")
        for kind, path in paths.items():
            _ok(f"{kind.upper():<10} → {path}")

    def _move_to_processed(self, file_path: Path) -> None:
        dest = self._processed_dir / file_path.name
        # Only move if file is inside incoming/; leave data/ intact
        if "incoming" in str(file_path):
            shutil.move(str(file_path), dest)
            _info(f"Moved to processed: {dest}")
