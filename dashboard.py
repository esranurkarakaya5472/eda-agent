"""tools/dashboard.py — Smart, context-aware interactive HTML dashboard."""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd
from tools.profiler import DatasetContext

logger = logging.getLogger(__name__)

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from jinja2 import Template
    HAS_DASHBOARD_DEPS = True
except ImportError:
    HAS_DASHBOARD_DEPS = False


# ── Markdown → HTML ───────────────────────────────────────────────────────────

def _md_to_html(text: str) -> str:
    if not text:
        return ""
    lines = text.split("\n")
    html_lines: list[str] = []
    in_ul = False
    for line in lines:
        if line.startswith("### "):
            if in_ul: html_lines.append("</ul>"); in_ul = False
            html_lines.append(f"<h3 class='md-h3'>{line[4:].strip()}</h3>")
        elif line.startswith("## "):
            if in_ul: html_lines.append("</ul>"); in_ul = False
            html_lines.append(f"<h2 class='md-h2'>{line[3:].strip()}</h2>")
        elif line.startswith("# "):
            if in_ul: html_lines.append("</ul>"); in_ul = False
            html_lines.append(f"<h1 class='md-h1'>{line[2:].strip()}</h1>")
        elif line.startswith("* ") or line.startswith("- "):
            if not in_ul:
                html_lines.append("<ul class='md-ul'>"); in_ul = True
            html_lines.append(f"  <li>{_inline_md(line[2:].strip())}</li>")
        elif re.match(r"^\d+\. ", line):
            if in_ul: html_lines.append("</ul>"); in_ul = False
            html_lines.append(f"<p class='md-ol'>{_inline_md(re.sub(r'^\\d+\\.\\s', '', line).strip())}</p>")
        elif line.strip() in ("---", "***", "___"):
            if in_ul: html_lines.append("</ul>"); in_ul = False
            html_lines.append("<hr class='md-hr'>")
        elif line.strip() == "":
            if in_ul: html_lines.append("</ul>"); in_ul = False
            html_lines.append("<br>")
        else:
            if in_ul: html_lines.append("</ul>"); in_ul = False
            html_lines.append(f"<p class='md-p'>{_inline_md(line)}</p>")
    if in_ul:
        html_lines.append("</ul>")
    return "\n".join(html_lines)


def _inline_md(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.+?)\*",     r"<em>\1</em>",         text)
    text = re.sub(r"`(.+?)`",       r"<code class='md-code'>\1</code>", text)
    return text


# ── HTML Template ─────────────────────────────────────────────────────────────

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EDA Agent — {{ filename }}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        * { box-sizing: border-box; }
        body {
            background: #080c14;
            color: #94a3b8;
            font-family: 'Inter', system-ui, sans-serif;
            margin: 0; padding: 0;
        }

        /* GLASSMORPHISM KART */
        .glass {
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 18px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            backdrop-filter: blur(12px);
        }
        .glass-light {
            background: rgba(30, 41, 59, 0.5);
            border: 1px solid rgba(255,255,255,0.05);
            border-radius: 14px;
        }

        /* RENK PALETİ */
        .purple { color: #c084fc; }
        .blue   { color: #60a5fa; }
        .green  { color: #34d399; }
        .yellow { color: #fbbf24; }
        .red    { color: #f87171; }
        .glow   { text-shadow: 0 0 30px rgba(192,132,252,0.5); }

        /* HEROo HEADER */
        .hero-gradient {
            background: linear-gradient(135deg, #c084fc 0%, #60a5fa 50%, #34d399 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        /* STAT KARTLARI */
        .stat-card {
            background: rgba(124,58,237,0.08);
            border: 1px solid rgba(124,58,237,0.2);
            border-radius: 14px;
            padding: 18px 20px;
            text-align: center;
            transition: transform 0.2s, border-color 0.2s;
        }
        .stat-card:hover { transform: translateY(-3px); border-color: rgba(124,58,237,0.5); }
        .stat-val { font-size: 2rem; font-weight: 800; color: #c084fc; line-height: 1; }
        .stat-lbl { font-size: 0.68rem; color: #475569; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 5px; }

        /* BÖLÜM BAŞLIĞI */
        .section-title {
            font-size: 1.15rem;
            font-weight: 700;
            color: #e2e8f0;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .section-title::after {
            content: '';
            flex: 1;
            height: 1px;
            background: rgba(255,255,255,0.06);
            margin-left: 8px;
        }

        /* BADGE */
        .badge {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 6px;
            font-size: 0.75rem;
            margin: 3px;
        }
        .badge-red    { background:rgba(239,68,68,0.12); border:1px solid rgba(239,68,68,0.3); color:#fca5a5; }
        .badge-yellow { background:rgba(251,191,36,0.12); border:1px solid rgba(251,191,36,0.3); color:#fde68a; }
        .badge-green  { background:rgba(52,211,153,0.12); border:1px solid rgba(52,211,153,0.3); color:#6ee7b7; }
        .badge-blue   { background:rgba(96,165,250,0.12); border:1px solid rgba(96,165,250,0.3); color:#93c5fd; }

        /* SCORE RENGI */
        .score-excellent { color: #34d399; }
        .score-good      { color: #60a5fa; }
        .score-warning   { color: #fbbf24; }
        .score-poor      { color: #f87171; }

        /* MARKDOWN */
        .md-h1 { font-size:1.35rem; font-weight:800; color:#c084fc; margin:1rem 0 0.4rem; }
        .md-h2 { font-size:1.1rem;  font-weight:700; color:#60a5fa; margin:0.9rem 0 0.3rem; border-bottom:1px solid rgba(255,255,255,0.06); padding-bottom:4px; }
        .md-h3 { font-size:0.95rem; font-weight:600; color:#34d399; margin:0.7rem 0 0.25rem; }
        .md-p  { margin:0.3rem 0; line-height:1.75; color:#cbd5e1; font-size:0.9rem; }
        .md-ul { margin:0.3rem 0 0.5rem 1.4rem; padding:0; list-style:disc; }
        .md-ul li { margin:0.2rem 0; color:#cbd5e1; line-height:1.65; font-size:0.9rem; }
        .md-code { background:rgba(124,58,237,0.18); color:#f0abfc; padding:1px 5px; border-radius:4px; font-size:0.85em; font-family:monospace; }
        .md-hr { border:none; border-top:1px solid rgba(255,255,255,0.07); margin:0.6rem 0; }
        strong { color:#e2e8f0; }

        /* FOOTER */
        footer { border-top: 1px solid rgba(255,255,255,0.05); }

        /* GRID HELPER */
        .chart-grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .chart-grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; }
        @media (max-width: 900px) {
            .chart-grid-2, .chart-grid-3 { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body class="p-4 md:p-8 lg:p-12">
<div class="max-w-screen-xl mx-auto space-y-8">

    <!-- ── HERO ───────────────────────────────────────────── -->
    <header class="text-center pt-4 pb-2">
        <p class="text-xs text-slate-600 uppercase tracking-widest mb-2">EDA Agent · Autonomous Data Intelligence</p>
        <h1 class="text-4xl md:text-6xl font-extrabold hero-gradient mb-3">{{ filename }}</h1>
        <p class="text-slate-500 text-sm">{{ ctx.rows | int }} satır &nbsp;·&nbsp; {{ ctx.columns }} sütun &nbsp;·&nbsp;
            {% if ctx.target_column %}
                <span class="badge badge-green">🎯 Hedef: {{ ctx.target_column }}</span>
            {% else %}
                <span class="badge badge-yellow">Hedef kolon tespit edilmedi</span>
            {% endif %}
        </p>
    </header>

    <!-- ── STAT KARTLARI ──────────────────────────────────── -->
    <div class="chart-grid-3" style="grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));">
        <div class="stat-card">
            <div class="stat-val">{{ ctx.rows | int }}</div>
            <div class="stat-lbl">Satır</div>
        </div>
        <div class="stat-card" style="background:rgba(96,165,250,0.08); border-color:rgba(96,165,250,0.2);">
            <div class="stat-val blue">{{ ctx.columns }}</div>
            <div class="stat-lbl">Sütun</div>
        </div>
        <div class="stat-card" style="background:rgba(52,211,153,0.08); border-color:rgba(52,211,153,0.2);">
            <div class="stat-val green">{{ ctx.numeric_columns | length }}</div>
            <div class="stat-lbl">Sayısal</div>
        </div>
        <div class="stat-card" style="background:rgba(251,191,36,0.08); border-color:rgba(251,191,36,0.2);">
            <div class="stat-val yellow">{{ ctx.categorical_columns | length }}</div>
            <div class="stat-lbl">Kategorik</div>
        </div>
        <div class="stat-card" style="background:rgba(239,68,68,0.08); border-color:rgba(239,68,68,0.2);">
            <div class="stat-val red">{{ ctx.risks | length }}</div>
            <div class="stat-lbl">Risk</div>
        </div>
        <div class="stat-card" style="background:rgba({{ score_color_rgb }},0.08); border-color:rgba({{ score_color_rgb }},0.25);">
            <div class="stat-val" style="color:{{ score_color }};">{{ quality_score }}</div>
            <div class="stat-lbl">Kalite Skoru</div>
        </div>
    </div>

    <!-- ── AI ÖZETİ ───────────────────────────────────────── -->
    {% if llm_summary_html %}
    <div class="glass p-6 md:p-8">
        <div class="section-title"><span>🧠</span> AI Yönetici Özeti</div>
        <div class="p-5 glass-light text-sm leading-relaxed">
            {{ llm_summary_html | safe }}
        </div>
    </div>
    {% endif %}

    <!-- ── AUTO-CLEANER LOGLARI ───────────────────────────── -->
    {% if logs %}
    <div class="glass p-6">
        <div class="section-title"><span>🛠️</span> Auto-Cleaner Operasyonları</div>
        <div class="space-y-2">
            {% for log in logs %}
            <div class="flex items-start gap-2 text-sm">
                <span class="green mt-0.5 flex-shrink-0">✓</span>
                <span class="text-slate-300">{{ log }}</span>
            </div>
            {% endfor %}
        </div>
    </div>
    {% endif %}

    <!-- ── RİSKLER ────────────────────────────────────────── -->
    {% if ctx.risks %}
    <div class="glass p-5 md:p-6">
        <div class="section-title"><span>⚠️</span> Tespit Edilen Riskler</div>
        <div>
            {% for risk in ctx.risks %}
            <span class="badge {{ 'badge-red' if 'critical' in risk or 'heavily' in risk else 'badge-yellow' }}">{{ risk }}</span>
            {% endfor %}
        </div>
    </div>
    {% endif %}

    <!-- ── HEDEF & KALİTE: Donut + Gauge ─────────────────── -->
    <div class="chart-grid-2">
        {% if plot_gauge %}
        <div class="glass p-3 overflow-hidden">{{ plot_gauge | safe }}</div>
        {% endif %}
        {% if plot_target %}
        <div class="glass p-3 overflow-hidden">{{ plot_target | safe }}</div>
        {% elif plot_gauge %}
        <div class="glass p-6 flex items-center justify-center">
            <p class="text-slate-600 text-sm">Hedef kolon tespit edilmedi.</p>
        </div>
        {% endif %}
    </div>

    <!-- ── FEATURE-TARGET KORELASYON ─────────────────────── -->
    {% if plot_feature_corr %}
    <div class="glass p-3 overflow-hidden">
        {{ plot_feature_corr | safe }}
    </div>
    {% endif %}

    <!-- ── VİOLIN DAĞILIMLAR + SKEWNESS ──────────────────── -->
    <div class="chart-grid-2">
        <div class="glass p-3 overflow-hidden">{{ plot_violin | safe }}</div>
        <div class="glass p-3 overflow-hidden">{{ plot_skewness | safe }}</div>
    </div>

    <!-- ── OUTLİER + KATEGORİK ───────────────────────────── -->
    <div class="chart-grid-2">
        <div class="glass p-3 overflow-hidden">{{ plot_outliers | safe }}</div>
        <div class="glass p-3 overflow-hidden">{{ plot_categorical | safe }}</div>
    </div>

    <!-- ── EKSİK VERİ ─────────────────────────────────────── -->
    {% if plot_missing %}
    <div class="glass p-3 overflow-hidden">{{ plot_missing | safe }}</div>
    {% endif %}

    <!-- ── KORELASYON ISISI ───────────────────────────────── -->
    {% if plot_corr %}
    <div class="glass p-3 overflow-hidden">{{ plot_corr | safe }}</div>
    {% endif %}

    <footer class="text-center text-xs text-slate-700 py-6 mt-4">
        Generated by <span class="text-purple-600 font-semibold">EDA Agent</span> &nbsp;·&nbsp; Powered by Gemini AI
    </footer>
</div>
</body>
</html>
"""


class DashboardGenerator:
    """Smart, context-aware HTML dashboard generator."""

    _DARK = dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", color="#94a3b8"),
        margin=dict(l=24, r=24, t=52, b=24),
    )

    def __init__(self, reports_dir: str | Path = "reports") -> None:
        self._dir = Path(reports_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        df: pd.DataFrame,
        ctx: DatasetContext,
        file_path: Path,
        llm_summary: str | None = None,
        cleaning_logs: list[str] | None = None,
    ) -> Path:
        if not HAS_DASHBOARD_DEPS:
            logger.error("Dashboard generation skipped: plotly/jinja2 not installed.")
            return Path()

        # ── Kalite skoru hesapla ───────────────────────────────────────────────
        score = self._calc_quality_score(ctx)
        if score >= 80:
            sc, sc_rgb = "#34d399", "52,211,153"
        elif score >= 60:
            sc, sc_rgb = "#60a5fa", "96,165,250"
        elif score >= 40:
            sc, sc_rgb = "#fbbf24", "251,191,36"
        else:
            sc, sc_rgb = "#f87171", "239,68,68"

        # ── Grafikleri oluştur ────────────────────────────────────────────────
        figs = {
            "gauge":        self._chart_quality_gauge(score, sc),
            "target":       self._chart_target_donut(df, ctx),
            "feature_corr": self._chart_feature_target_corr(df, ctx),
            "violin":       self._chart_violin(df, ctx),
            "skewness":     self._chart_skewness(ctx),
            "outliers":     self._chart_outliers(ctx),
            "categorical":  self._chart_categorical(df, ctx),
            "missing":      self._chart_missing(ctx),
            "corr":         self._chart_correlation(df, ctx),
        }

        def _html(fig) -> str | None:
            return fig.to_html(full_html=False, include_plotlyjs=False) if fig else None

        template = Template(HTML_TEMPLATE)
        final_html = template.render(
            filename=file_path.name,
            ctx=ctx,
            llm_summary_html=_md_to_html(llm_summary) if llm_summary else "",
            logs=cleaning_logs or [],
            quality_score=score,
            score_color=sc,
            score_color_rgb=sc_rgb,
            **{f"plot_{k}": _html(v) for k, v in figs.items()},
        )

        dest = self._dir / f"dashboard_{file_path.stem}.html"
        dest.write_text(final_html, encoding="utf-8")
        logger.info("Dashboard oluşturuldu: %s", dest)
        return dest

    # ── Kalite Skoru ─────────────────────────────────────────────────────────

    @staticmethod
    def _calc_quality_score(ctx: DatasetContext) -> int:
        score = 100
        for rec in ctx.missing_report:
            score -= 12 if rec["severity"] == "critical" else 4
        score -= len(ctx.skewed_columns) * 3
        score -= len(ctx.possible_identifiers) * 2
        score -= len(ctx.low_variance_columns) * 5
        return max(0, min(100, score))

    # ── 1. Kalite Skoru Gauge ─────────────────────────────────────────────────

    def _chart_quality_gauge(self, score: int, color: str) -> go.Figure:
        label = "Mükemmel" if score >= 80 else "İyi" if score >= 60 else "Orta" if score >= 40 else "Zayıf"
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score,
            title=dict(text=f"Veri Kalitesi Skoru<br><span style='font-size:0.85em;color:#64748b;'>{label}</span>", font=dict(size=15)),
            delta=dict(reference=80, decreasing=dict(color="#f87171")),
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor="#475569", tickfont=dict(color="#475569")),
                bar=dict(color=color, thickness=0.28),
                bgcolor="rgba(0,0,0,0)",
                steps=[
                    dict(range=[0,  40], color="rgba(239,68,68,0.08)"),
                    dict(range=[40, 60], color="rgba(251,191,36,0.08)"),
                    dict(range=[60, 80], color="rgba(96,165,250,0.08)"),
                    dict(range=[80,100], color="rgba(52,211,153,0.08)"),
                ],
                threshold=dict(line=dict(color=color, width=3), thickness=0.8, value=score),
            ),
            number=dict(font=dict(size=52, color=color), suffix="/100"),
        ))
        fig.update_layout(**self._DARK, height=320,
                          title=dict(text="📊 Veri Kalitesi Göstergesi", font=dict(size=14)))
        return fig

    # ── 2. Hedef Dağılımı Donut ──────────────────────────────────────────────

    def _chart_target_donut(self, df: pd.DataFrame, ctx: DatasetContext) -> go.Figure | None:
        if not ctx.target_column or ctx.target_column not in df.columns:
            return None

        vc = df[ctx.target_column].value_counts()
        labels = [str(l) for l in vc.index]
        values = vc.values.tolist()
        colors = ["#c084fc", "#60a5fa", "#34d399", "#fbbf24", "#f87171"]

        fig = go.Figure(go.Pie(
            labels=labels,
            values=values,
            hole=0.6,
            marker=dict(colors=colors[:len(labels)], line=dict(color="#080c14", width=3)),
            textinfo="label+percent",
            textfont=dict(size=13),
            hovertemplate="<b>%{label}</b><br>Sayı: %{value}<br>Oran: %{percent}<extra></extra>",
        ))
        fig.add_annotation(
            text=f"<b>{ctx.target_column}</b><br>Dağılımı",
            x=0.5, y=0.5, font=dict(size=13, color="#e2e8f0"),
            showarrow=False, align="center",
        )
        fig.update_layout(**self._DARK, height=320,
                          title=dict(text=f"🎯 Hedef Kolon: {ctx.target_column}", font=dict(size=14)),
                          showlegend=True,
                          legend=dict(orientation="v", x=1.02, y=0.5))
        return fig

    # ── 3. Feature → Target Korelasyon ──────────────────────────────────────

    def _chart_feature_target_corr(self, df: pd.DataFrame, ctx: DatasetContext) -> go.Figure | None:
        if not ctx.target_column or ctx.target_column not in df.columns:
            return None
        numeric_features = [c for c in ctx.numeric_columns if c != ctx.target_column]
        if len(numeric_features) < 2:
            return None

        try:
            corrs = df[numeric_features].corrwith(df[ctx.target_column]).dropna()
            corrs = corrs.abs().sort_values(ascending=True)
            if corrs.empty:
                return None
        except Exception:
            return None

        colors = ["#c084fc" if v >= 0.3 else "#60a5fa" if v >= 0.1 else "#334155"
                  for v in corrs.values]

        fig = go.Figure(go.Bar(
            x=corrs.values,
            y=corrs.index.tolist(),
            orientation="h",
            marker=dict(color=colors, line=dict(color="rgba(0,0,0,0)")),
            text=[f"{v:.3f}" for v in corrs.values],
            textposition="outside",
            textfont=dict(color="#94a3b8", size=11),
            hovertemplate="<b>%{y}</b><br>|Korelasyon|: %{x:.4f}<extra></extra>",
        ))
        # Eşik çizgileri
        fig.add_vline(x=0.1, line=dict(color="#fbbf24", width=1, dash="dot"),
                      annotation=dict(text="Zayıf (0.1)", font=dict(color="#fbbf24", size=10)))
        fig.add_vline(x=0.3, line=dict(color="#34d399", width=1, dash="dot"),
                      annotation=dict(text="Orta (0.3)", font=dict(color="#34d399", size=10)))
        fig.update_layout(
            **self._DARK,
            height=max(320, len(corrs) * 38 + 80),
            title=dict(text=f"🔗 Feature → Hedef Korelasyon (|r|) — Hedef: {ctx.target_column}", font=dict(size=14)),
            xaxis=dict(title="Pearson |r|", range=[0, min(1.0, corrs.max() * 1.3)]),
            yaxis=dict(title=""),
        )
        return fig

    # ── 4. Violin Plot ────────────────────────────────────────────────────────

    def _chart_violin(self, df: pd.DataFrame, ctx: DatasetContext) -> go.Figure:
        # Target hariç ilk 6 sayısal kolon
        cols = [c for c in ctx.numeric_columns if c != ctx.target_column][:6]
        if not cols:
            cols = ctx.numeric_columns[:6]

        palette = ["#c084fc", "#60a5fa", "#34d399", "#fbbf24", "#fb923c", "#f87171"]
        fig = go.Figure()
        for i, col in enumerate(cols):
            s = df[col].dropna()
            fig.add_trace(go.Violin(
                y=s,
                name=col,
                box_visible=True,
                meanline_visible=True,
                points="outliers",
                line_color=palette[i % len(palette)],
                fillcolor=palette[i % len(palette)].replace(")", ",0.15)").replace("rgb(", "rgba("),
                marker=dict(color=palette[i % len(palette)], size=3, opacity=0.6),
                spanmode="hard",
            ))
        fig.update_layout(
            **self._DARK,
            height=420,
            title=dict(text="🎻 Sayısal Dağılımlar — Violin Plot", font=dict(size=14)),
            showlegend=False,
            violingap=0.15,
            violinmode="overlay",
        )
        return fig

    # ── 5. Skewness Bar ───────────────────────────────────────────────────────

    def _chart_skewness(self, ctx: DatasetContext) -> go.Figure:
        if not ctx.numeric_summary:
            return go.Figure()

        rows = sorted(ctx.numeric_summary, key=lambda r: abs(r["skewness"]), reverse=True)
        cols  = [r["column"] for r in rows]
        skews = [r["skewness"] for r in rows]
        colors = []
        for s in skews:
            if abs(s) >= 2:   colors.append("#f87171")
            elif abs(s) >= 1: colors.append("#fbbf24")
            else:             colors.append("#34d399")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=cols,
            y=skews,
            marker=dict(color=colors, line=dict(color="rgba(0,0,0,0)")),
            text=[f"{s:.2f}" for s in skews],
            textposition="outside",
            textfont=dict(size=10),
            hovertemplate="<b>%{x}</b><br>Skewness: %{y:.4f}<extra></extra>",
        ))
        # Eşik çizgileri
        for val, label, color in [( 2, "+2 Eşiği", "#f87171"), (-2, "−2 Eşiği", "#f87171"),
                                   ( 1, "+1 Eşiği", "#fbbf24"), (-1, "−1 Eşiği", "#fbbf24")]:
            fig.add_hline(y=val, line=dict(color=color, width=1, dash="dot"),
                          annotation=dict(text=label, font=dict(color=color, size=9), x=0))

        fig.update_layout(
            **self._DARK,
            height=380,
            title=dict(text="📐 Çarpıklık Analizi (Skewness) — 🔴 >±2 Kritik", font=dict(size=14)),
            xaxis=dict(tickangle=-30),
        )
        return fig

    # ── 6. Outlier Sayısı Bar ─────────────────────────────────────────────────

    def _chart_outliers(self, ctx: DatasetContext) -> go.Figure:
        rows = [r for r in ctx.numeric_summary if r["outlier_count"] > 0]
        if not rows:
            fig = go.Figure()
            fig.add_annotation(text="✅ Hiç outlier tespit edilmedi!", x=0.5, y=0.5,
                               xref="paper", yref="paper", showarrow=False,
                               font=dict(size=16, color="#34d399"))
            fig.update_layout(**self._DARK, height=300,
                              title=dict(text="💥 Outlier (Aykırı Değer) Dağılımı", font=dict(size=14)))
            return fig

        rows = sorted(rows, key=lambda r: -r["outlier_count"])
        cols   = [r["column"] for r in rows]
        counts = [r["outlier_count"] for r in rows]

        # Yoğunluk oranına göre renk
        max_c = max(counts)
        colors = [f"rgba(248,113,113,{0.4 + 0.6 * (c / max_c):.2f})" for c in counts]

        fig = go.Figure(go.Bar(
            x=cols, y=counts,
            marker=dict(color=colors, line=dict(color="rgba(0,0,0,0)")),
            text=counts, textposition="outside",
            hovertemplate="<b>%{x}</b><br>Outlier Sayısı: %{y}<extra></extra>",
        ))
        fig.update_layout(
            **self._DARK,
            height=360,
            title=dict(text="💥 Outlier (Aykırı Değer) Sayısı", font=dict(size=14)),
            xaxis=dict(tickangle=-30),
        )
        return fig

    # ── 7. Kategorik Dağılım ─────────────────────────────────────────────────

    def _chart_categorical(self, df: pd.DataFrame, ctx: DatasetContext) -> go.Figure | None:
        # Identifier olmayan ilk anlamlı kat. kolon
        cat_cols = [c for c in ctx.categorical_columns if c not in ctx.possible_identifiers]
        if not cat_cols:
            cat_cols = ctx.categorical_columns
        if not cat_cols:
            return None

        col = cat_cols[0]
        vc  = df[col].value_counts().head(12).reset_index()
        x_col = col if col in vc.columns else vc.columns[0]
        y_col = "count" if "count" in vc.columns else vc.columns[1]

        fig = px.bar(
            vc, x=x_col, y=y_col,
            color=y_col,
            color_continuous_scale=["#1e1b4b", "#7c3aed", "#c084fc"],
            text=y_col,
        )
        fig.update_traces(
            textposition="outside",
            textfont=dict(size=10),
            hovertemplate="<b>%{x}</b><br>Adet: %{y}<extra></extra>",
            marker_line_width=0,
        )
        fig.update_layout(
            **self._DARK,
            height=380,
            title=dict(text=f"🏷️ Kategorik Dağılım: {col}", font=dict(size=14)),
            coloraxis_showscale=False,
            xaxis=dict(tickangle=-30, title=""),
            yaxis=dict(title="Adet"),
        )
        return fig

    # ── 8. Eksik Veri Bar ─────────────────────────────────────────────────────

    def _chart_missing(self, ctx: DatasetContext) -> go.Figure | None:
        if not ctx.missing_report:
            return None

        recs   = sorted(ctx.missing_report, key=lambda r: -r["missing_rate"])
        cols   = [r["column"] for r in recs]
        pcts   = [round(r["missing_rate"] * 100, 1) for r in recs]
        colors = ["#ef4444" if r["severity"] == "critical" else "#f59e0b" for r in recs]

        fig = go.Figure(go.Bar(
            x=pcts, y=cols,
            orientation="h",
            marker=dict(color=colors, line=dict(color="rgba(0,0,0,0)")),
            text=[f"%{p}" for p in pcts],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Eksik: %{x}%<extra></extra>",
        ))
        # %30 kritik eşiği
        fig.add_vline(x=30, line=dict(color="#ef4444", width=1, dash="dot"),
                      annotation=dict(text="Kritik Eşik (%30)", font=dict(color="#ef4444", size=10)))
        fig.update_layout(
            **self._DARK,
            height=max(280, len(cols) * 50 + 80),
            title=dict(text="❓ Eksik Veri Analizi — 🔴 Kritik / 🟡 Uyarı", font=dict(size=14)),
            xaxis=dict(title="Eksik Oran (%)", range=[0, min(100, max(pcts) * 1.3)]),
            yaxis=dict(autorange="reversed"),
        )
        return fig

    # ── 9. Korelasyon Isı Haritası ────────────────────────────────────────────

    def _chart_correlation(self, df: pd.DataFrame, ctx: DatasetContext) -> go.Figure | None:
        if len(ctx.numeric_columns) < 2:
            return None
        corr = df[ctx.numeric_columns].corr().round(2)
        fig = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
        )
        fig.update_traces(
            textfont=dict(size=9),
            hovertemplate="<b>%{x}</b> ↔ <b>%{y}</b><br>r = %{z}<extra></extra>",
        )
        fig.update_layout(
            **self._DARK,
            height=520,
            title=dict(text="🌡️ Korelasyon Isı Haritası — Kırmızı: Negatif / Mavi: Pozitif", font=dict(size=14)),
            coloraxis_colorbar=dict(
                title="r",
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=["-1", "-0.5", "0", "+0.5", "+1"],
                len=0.8,
            ),
        )
        return fig
