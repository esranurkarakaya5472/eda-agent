"""app.py — Streamlit Web Interface for EDA Agent (v2 — Production Ready)."""
from __future__ import annotations

import logging
import os
import io
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

# ── Loglama ve ortam değişkenleri ─────────────────────────────────────────────
load_dotenv()
from tools.log_config import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

from agent import EDAAgent

# ─────────────────────────────────────────────────────────────────────────────
# Sayfa Yapılandırması
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EDA Agent Hub",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Özel CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Genel arka plan ve font */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0b0f19 0%, #111827 100%);
        color: #e2e8f0;
    }
    [data-testid="stSidebar"] {
        background: rgba(17, 24, 39, 0.95);
        border-right: 1px solid rgba(255,255,255,0.07);
    }

    /* Metrik kartları */
    [data-testid="metric-container"] {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 16px !important;
        backdrop-filter: blur(8px);
    }
    [data-testid="metric-container"] label {
        color: #94a3b8 !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #c084fc !important;
        font-size: 1.7rem !important;
        font-weight: 700 !important;
    }

    /* Başlık gradient */
    .hero-title {
        background: linear-gradient(90deg, #c084fc, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        line-height: 1.2;
        margin-bottom: 0.3rem;
    }
    .hero-sub {
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    /* Risk badge */
    .risk-badge {
        display: inline-block;
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.35);
        color: #fca5a5;
        border-radius: 6px;
        padding: 3px 10px;
        font-size: 0.82rem;
        margin: 3px 2px;
    }
    .warn-badge {
        background: rgba(251, 191, 36, 0.12);
        border-color: rgba(251, 191, 36, 0.35);
        color: #fde68a;
    }
    .ok-badge {
        background: rgba(52, 211, 153, 0.12);
        border-color: rgba(52, 211, 153, 0.35);
        color: #6ee7b7;
    }

    /* Buton */
    [data-testid="stButton"] > button[kind="primary"] {
        background: linear-gradient(135deg, #7c3aed, #4f46e5);
        border: none;
        color: white;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.6rem 1.4rem;
        font-size: 0.95rem;
        transition: all 0.2s;
        width: 100%;
    }
    [data-testid="stButton"] > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(124, 58, 237, 0.4);
    }

    /* Chat mesajları */
    [data-testid="stChatMessage"] {
        background: rgba(30, 41, 59, 0.4);
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.05);
    }

    /* Sekme */
    [data-baseweb="tab-list"] {
        gap: 6px;
        background: rgba(15, 20, 30, 0.6);
        border-radius: 10px;
        padding: 4px;
    }
    [data-baseweb="tab"] {
        border-radius: 8px !important;
    }

    /* Sidebar uploader */
    [data-testid="stFileUploaderDropzone"] {
        border: 2px dashed rgba(124, 58, 237, 0.4) !important;
        border-radius: 12px !important;
        background: rgba(124, 58, 237, 0.05) !important;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: rgba(124, 58, 237, 0.8) !important;
        background: rgba(124, 58, 237, 0.1) !important;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }

    /* Divider */
    hr { border-color: rgba(255,255,255,0.06); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session State Başlatma
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULTS: dict[str, object] = {
    "chat_history":    [],
    "agent_run":       False,
    "dashboard_path":  None,
    "chat_session":    None,
    "agent_context":   None,   # DatasetContext — metrik kartları için
    "file_name":       None,
    "df_preview":      None,   # Yüklenen ham veri önizlemesi
}
for key, val in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ─────────────────────────────────────────────────────────────────────────────
# Yardımcı Fonksiyonlar
# ─────────────────────────────────────────────────────────────────────────────

def _reset_state() -> None:
    """Tüm session state'i sıfırla — yeni analiz başlatmak için."""
    for key, val in _DEFAULTS.items():
        st.session_state[key] = val
    logger.info("Session state sıfırlandı, yeni analiz için hazır.")
    st.rerun()


def _validate_csv(uploaded) -> tuple[bool, str, pd.DataFrame | None]:
    """
    Yüklenen dosyayı doğrular.
    Returns: (geçerli_mi, hata_mesajı, dataframe_or_None)
    """
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        logger.warning("CSV okuma hatası: %s", e)
        return False, f"Dosya okuma hatası: {e}", None

    if df.empty:
        return False, "Yüklenen dosya boş (hiç veri satırı yok).", None

    if df.shape[1] < 2:
        return False, "Veri seti en az 2 sütun içermelidir.", None

    if df.columns.duplicated().any():
        dupes = df.columns[df.columns.duplicated()].tolist()
        return False, f"Tekrarlayan sütun adları mevcut: `{dupes}` — lütfen düzeltin.", None

    return True, "", df


def _metric_card(label: str, value: str, delta: str | None = None) -> None:
    """Tek bir metrik kartı gösterir."""
    st.metric(label=label, value=value, delta=delta)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # ── Marka / Logo ──────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding: 12px 0 20px 0;">
        <div style="font-size:2.4rem;">🤖</div>
        <div style="font-weight:800; font-size:1.15rem; color:#c084fc; letter-spacing:-0.01em;">EDA Agent</div>
        <div style="font-size:0.72rem; color:#475569; margin-top:2px;">Autonomous Data Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Dosya Yükleme ─────────────────────────────────────────────────────────
    st.markdown("#### 📂 Veri Yükle")

    if st.session_state.agent_run:
        # Analiz yapıldıysa dosya adını göster, reset butonu sun
        st.success(f"✅ **{st.session_state.file_name}** analiz edildi!")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Yeni Analiz Başlat", type="primary", key="btn_reset"):
            logger.info("Kullanıcı yeni analiz isteğinde bulundu — state sıfırlanıyor.")
            _reset_state()
    else:
        uploaded_file = st.file_uploader(
            label="CSV dosyanızı sürükleyin veya seçin",
            type=["csv"],
            help="Yalnızca .csv formatı desteklenmektedir.",
            key="file_uploader",
        )

        if uploaded_file is not None:
            # ── Hızlı Doğrulama ───────────────────────────────────────────────
            valid, err_msg, df_preview = _validate_csv(uploaded_file)
            uploaded_file.seek(0)   # pointer'ı başa sar — tekrar kullanılacak

            if not valid:
                st.error(f"⛔ **Geçersiz Dosya:** {err_msg}")
                logger.warning("Geçersiz dosya yüklendi: %s", err_msg)
            else:
                # Önizleme bilgilerini kaydet
                st.session_state.df_preview = df_preview
                rows, cols = df_preview.shape
                missing_pct = round(df_preview.isna().mean().mean() * 100, 1)

                st.success(f"✅ **{uploaded_file.name}** hazır!")

                # Hızlı istatistikler
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    <div style="background:rgba(124,58,237,0.12); border-radius:8px;
                                padding:8px 12px; text-align:center; border:1px solid rgba(124,58,237,0.25);">
                        <div style="font-size:1.3rem; font-weight:700; color:#c084fc;">{rows:,}</div>
                        <div style="font-size:0.7rem; color:#64748b; text-transform:uppercase;">Satır</div>
                    </div>""", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div style="background:rgba(96,165,250,0.12); border-radius:8px;
                                padding:8px 12px; text-align:center; border:1px solid rgba(96,165,250,0.25);">
                        <div style="font-size:1.3rem; font-weight:700; color:#60a5fa;">{cols}</div>
                        <div style="font-size:0.7rem; color:#64748b; text-transform:uppercase;">Sütun</div>
                    </div>""", unsafe_allow_html=True)

                if missing_pct > 0:
                    st.warning(f"⚠️ Genel eksik veri oranı: **%{missing_pct}**")

                st.markdown("<br>", unsafe_allow_html=True)

                if st.button("🚀 Otonom Analizi Başlat", type="primary", key="btn_run"):
                    # Dosyayı incoming/ klasörüne kaydet
                    incoming_dir = Path("incoming")
                    incoming_dir.mkdir(exist_ok=True)
                    file_path = incoming_dir / uploaded_file.name

                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    logger.info("Dosya incoming/'a kaydedildi: %s (%d×%d)", uploaded_file.name, rows, cols)

                    # ── Agent Çalıştır ────────────────────────────────────────
                    try:
                        with st.spinner("🧠 Ajan bağımsız çalışıyor... Lütfen bekleyin."):
                            agent = EDAAgent()
                            result = agent.run(file_path)

                        # Sonuçları session state'e aktar
                        ctx = result["context"]
                        dash_path = Path("reports") / f"dashboard_{file_path.stem}.html"

                        st.session_state.agent_context  = ctx
                        st.session_state.file_name      = uploaded_file.name
                        st.session_state.dashboard_path = dash_path if dash_path.exists() else None

                        # Chat oturumu başlat
                        if agent._llm.is_available():
                            st.session_state.chat_session = agent._llm.start_chat(ctx)
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "text": (
                                    "Tebrikler, analiz tamamlandı! ✨ Verin artık pırıl pırıl temizlendi "
                                    "ve görsel şölene dönüştürüldü. Grafikleri **Dashboard** sekmesinde "
                                    "inceleyebilir, verini **Önizleme** sekmesinde görebilirsin. "
                                    "Veri setin hakkında merak ettiğin her şeyi sormaktan çekinme. 😎"
                                ),
                            })
                            logger.info("LLM sohbet oturumu başlatıldı.")
                        else:
                            logger.warning("LLM mevcut değil — chat oturumu açılmadı.")

                        st.session_state.agent_run = True
                        logger.info("Agent çalışması başarıyla tamamlandı: %s", uploaded_file.name)
                        st.rerun()

                    except Exception as exc:
                        logger.error("Agent çalışma hatası: %s", exc, exc_info=True)
                        st.error(f"❌ **Analiz sırasında bir hata oluştu:**\n\n`{exc}`")
                        st.info("Lütfen dosyanın geçerli bir CSV olduğundan emin olun ve tekrar deneyin.")

    # ── Alt Bilgi ─────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("""
    <div style="font-size:0.7rem; color:#334155; text-align:center; line-height:1.6;">
        Powered by <b style="color:#c084fc;">Gemini 2.5 Flash</b><br>
        EDA Agent — Autonomous Intelligence
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Ana İçerik Alanı
# ─────────────────────────────────────────────────────────────────────────────

if not st.session_state.agent_run:
    # ── Hero Boş Durum ────────────────────────────────────────────────────────
    st.markdown('<div class="hero-title">Data Intelligence Hub</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Verinizi yükleyin — yapay zeka analiz etsin, temizlesin, görselleştirsin.</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    features = [
        ("🔬", "Derin Profil", "Eksik veri, skewness, outlier, hedef kolon tespiti"),
        ("🛁", "Oto-Temizleme", "Imputation, outlier capping, düşük varyanslı kolon atma"),
        ("🧠", "AI Yönetici Özeti", "Gemini ile Türkçe profesyonel veri analiz raporu"),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3], features):
        with col:
            st.markdown(f"""
            <div style="background:rgba(30,41,59,0.5); border:1px solid rgba(255,255,255,0.07);
                        border-radius:14px; padding:22px; text-align:center; height:140px;">
                <div style="font-size:2rem;">{icon}</div>
                <div style="font-weight:700; color:#e2e8f0; margin:6px 0 4px;">{title}</div>
                <div style="font-size:0.78rem; color:#64748b;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👈  Lütfen sol menüden bir CSV dosyası yükleyip analizi başlatın.")

else:
    # ─────────────────────────────────────────────────────────────────────────
    # Analiz Tamamlandı → Sonuç Ekranı
    # ─────────────────────────────────────────────────────────────────────────
    ctx = st.session_state.agent_context

    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="hero-title">📊 {st.session_state.file_name}</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="hero-sub">Otonom analiz tamamlandı — aşağıdaki sekmeleri keşfet.</div>', unsafe_allow_html=True)

    # ── Metrik Kartları ───────────────────────────────────────────────────────
    if ctx:
        missing_count = len(ctx.missing_report)
        risk_count    = len(ctx.risks)

        m1, m2, m3, m4, m5 = st.columns(5)
        with m1: st.metric("📋 Toplam Satır",    f"{ctx.rows:,}")
        with m2: st.metric("📐 Toplam Sütun",    f"{ctx.columns}")
        with m3: st.metric("🔢 Sayısal Sütun",   f"{len(ctx.numeric_columns)}")
        with m4: st.metric("🏷️ Kategorik Sütun", f"{len(ctx.categorical_columns)}")
        with m5: st.metric("⚠️ Risk Sayısı",     f"{risk_count}",
                           delta=None if risk_count == 0 else f"{'Temiz' if risk_count == 0 else f'{missing_count} eksik kolon'}",
                           delta_color="normal" if risk_count == 0 else "inverse")

        # Hedef & Risk özeti
        if ctx.target_column:
            st.markdown(
                f'<span class="ok-badge">🎯 Hedef Kolon: <b>{ctx.target_column}</b></span>',
                unsafe_allow_html=True,
            )
        if ctx.risks:
            risk_html = "".join(
                f'<span class="{"risk-badge" if "critical" in r or "heavily" in r else "warn-badge"}">'
                f'{r[:55]}{"…" if len(r) > 55 else ""}</span>'
                for r in ctx.risks[:8]
            )
            st.markdown(
                f'<div style="display:flex; flex-wrap:wrap; gap:4px; margin-top:6px;">{risk_html}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Sekmeler ─────────────────────────────────────────────────────────────
    tab_dash, tab_prev, tab_chat = st.tabs([
        "📊 Dashboard",
        "🔍 Veri Önizleme",
        "💬 AI Analisti ile Sohbet",
    ])

    # ── Tab 1: Dashboard ──────────────────────────────────────────────────────
    with tab_dash:
        if st.session_state.dashboard_path and Path(st.session_state.dashboard_path).exists():
            html_string = Path(st.session_state.dashboard_path).read_text(encoding="utf-8")
            st.caption("💡 Grafikleri tam ekrana görmek için aşağıya kaydırabilirsiniz.")
            components.html(html_string, height=1500, scrolling=True)
        else:
            st.warning("⚠️ Dashboard HTML dosyası bulunamadı. Plotly veya Jinja2 kurulu değil olabilir.")
            st.code("pip install plotly jinja2", language="bash")

    # ── Tab 2: Veri Önizleme ──────────────────────────────────────────────────
    with tab_prev:
        df_prev = st.session_state.df_preview
        if df_prev is not None:
            sub1, sub2 = st.tabs(["📄 İlk 50 Satır", "📊 İstatistiksel Özet"])

            with sub1:
                st.dataframe(
                    df_prev.head(50),
                    use_container_width=True,
                    height=400,
                )

            with sub2:
                try:
                    desc = df_prev.describe(include="all").T.reset_index()
                    desc.columns = ["Sütun"] + list(desc.columns[1:])
                    st.dataframe(desc, use_container_width=True, height=400)
                except Exception as e:
                    st.error(f"İstatistik hesaplama hatası: {e}")

            # İndirme butonu — temizlenmiş veri için
            clean_path = Path("processed") / f"cleaned_{st.session_state.file_name}"
            if clean_path.exists():
                with open(clean_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Temizlenmiş CSV'yi İndir",
                        data=f,
                        file_name=f"cleaned_{st.session_state.file_name}",
                        mime="text/csv",
                        key="download_clean",
                    )
        else:
            st.info("Önizleme verisi bulunamadı.")

    # ── Tab 3: AI Chat ────────────────────────────────────────────────────────
    with tab_chat:
        st.markdown("### 🤖 Veri Analistinizle Sohbet Edin")

        if not st.session_state.chat_session:
            st.warning(
                "⚠️ AI sohbet özelliği aktif değil. "
                "`.env` dosyanızda `GEMINI_API_KEY` tanımlı olduğundan emin olun.",
                icon="🔑",
            )
            st.code("GEMINI_API_KEY=your_key_here   # .env dosyasına ekle", language="bash")
        else:
            # Geçmiş mesajlar
            for msg in st.session_state.chat_history:
                role = "user" if msg["role"] == "user" else "assistant"
                with st.chat_message(role):
                    st.markdown(msg["text"])

            # Kullanıcı girişi
            if prompt := st.chat_input(
                "Veri setin hakkında ne merak ediyorsun?",
                key="chat_input",
            ):
                st.session_state.chat_history.append({"role": "user", "text": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Analiz ediliyor…"):
                        try:
                            response = st.session_state.chat_session.send_message(prompt)
                            answer = response.text
                            logger.info("Chat yanıtı alındı — %d karakter.", len(answer))
                        except Exception as exc:
                            answer = f"❌ Yanıt alınamadı: `{exc}`"
                            logger.error("Chat yanıt hatası: %s", exc, exc_info=True)

                        st.markdown(answer)
                        st.session_state.chat_history.append({"role": "assistant", "text": answer})
