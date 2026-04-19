"""tools/llm.py — Generative AI integration for intelligent analysis."""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time

from tools.profiler import DatasetContext

logger = logging.getLogger(__name__)

# Try to import packages gracefully
try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Premium Spinner ──────────────────────────────────────────────────────────

class PremiumSpinner:
    """A threaded braille spinner for long-running CLI tasks."""
    def __init__(self, message: str = "🧠 AI Verileri Değerlendiriyor"):
        # Braille pattern creates a smooth circular motion
        self.chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        self.message = message
        self.is_running = False
        self.thread: threading.Thread | None = None

    def spin(self) -> None:
        idx = 0
        while self.is_running:
            char = self.chars[idx]
            # \033[95m is magenta, \033[1m is bold
            sys.stdout.write(f"\r  \033[95m{char}\033[0m  \033[1m{self.message}...\033[0m")
            sys.stdout.flush()
            time.sleep(0.08)
            idx = (idx + 1) % len(self.chars)

    def start(self) -> None:
        self.is_running = True
        self.thread = threading.Thread(target=self.spin)
        self.thread.start()

    def stop(self) -> None:
        self.is_running = False
        if self.thread:
            self.thread.join()
        # Clear the entire line smoothly
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()


# ── AI Analyst ───────────────────────────────────────────────────────────────

class LLMAnalyst:
    """Wraps Gemini API for executive data summarization."""
    
    SYSTEM_PROMPT = (
        "Sen Silikon Vadisi'nde çalışan Senior bir Veri Bilimci (Data Scientist) Ajansın. "
        "Görevin, sana verilen 'Veri Profili (Dataset Profiling)' istatistiklerini ve risklerini inceleyip, "
        "veri kalitesi, dağılımları ve potansiyel sorunları hakkında profesyonel bir 'Yönetici Özeti' (Executive Summary) yazmaktır. "
        "Analizini, teknik bilgiye sahip bir veri ekibine veya yöneticilere sunuyormuş gibi yap. "
        "Tespit edilen eksik verileri (missing values), aykırı değerleri (outliers), ve veri çarpıklıklarını (skewness) modellemede çıkaracağı potansiyel problemlerle harmanla. "
        "CEVABINI MUTLAKA TÜRKÇE YAZ. Dili son derece profesyonel, akıcı ve aksiyon odaklı olsun. Markdown formatını kullanabilirsin."
    )

    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.has_credentials = bool(self.api_key)
        if HAS_GENAI and self.has_credentials:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None

    def is_available(self) -> bool:
        return self.client is not None

    def _build_prompt(self, ctx: DatasetContext) -> str:
        # Create a compressed, readable representation of context
        payload = {
            "Satır Sayısı": ctx.rows,
            "Sütun Sayısı": ctx.columns,
            "Hedef Sütun (Target)": ctx.target_column,
            "Sayısal Sütunlar": ctx.numeric_columns,
            "Kategorik Sütunlar": ctx.categorical_columns,
            "Eksik Veri (Missing Data) Raporu": ctx.missing_report,
            "Riskler ve Uyumsuzluklar": ctx.risks,
            "Çarpık (Skewed) Sütunlar": ctx.skewed_columns,
            "Yüksek Kardinalite (Olası ID'ler)": ctx.possible_identifiers,
            "Düşük Varyanslı Sütunlar": ctx.low_variance_columns
        }
        return f"Aşağıdaki Veri Seti Profiline göre lütfen detaylı bir profesyonel veri analizi ve aksiyon adımları yaz:\n\n{json.dumps(payload, indent=2, ensure_ascii=False)}"

    def analyze(self, ctx: DatasetContext) -> str | None:
        if not HAS_GENAI:
            logger.warning("google-genai kütüphanesi kurulu değil!")
            return None
            
        if not self.has_credentials:
            logger.warning("GEMINI_API_KEY ortam değişkeni bulunamadı!")
            return None
        
        spinner = PremiumSpinner("🧠 AI Veri Setini Analiz Ediyor")
        spinner.start()
        
        try:
            # Using standard recommended model
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=self._build_prompt(ctx),
                config=types.GenerateContentConfig(
                    system_instruction=self.SYSTEM_PROMPT,
                    temperature=0.4,
                )
            )
            return response.text
        except Exception as e:
            logger.error(f"LLM API Error: {e}")
            return None
        finally:
            spinner.stop()

    def start_chat(self, ctx: DatasetContext) -> Any:
        """Starts an interactive chat session seeded with the dataset context."""
        if not self.is_available():
            return None
        
        system_msg = (
            "Sen enerjik, hevesli ama profesyonelliği ve kurumsal resmiyeti asla elden bırakmayan "
            "Kıdemli bir Veri Bilimci (Data Scientist) Ajansın. Karşındaki takım arkadaşına/yöneticine projesinde yardım ediyorsun. "
            "Aşağıda analiz ettiğimiz veri setinin detaylı istatistikleri ve Auto-Cleaner logları yer alıyor. "
            "Kullanıcının soracağı soruları sadece bu verilere dayanarak, pozitif, çözüm odaklı ve akıcı bir Türkçe ile cevapla.\n\n"
            f"VERİ PROFİLİ VE DURUMU:\n{self._build_prompt(ctx)}"
        )
        try:
            chat = self.client.chats.create(
                model='gemini-2.5-flash',
                config=types.GenerateContentConfig(
                    system_instruction=system_msg,
                    temperature=0.6,
                )
            )
            return chat
        except Exception as e:
            logger.error(f"LLM Chat Başlatma Hatası: {e}")
            return None
