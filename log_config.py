"""tools/log_config.py — Merkezi loglama yapılandırması (Streamlit + CLI ortak kullanır)."""
from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path


_CONFIGURED = False   # idempotent: ikinci çağrıda tekrar kurulmasın


def setup_logging(
    log_file: str | Path = "eda_agent.log",
    console_level: int = logging.WARNING,
    file_level: int = logging.DEBUG,
) -> None:
    """
    Loglama altyapısını yapılandırır.

    - Console:  WARNING ve üzeri mesajlar (kullanıcıya gürültü çıkarmaz)
    - Dosya:    DEBUG dahil tüm mesajlar dönen RotatingFileHandler
                (10 MB × 3 yedek — üretim ortamı dostu)

    Args:
        log_file:       Log dosyasının yolu (varsayılan: proje kökü/eda_agent.log)
        console_level:  Konsol için minimum log seviyesi.
        file_level:     Dosyaya yazılacak minimum log seviyesi.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # en düşük seviye — handler'lar filtreler

    # ── Format ────────────────────────────────────────────────────────────────
    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Handler 1: Rotating dosya ─────────────────────────────────────────────
    file_handler = logging.handlers.RotatingFileHandler(
        filename=str(log_path),
        maxBytes=10 * 1024 * 1024,   # 10 MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(fmt)

    # ── Handler 2: Konsol (stderr) ────────────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(fmt)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("watchdog").setLevel(logging.WARNING)

    logging.info("EDA Agent loglama sistemi başlatıldı → %s", log_path.resolve())
