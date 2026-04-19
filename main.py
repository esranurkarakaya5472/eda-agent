"""main.py — Entry point: single-run mode or event-driven watcher."""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from tools.log_config import setup_logging
setup_logging()   # dosyaya + konsola yazar

from agent import EDAAgent

logger = logging.getLogger(__name__)

# ── Watcher ───────────────────────────────────────────────────────────────────
_POLL_INTERVAL = 2          # seconds
_INCOMING_DIR  = Path("incoming")
_PROCESSED_DIR = Path("processed")


def _watch() -> None:
    """Poll incoming/ and run agent on every new CSV."""
    _INCOMING_DIR.mkdir(parents=True, exist_ok=True)
    _PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n\033[1m\033[96m{'─'*60}\033[0m")
    print( "\033[1m\033[96m  EDA AGENT  ·  WATCHER MODE\033[0m")
    print(f"\033[1m\033[96m  Watching:  {_INCOMING_DIR.resolve()}\033[0m")
    print(f"\033[1m\033[96m{'─'*60}\033[0m")
    print( "  Drop a CSV into incoming/ to trigger analysis.")
    print( "  Press Ctrl-C to exit.\n")

    agent = EDAAgent(reports_dir="reports", processed_dir=str(_PROCESSED_DIR))
    seen: set[str] = set()

    try:
        while True:
            csv_files = list(_INCOMING_DIR.glob("*.csv"))
            for csv in csv_files:
                if csv.name not in seen:
                    seen.add(csv.name)
                    try:
                        logger.info("Watcher: işleniyor → %s", csv.name)
                        agent.run(csv)
                        logger.info("Watcher: tamamlandı → %s", csv.name)
                    except Exception as exc:           # noqa: BLE001
                        logger.error("Watcher hata: %s — %s", csv.name, exc, exc_info=True)
                        print(f"\033[91m  ERROR processing {csv.name}: {exc}\033[0m")
            time.sleep(_POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n\033[93m  Watcher stopped.\033[0m\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _usage() -> None:
    print(
        "\nUsage:\n"
        "  python main.py data/sample.csv      # analyse a single file\n"
        "  python main.py --watch              # watch incoming/ folder\n"
    )


def main() -> None:
    if len(sys.argv) < 2:
        _usage()
        sys.exit(1)

    if sys.argv[1] == "--watch":
        _watch()
    else:
        file_path = Path(sys.argv[1])
        if not file_path.exists():
            logger.error("Dosya bulunamadı: %s", file_path)
            print(f"\033[91mFile not found: {file_path}\033[0m")
            sys.exit(1)
        logger.info("Tek dosya modu: %s", file_path)
        agent = EDAAgent()
        agent.run(file_path)
        logger.info("Analiz tamamlandı: %s", file_path)


if __name__ == "__main__":
    main()
