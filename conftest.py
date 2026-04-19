"""pytest yapılandırma dosyası — proje kökünü Python path'e ekler."""
import sys
import os

# Proje kökünü her zaman path'e ekle — tests/ içinden import sorunlarını önler
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
