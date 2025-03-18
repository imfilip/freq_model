"""
Konfiguracja projektu - stałe, ścieżki, parametry modeli itp.
"""

import os
from pathlib import Path

# Ścieżki podstawowe
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

# Tworzenie katalogów jeśli nie istnieją
for dir_path in [DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)