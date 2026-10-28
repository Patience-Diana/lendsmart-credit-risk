"""Project settings: all folder and file paths live here, in one place.

Other files import these names (e.g. `from lendsmart.config import DATA_PATH`)
instead of working out paths themselves. If a folder ever moves, only this
file needs to change.
"""

# Path helps us build file paths that work on any computer (Mac, Windows, Linux).
from pathlib import Path

# Find the project's main folder.
# __file__ is this file (src/lendsmart/config.py). Each .parent goes up one folder:
# lendsmart/ -> src/ -> the main project folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Raw data. The data/ folder is ignored by git, so it never gets committed.
DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "cs-training.csv"

# Charts and other outputs that are safe to commit and show in the README.
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Saved, trained models. Also ignored by git (they are large binary files).
MODELS_DIR = PROJECT_ROOT / "models"
