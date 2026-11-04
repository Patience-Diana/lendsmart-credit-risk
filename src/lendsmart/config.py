"""Project settings: all folder paths and database settings live here, in one place.

Other files import these names (e.g. `from lendsmart.config import DATA_PATH`)
instead of working out paths themselves. If a folder ever moves, only this
file needs to change.
"""

# os lets us read environment variables (settings given to a program from outside).
import os

# Path helps us build file paths that work on any computer (Mac, Windows, Linux).
from pathlib import Path

# Find the project's main folder.
# By default: __file__ is this file (src/lendsmart/config.py), and each .parent
# goes up one folder: lendsmart/ -> src/ -> the main project folder.
# That only works when the code stays inside the project (the editable install
# in .venv). In the Docker image the package is copied into Python's own folder,
# so the Dockerfile sets the LENDSMART_ROOT environment variable to say where
# the project files (models/, reports/) are instead.
PROJECT_ROOT = Path(
    os.environ.get("LENDSMART_ROOT", Path(__file__).resolve().parent.parent.parent)
)

# Raw data. The data/ folder is ignored by git, so it never gets committed.
DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "cs-training.csv"

# Charts and other outputs that are safe to commit and show in the README.
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Saved, trained models. Also ignored by git (they are large binary files).
MODELS_DIR = PROJECT_ROOT / "models"

# ---------- Database settings ----------
# The real values (including the password) live in the .env file, which git
# ignores. docker-compose.yml reads the same file, so Python and Docker always
# agree on the user, password, database name and port.
ENV_PATH = PROJECT_ROOT / ".env"


def load_env_file(path=ENV_PATH):
    """Read KEY=VALUE lines from a .env file into the environment variables.

    Settings that are already in the environment win. That matters later: inside
    a Docker container there is no .env file, and Docker passes the settings in
    directly as environment variables instead.
    """
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        # Skip blank lines and comments.
        if not line or line.startswith("#"):
            continue
        # "POSTGRES_USER=lendsmart" -> key "POSTGRES_USER", value "lendsmart".
        key, _, value = line.partition("=")
        # setdefault only sets the value if it isn't already set.
        os.environ.setdefault(key.strip(), value.strip())


def database_url():
    """Build the address Python uses to connect to Postgres.

    It looks like: postgresql+psycopg://user:password@host:port/database
    - "postgresql+psycopg" means: a Postgres database, reached with the psycopg driver.
    """
    load_env_file()
    user = os.environ["POSTGRES_USER"]
    password = os.environ["POSTGRES_PASSWORD"]
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    database = os.environ["POSTGRES_DB"]
    return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{database}"
