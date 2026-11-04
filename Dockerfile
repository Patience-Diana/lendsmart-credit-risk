# Dockerfile: the recipe for the LendSmart API image.
#   Build and run it with the database:  docker compose up -d --build
#
# Before building, these must exist locally (they are not in git):
#   .venv/bin/python scripts/train.py             -> models/
#   .venv/bin/python scripts/final_evaluation.py  -> reports/final_results.json
#   bash scripts/download_wheels.sh               -> wheels/ (Linux package files)

# 1. Start from an official image that already has Python 3.10 (the same
#    version as the .venv). "slim" leaves out extras, so the image stays small.
FROM python:3.10-slim

# 2. XGBoost needs the libgomp system library (for running on several CPU
#    cores), which the slim image doesn't include. Install it, then delete
#    apt's download lists so they don't take up space in the image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 3. All the following commands run inside the /app folder of the image.
#    LENDSMART_ROOT tells config.py that /app is the project folder, where
#    models/ and reports/ are copied below.
WORKDIR /app
ENV LENDSMART_ROOT=/app

# 4. Install the pinned packages FIRST, before copying the code.
#    Docker caches each step: if only the code changes, this slow step is
#    reused instead of reinstalling everything on every build.
#    The package files were downloaded beforehand into wheels/ by
#    scripts/download_wheels.sh, so this step needs no internet:
#    --no-index        don't look on the internet (PyPI) at all
#    --find-links      look in the wheels/ folder instead
COPY requirements-api.txt .
COPY wheels/ /tmp/wheels/
RUN pip install --no-cache-dir --no-index --find-links /tmp/wheels -r requirements-api.txt

# 5. Copy the lendsmart package and install it.
#    --no-deps: don't install pyproject.toml's dependencies (shap, matplotlib...),
#    because the API doesn't use them and step 4 already installed what it needs.
#    Building it needs setuptools and wheel, which also come from /tmp/wheels.
#    Then delete /tmp/wheels: the packages are installed, the files aren't needed.
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir --no-deps --no-index --find-links /tmp/wheels . \
    && rm -rf /tmp/wheels

# 6. Copy the trained model and the chosen cutoff, which the API loads.
#    config.py finds them relative to the project folder, which here is /app.
COPY models/ models/
COPY reports/final_results.json reports/final_results.json

# 7. Don't run as the all-powerful "root" user. If the API were ever broken
#    into, a normal user limits the damage.
RUN useradd --create-home appuser
USER appuser

# 8. Document that the API listens on port 8000, then start it.
#    --host 0.0.0.0 means "accept connections from outside the container"
#    (the default, 127.0.0.1, would only accept them from inside).
EXPOSE 8000
CMD ["uvicorn", "lendsmart.api:app", "--host", "0.0.0.0", "--port", "8000"]
