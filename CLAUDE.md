# LendSmart

LendSmart is a loan default risk scoring tool: given information about a loan
applicant, it estimates how likely they are to default. It is a data science
portfolio project, built with a production-style structure.

It is also how the user learns two new tools: storing data in a **PostgreSQL**
database, and packaging the app with **Docker**.

## Roadmap

Everything is due **Friday 2026-10-30**. Each part is kept minimal to fit.

- Wednesday: package structure, train/validation/test split, first tests,
  logistic regression + XGBoost.
- Thursday: cost-based cutoff, final test score, SHAP chart, age fairness check;
  then Postgres running in Docker, with the data loaded into a table.
- Friday: a simple FastAPI endpoint, a Dockerfile + docker-compose for the app
  and database, and the README.

Cut first if behind: the "model without age" comparison, XGBoost tuning,
storing predictions in Postgres.

## About the user

- The user is a learner. Explain each step in simple, plain terms: what you are
  doing, why, and what the result means. Avoid jargon, or define it when you use it.
- Prefer small, understandable steps over large changes made all at once.

## Project layout

- `src/lendsmart/` — the app's code, as one installable package named `lendsmart`.
  Notebooks, scripts, tests, the API and the Docker image all import from it
  (e.g. `from lendsmart.data import load_data`) rather than copying code.
  - `config.py` — all paths and settings in one place. Other modules import
    paths from here instead of building their own.
- `scripts/` — commands to run (e.g. training, loading data into Postgres).
- `notebooks/` — Jupyter notebooks for exploration and telling the story.
- `tests/` — tests for the code in `src/lendsmart/`.
- `reports/figures/` — charts for the README (committed).
- `models/` — saved trained models. Never committed.
- `data/` — local data files only. Never committed (see below).
- `pyproject.toml` — describes the package and its core dependencies.
- `requirements.txt` — all the project's Python packages, including tools.

## Code style

- Keep code simple and readable. Choose the clear approach over the clever one.
- Comment the code so a learner can follow it: explain what each step does and why.
- Use pandas for data handling and scikit-learn for modeling.

## Environment

- Always use the `.venv` virtual environment. Run Python and tools through it,
  e.g. `.venv/bin/python`, `.venv/bin/pip`, `.venv/bin/jupyter`, or activate it
  first with `source .venv/bin/activate`.
- Never use the system Python or install packages globally.
- The `lendsmart` package is installed into `.venv` in editable mode
  (`.venv/bin/pip install -e .`), so code changes take effect without reinstalling.
- Run modules from the project root with `-m`, e.g. `.venv/bin/python -m lendsmart.eda`.
- Postgres runs in Docker, not the Homebrew install on the Mac.

## Allowed packages

These packages are approved for use in this project:

- `pandas` — loading and cleaning tabular data
- `numpy` — numerical arrays and math
- `scikit-learn` — main modeling library (preprocessing, models, metrics)
- `xgboost` — gradient-boosted tree models
- `shap` — explaining which features drive a model's predictions
- `matplotlib` and `seaborn` — charts and plots
- `jupyter` — running notebooks
- `fastapi` — serving the model as a simple web API
- `uvicorn` — running the FastAPI app
- `pytest` — running the tests in `tests/`

Planned for later steps, not yet approved: `sqlalchemy` and `psycopg` (talking
to Postgres from Python).

**Ask the user before installing any package**, including one from this list
that is not installed yet, and explain what it is for. Packages not on this
list need the user's approval before they are used at all. When a package is
installed, add it to `requirements.txt`.

## Git rules

- **Never commit data files or secrets.** This includes anything in `data/`,
  `*.csv` and other datasets, `.env` files, API keys, passwords, and tokens.
  These are listed in `.gitignore`; check `git status` before any commit to be sure.
- Work one step at a time: each step gets its own branch, then a pull request
  into `main` that the user reviews and merges themselves.
- After each finished piece of work, remind the user to commit, and suggest a
  short commit message describing the change.
- Database passwords and connection settings go in `.env` (never committed).
  Commit a `.env.example` with placeholder values instead.
