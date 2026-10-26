# LendSmart

LendSmart is a loan default risk scoring tool: given information about a loan
applicant, it estimates how likely they are to default. It is a data science
portfolio project.

## About the user

- The user is a learner. Explain each step in simple, plain terms: what you are
  doing, why, and what the result means. Avoid jargon, or define it when you use it.
- Prefer small, understandable steps over large changes made all at once.

## Project layout

- `notebooks/` — Jupyter notebooks for exploration, analysis, and experiments.
- `src/` — reusable Python code (functions for loading data, cleaning, features,
  training, scoring). Notebooks should import from `src/` rather than copy code.
- `tests/` — tests for the code in `src/`.
- `data/` — local data files only. Never committed (see below).
- `requirements.txt` — the project's Python packages.

## Code style

- Keep code simple and readable. Choose the clear approach over the clever one.
- Comment the code so a learner can follow it: explain what each step does and why.
- Use pandas for data handling and scikit-learn for modeling.

## Environment

- Always use the `.venv` virtual environment. Run Python and tools through it,
  e.g. `.venv/bin/python`, `.venv/bin/pip`, `.venv/bin/jupyter`, or activate it
  first with `source .venv/bin/activate`.
- Never use the system Python or install packages globally.

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

**Ask the user before installing any package**, including one from this list
that is not installed yet, and explain what it is for. Packages not on this
list need the user's approval before they are used at all. When a package is
installed, add it to `requirements.txt`.

## Git rules

- **Never commit data files or secrets.** This includes anything in `data/`,
  `*.csv` and other datasets, `.env` files, API keys, passwords, and tokens.
  These are listed in `.gitignore`; check `git status` before any commit to be sure.
- After each finished piece of work, remind the user to commit, and suggest a
  short commit message describing the change.
