# Contributing to ScientificLab

Thank you for helping! Scientific accuracy comes before the number of features.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Before opening a pull request

```bash
ruff check . && ruff format --check .
mypy
pytest
```

All three must pass (CI runs them on Python 3.11, 3.12 and 3.13, and with the
oldest supported dependencies). Documentation examples in `docs/` are executed by
the tests; build the site with `pip install -e ".[docs]" && mkdocs build --strict`.

## Rules for new features

1. Implement a real calculation - no placeholder functions.
2. Type-annotate everything; document public functions (Google-style docstrings)
   including **units, sign conventions, validity domain and limits**.
3. Validate inputs in `utils.validation` helpers and raise explicit exceptions
   from `utils.exceptions`; never return a physically or mathematically
   inconsistent result silently.
4. Keep computation, validation and display separate.
5. Add tests with known reference values: normal cases, boundary values, invalid
   inputs, division by zero, no solution, complex solutions, numerical precision.
6. Never confuse symbolic (exact) results with numerical approximations - say
   which one a function returns.
7. Avoid new dependencies unless clearly needed.
8. Update `README.md` and `CHANGELOG.md`.

## Pull requests and issues

Use the templates in `.github/`. Describe the physics or mathematics behind the
change and cite a reference when relevant.
