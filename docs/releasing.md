# Releasing

ScientificLab follows [Semantic Versioning](https://semver.org). While the major
version is 0, a minor bump (`0.x`) may change the public API; the
[changelog](changelog.md) lists every such change. From 1.0.0, breaking changes
only happen in a major release.

## One-time setup (maintainer)

1. Create the project on PyPI (and TestPyPI) and add a **trusted publisher**:
   owner and repository, workflow `release.yml`, environment `pypi`
   (`testpypi` for TestPyPI). No API token is stored anywhere.
2. In the GitHub repository settings, create the environments `pypi` and
   `testpypi` (optionally require a manual approval for `pypi`).
3. Replace every `OWNER` placeholder (`pyproject.toml`, `mkdocs.yml`, `README.md`,
   docs) with the real GitHub account or organisation.
4. Enable GitHub Pages with the source "GitHub Actions" to publish the docs.

## Each release

1. Update `CHANGELOG.md` and the version in `pyproject.toml` and
   `src/scientificlab/__init__.py` (a test checks they agree).
2. `ruff check . && ruff format --check . && mypy && pytest` must pass.
3. Optional dry run: run the **Release** workflow manually to publish to TestPyPI,
   then `pip install -i https://test.pypi.org/simple/ scientificlab`.
4. Tag and push: `git tag v0.7.0 && git push origin v0.6.0`. The workflow checks
   that the tag equals the package version, runs lint, types and tests, builds,
   publishes to PyPI and creates the GitHub release.
