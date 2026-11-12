"""Every ``python`` code block in the documentation must run.

Run from a temporary directory because some examples save a figure.
"""

import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[1] / "docs"
BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)

EXAMPLES = [
    pytest.param(block, id=f"{path.relative_to(DOCS)}#{index}")
    for path in sorted(DOCS.rglob("*.md"))
    for index, block in enumerate(BLOCK.findall(path.read_text(encoding="utf-8")))
]


def test_docs_contain_examples():
    assert len(EXAMPLES) >= 10


@pytest.mark.parametrize("code", EXAMPLES)
def test_doc_example_runs(code, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    exec(compile(code, "<docs>", "exec"), {})
