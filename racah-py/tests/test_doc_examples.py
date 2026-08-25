"""Run every Python code block in the docs that carry runnable ones.

The Rust twin is tests/doc_examples.rs; this keeps the Python-facing prose honest
the same way.

Two files, for two different reasons. ``docs/user-guide/python.md`` is the chapter a
reader works through. ``racah-py/README.md`` is the **PyPI long description** -- the
first thing anyone installing this package reads, and the one piece of prose that is
published rather than merely committed, so a rotted quickstart there is the most
expensive stale text in the repository.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCS = {
    "guide": ROOT / "docs" / "user-guide" / "python.md",
    "readme": ROOT / "racah-py" / "README.md",
}


def blocks(path: Path) -> list[str]:
    return re.findall(r"```python\n(.*?)```", path.read_text(), re.DOTALL)


CASES = [(name, i) for name, path in DOCS.items() for i in range(len(blocks(path)))]


@pytest.mark.parametrize("name", sorted(DOCS))
def test_the_document_has_blocks(name):
    """A regex that silently matched nothing would make the run below vacuous."""
    assert len(blocks(DOCS[name])) >= 2


@pytest.mark.parametrize(("name", "i"), CASES, ids=lambda x: str(x))
def test_block_runs(name, i):
    path = DOCS[name]
    exec(compile(blocks(path)[i], f"{path}:block{i}", "exec"), {})  # noqa: S102
