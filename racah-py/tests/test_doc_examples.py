"""Run every Python code block in docs/user-guide/python.md.

The Rust twin is tests/doc_examples.rs; this keeps the Python guide honest the
same way.
"""

import re
from pathlib import Path

import pytest

GUIDE = Path(__file__).resolve().parents[2] / "docs" / "user-guide" / "python.md"
BLOCKS = re.findall(r"```python\n(.*?)```", GUIDE.read_text(), re.DOTALL)


def test_guide_has_blocks():
    assert len(BLOCKS) >= 3


@pytest.mark.parametrize("i", range(len(BLOCKS)))
def test_guide_block(i):
    exec(compile(BLOCKS[i], f"{GUIDE}:block{i}", "exec"), {})
