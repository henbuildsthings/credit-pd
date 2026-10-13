"""Count lines of Python in src/ and tests/: code-only and total. Run: `make loc`.

The project's size budget (~800 lines, see CLAUDE.md) applies to code lines only, so
documentation never competes with code for space. A line counts as code if it holds
any token other than a comment or a docstring; blank lines don't count either.
"""

import ast
import io
import tokenize
from pathlib import Path

# Token types that never make a line count as code.
NON_CODE = {
    tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE,
    tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER,
}  # fmt: skip


def count(path: Path) -> tuple[int, int]:
    """Return (code lines, total lines) for one Python file."""
    src = path.read_text()
    # Line numbers covered by module/class/function docstrings.
    doc = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            if ast.get_docstring(node) is not None:
                expr = node.body[0]
                doc.update(range(expr.lineno, expr.end_lineno + 1))
    tokens = tokenize.generate_tokens(io.StringIO(src).readline)
    code = {t.start[0] for t in tokens if t.type not in NON_CODE} - doc
    return len(code), len(src.splitlines())


if __name__ == "__main__":
    files = sorted(p for d in ("src", "tests") for p in Path(d).rglob("*.py"))
    totals = [0, 0]
    for f in files:
        code, total = count(f)
        totals[0] += code
        totals[1] += total
        print(f"{code:5} {total:6}  {f}")
    print(f"{totals[0]:5} {totals[1]:6}  total (code / all lines; budget ~800 code)")
