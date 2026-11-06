#!/usr/bin/env python3
"""Collapsible source excerpts for result pages, each linked to its lines on GitHub (owner, 2026-11-06: "collapsible code
blocks (linkable to the github file line) that really directly correspond to the step in the algorithm/controls").

A block is found by name or by text, never by a typed line number, so it follows the code when lines move:

    import code_blocks as CB
    CB.block("src/sr2hand/control/hom.py", CB.find_def("src/sr2hand/control/hom.py", "HomController.eq1_rows"),
             "Eq. (1): the rows A and b")
    CB.find_span(path, "anti-windup (Sec. III-C)", before=3)       # the lines around a line that contains a text

The link names the last commit that changed the file, so its line anchors match the excerpt; a file with uncommitted
changes is linked on the branch main, and the builder prints a warning, because its lines may then differ. Excerpts
escape "$" as an HTML entity, so a page's LaTeX pass leaves them alone. Comments are set in grey through Python's
tokenizer; anything it cannot tokenize stays plain.
"""
from __future__ import annotations

import ast
import html
import io
import subprocess
import tokenize
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = "https://github.com/badinkajink/hand"
CSS = """
details.code { margin: 6px 0 12px; border: 1px solid #999; }
details.code > summary { cursor: pointer; padding: 3px 8px; font-size: 16px; background: #f2f2f2; }
details.code > summary code { font-size: 15px; }
details.code pre { margin: 0 !important; padding: 8px 10px !important; overflow-x: auto; font-size: 14px !important;
                   line-height: 1.35; background: #fbfbfb !important; }
details.code pre .ln { color: #999; user-select: none; display: inline-block; width: 3.2em; text-align: right;
                       margin-right: 0.9em; }
details.code pre .cm { color: #6a6a6a; font-style: italic; }
details.code .src { font-size: 15px; padding: 2px 8px 4px; }
"""


@lru_cache(maxsize=None)
def _commit(path: str) -> tuple[str, bool]:
    """(the last commit that changed path, whether the working tree's file equals that commit's)."""
    sha = subprocess.run(["git", "log", "-1", "--format=%H", "--", path], cwd=ROOT, capture_output=True,
                         text=True).stdout.strip()
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", path], cwd=ROOT).returncode == 0
    return sha, clean and bool(sha)


def permalink(path: str, l0: int, l1: int) -> str:
    sha, clean = _commit(path)
    if not clean:
        print(f"code_blocks: {path} differs from its last commit; linked on main", flush=True)
    ref = sha if clean else "main"
    return f"{REPO}/blob/{ref}/{path}#L{l0}" + (f"-L{l1}" if l1 > l0 else "")


@lru_cache(maxsize=None)
def _lines(path: str) -> tuple[str, ...]:
    return tuple((ROOT / path).read_text().splitlines())


def find_def(path: str, qualname: str) -> tuple[int, int]:
    """The lines (1-based, inclusive) of a function, class or method, "Class.method" for a method; decorators
    included."""
    tree = ast.parse((ROOT / path).read_text())
    parts = qualname.split(".")
    nodes = tree.body
    node = None
    for k, name in enumerate(parts):
        node = next((n for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                     and n.name == name), None)
        if node is None:
            raise KeyError(f"{qualname} not found in {path}")
        nodes = node.body
    l0 = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    return l0, node.end_lineno


def find_span(path: str, text: str, after: int = 0, before: int = 0, until: str | None = None,
              within: str | None = None) -> tuple[int, int]:
    """The lines around the first line that contains text: `before` lines above it and `after` below it, or down to the
    first later line that contains `until`. `within` names a def (find_def) to search in."""
    L = _lines(path)
    lo, hi = find_def(path, within) if within else (1, len(L))
    hits = [i for i in range(lo, hi + 1) if text in L[i - 1]]
    if not hits:
        raise KeyError(f"{text!r} not found in {path}" + (f" ({within})" if within else ""))
    i = hits[0]
    if until is not None:
        ends = [j for j in range(i, hi + 1) if until in L[j - 1]]
        if not ends:
            raise KeyError(f"{until!r} not found after {text!r} in {path}")
        return i - before, ends[0]
    return i - before, i + after


def _comments(src: str) -> dict[tuple[int, int], int]:
    """(row, col) of each comment's start -> its length, through the tokenizer; {} when the excerpt does not tokenize."""
    out = {}
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.COMMENT:
                out[tok.start] = len(tok.string)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return {}
    return out


def excerpt(path: str, l0: int, l1: int) -> str:
    """The lines l0..l1 of path as numbered HTML, comments in grey, '$' as an entity."""
    L = _lines(path)[l0 - 1:l1]
    ind = min((len(x) - len(x.lstrip()) for x in L if x.strip()), default=0)
    L = [x[ind:] for x in L]
    com = _comments("\n".join(L) + "\n")
    rows = []
    for k, line in enumerate(L, 1):
        starts = [c for (r, c) in com if r == k]
        if starts:
            c = starts[0]
            body = html.escape(line[:c]) + '<span class="cm">' + html.escape(line[c:]) + "</span>"
        else:
            body = html.escape(line)
        rows.append(f'<span class="ln">{l0 + k - 1}</span>{body}')
    return "\n".join(rows).replace("$", "&#36;")


def block(path: str, span: tuple[int, int], title: str, note: str = "", open_: bool = False) -> str:
    """A collapsed excerpt: the summary names the step and the code, the body links the lines on GitHub."""
    l0, l1 = span
    url = permalink(path, l0, l1)
    head = (f'<div class="src"><a href="{url}">{html.escape(path)}, lines {l0}&ndash;{l1}</a> on GitHub'
            + (f"; {note}" if note else "") + "</div>")
    return (f'<details class="code"{" open" if open_ else ""}><summary>{title}</summary>{head}'
            f"<pre><code>{excerpt(path, l0, l1)}</code></pre></details>")


def link(path: str, span: tuple[int, int], text: str) -> str:
    """An inline link to the lines on GitHub."""
    return f'<a href="{permalink(path, *span)}">{text}</a>'
