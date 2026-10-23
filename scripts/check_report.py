"""Verify that the technical report stays in sync with the code and the result files.

1. Every fenced code block preceded by ``<!-- src: path/to/file.py -->`` must appear in that file
   (compared line by line after stripping trailing whitespace; ``# ...`` elision lines in the
   snippet split it into chunks that must each appear, in order).
2. Every ``<!-- num: file.json:dotted.key[:fmt] -->VALUE`` marker must match the value stored in
   that file (``fmt`` is a Python format spec such as ``.1%`` or ``.2f``).

Usage:
    python scripts/check_report.py [docs/technical_report.md]
Exits with status 1 and a list of problems if anything is out of sync.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC_RE = re.compile(r"<!-- src: (?P<path>[^ ]+) -->\s*```[a-z]*\n(?P<code>.*?)```", re.S)
NUM_RE = re.compile(r"<!-- num: (?P<file>[^:]+):(?P<key>[^: ]+)(?::(?P<fmt>[^ ]+))? -->(?P<val>[^\s|,;)]+)")


def chunks(code: str) -> list[list[str]]:
    out, cur = [], []
    for line in code.rstrip("\n").split("\n"):
        if line.strip() in ("# ...", "..."):
            if cur:
                out.append(cur)
            cur = []
        else:
            cur.append(line.rstrip())
    if cur:
        out.append(cur)
    return out


def contains(lines: list[str], chunk: list[str], start: int) -> int:
    n = len(chunk)
    for i in range(start, len(lines) - n + 1):
        if lines[i : i + n] == chunk:
            return i + n
    return -1


def lookup(data, key: str):
    for part in key.split("."):
        data = data[int(part)] if isinstance(data, list) else data[part]
    return data


def main() -> int:
    report = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "docs" / "technical_report.md"
    text = report.read_text(encoding="utf-8")
    problems: list[str] = []
    n_src = n_num = 0
    for m in SRC_RE.finditer(text):
        n_src += 1
        path = REPO / m["path"]
        if not path.exists():
            problems.append(f"snippet source missing: {m['path']}")
            continue
        lines = [ln.rstrip() for ln in path.read_text(encoding="utf-8").split("\n")]
        pos = 0
        for ch in chunks(m["code"]):
            pos = contains(lines, ch, pos)
            if pos < 0:
                problems.append(f"snippet not found in {m['path']}: {ch[0][:70]!r} ...")
                break
    for m in NUM_RE.finditer(text):
        n_num += 1
        f = REPO / m["file"]
        try:
            v = lookup(json.loads(f.read_text(encoding="utf-8")), m["key"])
        except (OSError, KeyError, IndexError, ValueError) as exc:
            problems.append(f"number source error {m['file']}:{m['key']}: {exc}")
            continue
        expect = format(v, m["fmt"]) if m["fmt"] else str(v)
        if expect != m["val"]:
            problems.append(f"number mismatch {m['file']}:{m['key']}: report {m['val']} vs file {expect}")
    print(f"checked {n_src} code snippets and {n_num} numbers")
    for p in problems:
        print("PROBLEM:", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
