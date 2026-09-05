"""§7.2 -- origami/train_origami.py is a vendored copy of
`T-Rex@full-pipeline:scripts/midtrain.py` with a small, precisely marked delta set.

CPU-only, network-free: `full-pipeline` was already fetched into the T-Rex submodule during
step 1 (see origami/PROGRESS.md), so `git show origin/full-pipeline:...` is a local ref
lookup, not a network call.
"""
from __future__ import annotations

import difflib
import subprocess
from pathlib import Path

import pytest

from origami.constants import REPO_ROOT

TREX_ROOT = REPO_ROOT / "T-Rex"
VENDORED_PATH = REPO_ROOT / "origami" / "train_origami.py"
B23EAFE = "b23eafe564a1457cd4eacb889aaf6fbf29a29034"

MARKER = "ORIGAMI-DELTA"
CONTEXT_LINES = 3  # how many surrounding lines of an inserted/replaced hunk we search


def _git_show(ref: str, path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        cwd=TREX_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    return result.stdout


def _upstream_lines() -> list[str]:
    return _git_show(B23EAFE, "scripts/midtrain.py").splitlines(keepends=True)


def _vendored_lines() -> list[str]:
    return VENDORED_PATH.read_text().splitlines(keepends=True)


def test_vendored_file_exists_and_is_nontrivially_different():
    upstream = _upstream_lines()
    vendored = _vendored_lines()
    assert vendored != upstream
    assert len(vendored) > 0.9 * len(upstream)  # a vendor+small-delta, not a rewrite


def test_every_changed_hunk_carries_origami_delta_marker():
    upstream = _upstream_lines()
    vendored = _vendored_lines()
    sm = difflib.SequenceMatcher(a=upstream, b=vendored, autojunk=False)

    unmarked_hunks = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        # Search a window around the changed region in the VENDORED file (insertions and
        # replacements land there) for the marker. For pure deletions (tag == "delete",
        # j1 == j2), there's nothing to mark in the vendored file by construction -- check
        # the marker is present just before/after the deletion point instead.
        lo = max(0, j1 - CONTEXT_LINES)
        hi = min(len(vendored), j2 + CONTEXT_LINES)
        window = "".join(vendored[lo:hi])
        if MARKER not in window:
            unmarked_hunks.append({
                "tag": tag,
                "upstream_lines": "".join(upstream[i1:i2]),
                "vendored_lines": "".join(vendored[j1:j2]),
            })

    assert not unmarked_hunks, (
        f"{len(unmarked_hunks)} changed hunk(s) vs. full-pipeline@{B23EAFE}'s "
        f"scripts/midtrain.py have no nearby `# {MARKER}:` marker:\n"
        + "\n---\n".join(
            f"upstream:\n{h['upstream_lines']}\nvendored:\n{h['vendored_lines']}"
            for h in unmarked_hunks
        )
    )


def test_vendored_file_is_valid_python():
    import ast
    ast.parse(VENDORED_PATH.read_text())
