"""Gate G0 — upstream drift.

Implements REDESIGN_PLAN.md §2.1 / §10 (G0) / §13.

T-Rex/ must stay byte-identical to upstream: pinned at f88e10c with zero local diff
(REDESIGN_PLAN.md D2). This test hashes every upstream file origami/ imports or vendors,
on both pinned refs (f88e10c for direct imports/copies, b23eafe for the full-pipeline
vendor source of train_origami.py), and fails if any of them differ from the frozen
manifest recorded at design time -- whether because the submodule pin moved, the
working tree was edited, or a ref was force-pushed upstream.
"""

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TREX_ROOT = REPO_ROOT / "T-Rex"
MANIFEST_PATH = REPO_ROOT / "origami" / "upstream_manifest.json"

F88E10C = "f88e10c61da123c68bf0927cf4860bc97a0381f3"
B23EAFE = "b23eafe564a1457cd4eacb889aaf6fbf29a29034"


def _load_manifest() -> dict:
    with open(MANIFEST_PATH) as fh:
        return json.load(fh)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_show(ref: str, path: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        cwd=TREX_ROOT,
        capture_output=True,
        check=True,
    )
    return result.stdout


def _submodule_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=TREX_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    return result.stdout.strip()


MANIFEST = _load_manifest()


def test_manifest_covers_both_pinned_refs():
    assert set(MANIFEST.keys()) == {F88E10C, B23EAFE}


def test_submodule_pinned_at_f88e10c():
    assert _submodule_head() == F88E10C, (
        "T-Rex submodule HEAD does not match the pinned commit f88e10c61da123c68bf0927cf4"
        "860bc97a0381f3 -- REDESIGN_PLAN.md D2 requires zero drift."
    )


def test_submodule_has_zero_local_diff():
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=TREX_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    assert result.stdout == "", f"T-Rex/ has local modifications:\n{result.stdout}"


@pytest.mark.parametrize("path", sorted(MANIFEST[F88E10C].keys()))
def test_f88e10c_file_hash_unchanged(path):
    expected = MANIFEST[F88E10C][path]
    working_tree_bytes = (TREX_ROOT / path).read_bytes()
    assert _sha256_bytes(working_tree_bytes) == expected, (
        f"{path} at f88e10c has drifted from the recorded manifest hash."
    )


@pytest.mark.parametrize("path", sorted(MANIFEST[B23EAFE].keys()))
def test_b23eafe_file_hash_unchanged(path):
    expected = MANIFEST[B23EAFE][path]
    blob_bytes = _git_show(B23EAFE, path)
    assert _sha256_bytes(blob_bytes) == expected, (
        f"{path} at full-pipeline@b23eafe has drifted from the recorded manifest hash."
    )
