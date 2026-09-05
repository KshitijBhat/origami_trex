# Progress — Origami x T-Rex redesign

Tracks REDESIGN_PLAN.md §12 steps and G0-G18 gates. Updated as work proceeds; this is the
handoff artifact if context is compacted.

## §12 build order

| step | deliverable | status | gates |
|---|---|---|---|
| 1 | Repo hygiene: submodule pinned + full-pipeline fetched, deletions, .gitignore, pyproject.toml | **done** | G0 pass |
| 2 | test_lerobot_probe.py + test_processor_probe.py | not started | - |
| 3 | constants.py + splits.json + kinematics.py | not started | - |
| 4 | decode.py | not started | - |
| 5 | stats.py | not started | - |
| 6 | convert.py | not started | - |
| 7 | fetch.py + merge.py + prepare.py | not started | - |
| 8 | verify.py | not started | - |
| 9 | trex_patch.py + train_origami.py + delayed_lerobot_dataset.py + .sh | not started | - |
| 10 | Full prep run (user-executed) | not started | - |
| 10b | diagnose_shift.py + eval_offline.py --zero-shot | not started | - |
| 10c | refit_vqvae_stats.py (if needed) | not started | - |
| 11 | Pilot train 2000 steps (user-executed) | not started | - |
| 11b | Optional phase 0 / phase A ablations | not started | - |
| 12 | Phase B: 40000 steps (user-executed) | not started | - |
| 13 | policy.py + retarget.py + serve_zenoh.py + bench_latency.py | not started | - |
| 14 | eval_offline.py §8.1-B..F + eval_shadow.py | not started | - |
| 15 | docker/ submission image | not started | - |

## Gates

| id | gate | status | measured |
|---|---|---|---|
| G0 | Upstream drift: sha256 of every T-Rex file we import/vendor, both refs | **PASS** | 11/11 checks pass (submodule pinned f88e10c, zero local diff, 7 files hashed at f88e10c, scripts/midtrain.py hashed at b23eafe) |
| G1a | URDF joint set == 65 mapped names | not started | - |
| G1b | Reduced model nq==nv==14 | not started | - |
| G1c | LockedConfig.digest() consistency | not started | - |
| G2 / G2b | FK/IK round-trip | not started | - |
| G3 / G3b | delta9/rot6d round-trip | not started | - |
| G4 / G4b | Chunk parity | not started | - |
| G5 | Deform tile mapping | not started | - |
| G6 | Deform video round-trip | not started | - |
| G7 / G7b | Loader parity / forward pass | not started | - |
| G8 / G8b | Reservoir stats / tactile mask | not started | - |
| G9a / G9b | Smoke train / resume | not started | - |
| G10 | serve_zenoh SDK checks | not started | - |
| G11 | Shadow replay | not started | - |
| G12 | Prep<->deploy parity | not started | - |
| G13 | Latency | not started | - |
| G14 | Instruction identity | not started | - |
| G15 | VQ-VAE codebook health | not started | - |
| G16 | Token budget | not started | - |
| G17 | Delay-curriculum parity | not started | - |
| G18 | Season-set integrity | not started | - |

## Notes / surprises

* **T-Rex was not yet a submodule.** It was tracked as ~1200 plain files in the main repo
  with zero local diff. Converted per §2.1: `git rm -r --cached T-Rex && rm -rf T-Rex`,
  `git submodule add https://github.com/ZhuoyangLiu2005/T-Rex.git T-Rex`, checked out at
  `f88e10c` (already HEAD of the upstream default branch), fetched `origin/full-pipeline`
  (`b23eafe`) for the step-9 vendoring source. Both SHAs match the plan's citations exactly.
* **PyPI package name trap: `pink` != Pink IK.** `pip/uv install pink` installs an unrelated
  code-formatting tool (`chenjiandongx/pink`) whose `pink.py` shadows the real
  Stephane-Caron/Pink inverse-kinematics library needed by REDESIGN_PLAN.md §1.4/§3.1. The
  correct PyPI distribution is **`pin-pink`** (imports as `pink`, confirmed
  `from pink.tasks import FrameTask, PostureTask` works). Fixed in `pyproject.toml`.
  `daqp` solver confirmed available via `qpsolvers.available_solvers`.
* **Secret found in working tree, not committed.** The uncommitted diff on
  REDESIGN_PLAN.md (before this session touched it) added a live `HF_TOKEN` value inline
  in the markdown. Moved to `.env` (already gitignored) instead of committing a credential
  into tracked file history; reverted the plan text to match the committed version.
* **Fixture season is absent from this instance.** REDESIGN_PLAN.md and the task prompt both
  describe `season_POC22061_2026_07_23_10_20_33_train/lerobot3.0/` as already present locally
  ("the only local season"), but it does not exist anywhere on this instance (checked
  `/workspace`, `$HOME`, common mount points, `.hf_home`). This blocks step 2's
  `test_lerobot_probe.py`/`test_processor_probe.py` and every later gate that exercises real
  data (G2, G4-G8, G12, etc.), which all depend on it. **Needs to be supplied before step 2
  can run for real** — flagged to the user; not yet resolved.
* Local venv created at `.venv` with Python 3.12 (lerobot requires >=3.12); `uv.lock`
  committed. `origami/__init__.py` puts `T-Rex/` on `sys.path`; verified
  `import utils.lerobot_common` resolves `ACTION_DIM=62`, `ACTION_CHUNK=16`.

## Environment

* `HF_TOKEN` lives in `.env` (gitignored), not in any tracked file.
* venv: `.venv` (uv, Python 3.12). Activate or use `.venv/bin/python` / `uv run`.
