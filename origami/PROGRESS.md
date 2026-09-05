# Progress — Origami x T-Rex redesign

Tracks REDESIGN_PLAN.md §12 steps and G0-G18 gates. Updated as work proceeds; this is the
handoff artifact if context is compacted.

## §12 build order

| step | deliverable | status | gates |
|---|---|---|---|
| 1 | Repo hygiene: submodule pinned + full-pipeline fetched, deletions, .gitignore, pyproject.toml | **done** | G0 pass |
| 2 | test_lerobot_probe.py + test_processor_probe.py | **done** | §11.4 resolved; §4.5-A size chosen (384 384) |
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
* **Fixture season resolved — but not the one the plan names.**
  `season_POC22061_2026_07_23_10_20_33_train` does not exist on the hub at all (confirmed via
  `HfApi().list_repo_tree`: `POC22061` seasons on the hub run 2026-05-18 -> 2026-07-18, nothing
  past that). This matches REDESIGN_PLAN.md §1.1a's own claim that the fixture is
  hub-absent by design, so it was never fetchable under that name. Per user instruction,
  substituted a genuine hub season **outside both split lists** (same held-out role as the
  plan's fixture) as the local test fixture:
  **`season_POC22061_2026_05_23_19_21_25_train`** — 15 episodes, 95 347 frames, confirmed
  `codebase_version: v3.0`, schema matches §1.1 exactly (`observation.state.tcp` all-zero
  verified). Downloaded `meta/**`, `data/**`, and the 4 needed video keys
  (`head_left`, `wrist_left`, `wrist_right`, `tactile_deform`; `head_right`/`tactile_raw`
  skipped per §5.1 defaults) to `season_POC22061_2026_05_23_19_21_25_train/lerobot3.0/` at repo
  root, 2.8 GB. Added `/season_*_train/` to `.gitignore`.
  **All references in tests/docs to "the fixture season" now mean this season, not the
  plan-cited name.**
* **Bug found in the hub's 17 "extras" seasons (outside the 101/25 split) — relevant to
  `fetch.py`'s `have_season` (§5.1).** Several extras have incomplete `lerobot3.0/` uploads:
  `meta/info.json`/`modality.json`/`stats.json` present but **no `data/` or `videos/` at
  all** (confirmed for `season_POC22061_2026_07_07_14_36_10_train` and
  `season_POC22061_2026_07_13_20_18_42_train`; likely others among the July-dated extras).
  Others (`2026_05_23_19_21_25`, `2026_05_24_16_27_19`) are fully backfilled and have a
  `.backfill_complete` marker file at the season root — **`have_season`/`download_season`
  should check for `.backfill_complete` before trusting a season's data/video tree**, not just
  glob for any `*.parquet`/`*.mp4`. This is separate from and in addition to interrupted
  local downloads (§5.1's original concern).
* Local venv created at `.venv` with Python 3.12 (lerobot requires >=3.12); `uv.lock`
  committed. `origami/__init__.py` puts `T-Rex/` on `sys.path`; verified
  `import utils.lerobot_common` resolves `ACTION_DIM=62`, `ACTION_CHUNK=16`.

## Step 2 findings (probes)

* **§11.4 resolved.** `test_lerobot_probe.py`: `LeRobotDataset.create/add_frame/save_episode/
  finalize` all work as the plan expects. Non-`observation.images.*` keys (the 10
  `observation.tactile_deform.*` deform keys) ARE accepted as `dtype: "video"` — no
  key-name pattern is enforced anywhere in `lerobot.datasets.{feature_utils,
  dataset_writer,dataset_metadata}`, only the `dtype` field matters. So §5.3's hand-rolled
  merger is unnecessary: **`lerobot.datasets.aggregate.aggregate_datasets(repo_ids,
  aggr_repo_id, roots=..., aggr_root=..., ...)` exists in the pinned version and
  `origami/merge.py` will use it directly.**
  Two API quirks worth recording for `convert.py`/`delayed_lerobot_dataset.py`: (1) a
  **single-offset `delta_timestamps` entry returns a tensor without the leading temporal
  dimension** (squeezed to the plain per-frame shape), unlike multi-offset entries which
  keep `[T, ...]`; (2) the pinned lerobot version's **default video codec is AV1** (SVT-AV1
  encoder), not h264 — §5.2's `--deform_codec` default (`lossless_h264`) must be passed
  explicitly at `LeRobotDataset.create()` time via the encoder config, not assumed as the
  library default.
* **§4.5-A resolved: `--image_size 384 384`.** No dedicated `qwen3_vl` image-processor
  module exists in the installed transformers version; `AutoImageProcessor`'s
  `IMAGE_PROCESSOR_MAPPING_NAMES["qwen3_vl"]` resolves to `Qwen2VLImageProcessor`
  (confirmed). Class defaults: `Qwen2VLImageProcessor.{patch_size,merge_size} = (14, 2)` ->
  factor 28; `Qwen3VLVisionConfig.{patch_size,spatial_merge_size} = (16, 2)` -> factor 32.
  Both match the plan's table exactly. Token counts computed via the real `smart_resize`
  reproduce the plan's worked numbers exactly (upstream `384 288`: 108 @ f32 / 140 @ f28;
  `384 384`: 144 @ f32 / 196 @ f28; `336 336`: 100 @ f32 / 144 @ f28).
  **Selection-rule note:** plain nearest-by-absolute-token-count picks `336 336` at *both*
  factors (100 vs 108, and 144 vs 140) — it does not reproduce the plan's stated picks
  ("384 384 at f=32, 336 336 at f=28"). The rule that does reproduce them is *smallest
  square candidate with token count >= upstream's* (never run the frozen midtrain ViT
  below the token budget it was trained at; overshooting is safe, undershooting isn't).
  Recorded that rule in `test_processor_probe.py`'s docstring since the plan states the
  conclusion but not this exact tie-break. **Until a real checkpoint's
  `preprocessor_config.json` is read (G16, at actual training time), we don't know which
  factor is real** — `384 384` is the working choice for `train_origami.sh` per §7.3, valid
  under f=32 (the vision model's own default); if the checkpoint turns out to use f=28,
  revisit to `336 336` before the real 40k run.
* **Dependency conflict, noted not fought:** `lerobot`'s `huggingface-hub>=1.6.0`
  requirement is incompatible with `transformers==4.57.*` (`huggingface-hub<1.0` cap), and
  there is no transformers release in `[4.58, 5.0)` at all (the project jumped straight to
  `5.0.0`). Locally resolved to `transformers==5.16.1` for dev/test — this does **not**
  match REDESIGN_PLAN.md §9.5's Dockerfile pin of `transformers 4.57.3`, but that's fine:
  the deploy Docker image (step 15) has no `lerobot` dependency at all, so it can pin
  `4.57.3` in isolation without this conflict. `pyproject.toml` states `transformers>=4.53.0`
  (T-Rex's own floor) with no upper cap for the dev venv.

## Environment

* `HF_TOKEN` lives in `.env` (gitignored), not in any tracked file.
* venv: `.venv` (uv, Python 3.12). Activate or use `.venv/bin/python` / `uv run`.
* Local test fixture: `season_POC22061_2026_05_23_19_21_25_train/lerobot3.0/` at repo root
  (gitignored, 2.8 GB) — see the note above under step 1 for why this replaces the
  plan-cited season name.
