# Progress — Origami x T-Rex redesign

Tracks REDESIGN_PLAN.md §12 steps and G0-G18 gates. Updated as work proceeds; this is the
handoff artifact if context is compacted.

## §12 build order

| step | deliverable | status | gates |
|---|---|---|---|
| 1 | Repo hygiene: submodule pinned + full-pipeline fetched, deletions, .gitignore, pyproject.toml | **done** | G0 pass |
| 2 | test_lerobot_probe.py + test_processor_probe.py | **done** | §11.4 resolved; §4.5-A size chosen (384 384) |
| 3 | constants.py + splits.json + kinematics.py | **done** | G1a, G1b, G2, G2b, G3, G3b pass; G18 list-parsing pass |
| 4 | decode.py | **done (code)** | G5/G6 not yet tested against real data |
| 5 | stats.py | **done** | G8, G8b pass (see notes -- G8b's exact-finger claim doesn't reproduce on this fixture) |
| 6 | convert.py | **done** | smoke-tested against real fixture season (1 episode full-length + truncated unit tests); G4/G4b, G6 pass in `test_convert.py` |
| 7 | fetch.py + merge.py + prepare.py | **done** (see notes) | - |
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
| G1a | URDF joint set == 65 mapped names | **PASS** | exact set match on `pin.buildModelFromUrdf` |
| G1b | Reduced model nq==nv==14 | **PASS** | `nq==nv==14`, names == the 14 arm joints |
| G1c | LockedConfig.digest() consistency | not started | code exists (`LockedConfig.digest`), cross-artifact check deferred to prepare.py/serve_zenoh.py (step 7/13) |
| G2 / G2b | FK/IK pose round-trip, 10 000 samples each | **PASS (redefined -- see note below)** | pos err max ~1e-5 m, rot err max ~1e-4 deg (both ≪ thresholds 1e-4m/0.01deg); max\|dq\| bounded but NOT gated at 1e-3 rad -- see note |
| G3 | delta9→rot6d_to_matrix reconstructs target pose, 2000 samples | **PASS** | max err well under 1e-9 |
| G3b | rot6d round-trip on 10 000 random SO(3) | **PASS** | max err < 1e-12 |
| G4 / G4b | Chunk parity | **PASS** | `test_g4_chunk_reconstructs_abs_target` (test_convert.py): chunk-base delta9 (k=0) + `observation.state` FK pose reconstructs `action_abs` FK pose to <1e-4 on real fixture frames |
| G5 | Deform tile mapping | not started | decode.py written, not yet run against real data with known-force frames (only the round-trip below (G6) has been checked so far) |
| G6 | Deform video round-trip | **PASS** | `test_deform_video_round_trips_losslessly`: source luma tile == decoded 3-ch-replicated tile, exact (`assert_array_equal`), on real fixture data |
| G7 / G7b | Loader parity / forward pass | not started | - |
| G8 / G8b | Reservoir stats / tactile mask | **PASS (G8b redefined -- see note)** | G8: q01/q99 rel err < 1% (reservoir exact at this scale); G8b: masking mechanism verified deterministically; on THIS fixture only 1/60 tactile dims cross the 1e-3 span threshold (not the full ring/pinky blocks the plan's original fixture showed) |
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

## Step 6 findings (convert.py smoke test) — two real bugs found and fixed

* **This session's environment had no venv packages or fixture data installed** (both are
  gitignored and this was a fresh checkout) — `uv sync --extra dev` and a fresh
  `download_season(...)` were required before anything could run. HF_TOKEN provided by the
  user, stored in `.env` (gitignored), not committed.
* **Bug in `fetch.py::download_season`/`download_season_meta_and_data` (never previously
  exercised — added in the same WIP commit as convert.py, untested): 0 files ever
  downloaded.** `snapshot_download`'s `allow_patterns` match full repo-relative paths
  (including the `season_.../` prefix), and `local_dir` mirrors that same full repo-relative
  layout. The code stripped the season prefix from the patterns while still setting
  `local_dir=cache_root/season` — patterns and `local_dir` disagreed, so nothing matched.
  Fixed: `local_dir=cache_root`, patterns keep the full season-prefixed path; return value
  (`cache_root/season`) unchanged so callers (`have_season`, `convert_season`) are unaffected.
* **Bug in `decode.py::decode_episode_stream`: OOM-killed (exit 137) on a real 2-episode
  smoke run on a 15GB machine.** The function was a generator only in name — internally it
  called `_decode_from_seek(...)`, which fully decoded and materialized *every* frame of an
  episode into a Python list *before* the first `yield`. For a ~5000-frame, 480x480 episode
  across 4 concurrently-open video streams (head/wrist_left/wrist_right/deform), that's
  10+GB of ndarrays held at once — the same anti-pattern `stats.py`'s `StreamingNormStats`
  was explicitly built to avoid (§5.4), just in the video path instead of the stats path.
  Fixed: rewrote as true frame-by-frame streaming (`_stream_from_seek`/`_stream_linear_scan`
  generators, one frame in flight at a time). The seek-vs-linear-scan decision now uses the
  already-existing cheap `probe_available_frames` (counts frames without ever calling
  `to_ndarray`) *before* any pixel data is decoded, preserving the original
  never-hand-the-caller-a-short/duplicate-sequence guarantee without buffering.
  **After this fix, 1 real full-length episode (7924 frames, all 4 video streams, both
  writer shards) converts successfully; a 2-episode run was not re-tried (not needed — the
  fix is a strict memory-bound improvement, and per-episode processing is exactly what
  `merge.py`/`prepare.py` (§5.2/§5.5) already assume).**
* Added `convert_season(..., max_frames_per_episode=...)` (test/debug only, mirrors the
  existing `max_episodes`) so `test_convert.py` can run the *real* pipeline (real FK, real
  PyAV decode/encode, real `LeRobotDataset` writer, real lossless-deform round-trip) in ~28s
  instead of minutes, without weakening what's being tested.
* `test_convert.py` (5 tests, all pass): writer produces exactly N frames; the shard is
  readable by the real `lerobot.datasets.lerobot_dataset.LeRobotDataset` with the exact
  T-Rex-contract shapes (§1.4); `trex_norm_stats.json` is written with the right shapes;
  **G4/G4b** now pass (chunk-base delta9 reconstructs `action_abs`'s FK pose to <1e-4 m/rot6d
  on real fixture frames); **G6** now passes (deform video round-trips losslessly, exact
  match, real fixture data).
* Full `origami/tests/` suite: **36/36 pass**, ~72s total.

## Step 7 findings (merge.py) — §5.3's first-choice API doesn't work for our schema

* **`lerobot.datasets.aggregate.aggregate_datasets` exists in the pinned version (confirmed
  step 2) but cannot merge our shards.** `aggregate_data` always rewrites each data parquet
  via `to_parquet_one_row_group_per_episode(df, dst_path)`, which does
  `pa.Table.from_pandas(df, preserve_index=False)` with **no explicit schema** -- confirmed
  by reading the installed library's source, then reproducing the failure directly on two
  real converted shards: `pyarrow.lib.ArrowTypeError: Conversion failed for column action
  with type array[float32]`. Our `action` feature is `[16,62]` per row (§1.4); the
  pandas-to-pyarrow round trip loses that fixed-shape-array typing regardless of
  ``contains_images`` (the only place a schema is passed through is the images path, which
  our video features don't hit either). This is a real limitation of the pinned lerobot
  version for any multi-dimensional array feature, not a bug in our code.
* **Resolution: implemented the plan's own explicit fallback** (§5.3: "otherwise
  `merge_shards(...)`") as a genuine copy-and-renumber, made simple by §5.2's
  `ONE_EPISODE_PER_FILE_MB` writer setting (every source data/video file holds exactly one
  episode): read each data parquet with **`pyarrow` directly** (never through pandas, so the
  2D array column is never re-inferred), patch only the `index`/`episode_index` int columns,
  byte-copy video files untouched, and rewrite `meta/episodes` + `meta/info.json` with
  renumbered indices. `merge_shards` also merges `trex_norm_stats.pkl` (added
  `acc.dump(...)` to `convert_season`, needed for this -- previously only `acc.write()` the
  assembled JSON) and a new `meta/origami_prep.json` (per-season truncation log +
  `LockedConfig.digest()`, also newly written by `convert_season`), refusing to merge shards
  converted under different `LockedConfig` digests (§3.2).
* `test_merge.py` (4 tests, all pass, ~47s): builds 2 real truncated shards from the fixture
  season and merges them, checking global episode/frame renumbering at the shard boundary,
  merged `trex_norm_stats.json` transition/trajectory counts, merged `origami_prep.json`, and
  that mismatched `LockedConfig` digests raise instead of silently merging.
* Full `origami/tests/` suite: **40/40 pass**, ~116s total.

## Step 7 findings (prepare.py) — CLI orchestrator

* `prepare.py` implements §5.5's phase 0 (meta+data-only fetch per season -> exact frame
  total + `LockedConfig`/arm-medians, §3.3) / phase 1 (per-season download+convert+drop,
  resumable via `DONE` markers + `manifest.jsonl`) / phase 2 (`merge_shards`), plus G18
  season-list resolution (`resolve_season_list`) and the §5.6 disk-budget refusal
  (`check_disk_budget`) and §5.5 config-invalidation refusal (`PrepConfig` recorded in/compared
  against `meta/origami_prep.json`).
  **Simplification, documented in the module docstring:** §5.5 describes a
  `ProcessPoolExecutor` over seasons *plus* a separate `ThreadPoolExecutor` prefetching
  downloads so at most `--disk-budget` seasons are resident at once. Implemented instead as
  one `ProcessPoolExecutor` with a `multiprocessing.Manager().Semaphore(disk_budget)` acquired
  around each worker's download+convert+drop -- same resource bound (never more than
  `disk-budget` seasons' raw video on disk simultaneously), simpler to get right, at the cost
  of not overlapping "downloading season N+1" with "converting season N". Revisit only if the
  real prep server turns out to be download-bound.
* **Not yet exercised for real:** the `ProcessPoolExecutor`/semaphore plumbing in
  `run_phase1` needs real network + real per-season conversion to test meaningfully (a worker
  function pickled into a child process can't be monkeypatched from the parent test process),
  so it's only covered by the "everything already DONE, skip" resumability path in
  `test_prepare.py`. The season-by-season pipeline it calls (`download_season`,
  `convert_season`, `merge_shards`) is independently real-data-tested (step 6/7 above) --
  what's untested here specifically is the concurrency glue itself. Real coverage comes from
  step 10 (the actual full prep run, user-executed on the large-RAM/large-disk server, §0 D3).
* `test_prepare.py` (8 tests, all pass, ~3s, no network): `PrepConfig.as_dict` stability; G18
  `resolve_season_list` (happy path, hub-missing-season assertion, fixture-in-split
  assertion, extras-fold-into-train-only); `check_disk_budget` pass/fail; phase-1
  all-already-done resumability skip. `phase0_locked_config` additionally spot-checked
  manually (not as a committed test, since it needs a monkeypatched `download_season_meta_and_data`
  writing synthetic parquet) against synthetic 65-D state data — frame totals and
  `LockedConfig`/arm-default shapes came out correct.
* Full `origami/tests/` suite: **48/48 pass**, ~115s total.

## Environment

* `HF_TOKEN` lives in `.env` (gitignored), not in any tracked file.
* venv: `.venv` (uv, Python 3.12). Activate or use `.venv/bin/python` / `uv run`.
* Local test fixture: `season_POC22061_2026_05_23_19_21_25_train/lerobot3.0/` at repo root
  (gitignored, 2.8 GB) — see the note above under step 1 for why this replaces the
  plan-cited season name.

## Step 3 findings (kinematics) — G2 redefined, user-approved

**Confirmed empirically (not a guess): the verbatim `ik_utils.py::PinkLocalIK.solve_ik`
algorithm (§1.4/§3.1's required transcription for deploy) cannot meet G2's literal
`max|Δq| < 1e-3 rad` threshold on this robot, for a structural reason, not a bug.**
Our arm is 7-DOF solving a 6-DOF pose task (a genuine 1-DOF self-motion null space). Measured:
* Even from a **perfect** warm start (zero injected noise), `solve_ik` converges to a
  steady-state joint offset of ~0.03-0.07 rad / up to ~2° orientation error that does **not**
  shrink with more iterations (tested up to 200) — a real equilibrium of the task weights
  (`position_cost=50` vs `orientation_cost=1.0` vs the posture-regularization task pulling
  toward a fixed default), not an under-convergence artifact.
* Retuning the smoothness-task cost across 1..200 while removing the competing regularization
  term left `max|Δq|` essentially unchanged (~0.01-0.03 rad for warm noise std=0.02) while
  driving pose error to ~1e-9 deg / ~1e-8 m — i.e. **any** correct IK solver returns the
  closest point on the (curved) 6-DOF solution manifold to the noisy 7-DOF warm start, and the
  component of injected noise along the null space is mathematically uncorrectable by
  construction. This is a property of genuine kinematic redundancy + the test's own method of
  injecting noise on all 7 raw joints, not something any IK tuning can fix.
* A follow-up attempt to project the injected noise onto the null space's orthogonal
  complement (via per-config Jacobian SVD) did **not** cleanly fix it either — deprioritized
  further debugging of that approach once the core finding (structural, not tunable) was
  confirmed via the direct sweep.

**Resolution (user chose "add a tight IK variant for G2/G3 only" via AskUserQuestion):** added
`OrigamiKinematics.solve_ik_tight()` — high pose costs (200/200), light smoothness, no
competing regularization, 15 iterations — used ONLY by `test_kinematics.py`'s G2/G2b/G3.
`solve_ik()` (deploy path, used by `retarget.py`/`serve_zenoh.py`) is untouched, still verbatim.
G2/G2b now gate **pose reconstruction** (position < 1e-4 m, rotation < 0.01°, the plan's own
thresholds — met with large margin) and **report but do not hard-gate** `max|Δq|` (asserted
only to be `< 8x` the injected warm-noise std, ruling out IK *amplifying* the noise, which is
the only thing meaningfully testable given the null-space argument above). Full detail and the
sweep data are worth re-reading before touching `kinematics.py::solve_ik*` again.

**G8b similarly redefined:** the plan's example ("masked set includes L-ring, L-pinky, R-ring,
R-pinky") was measured on a *different* fixture season we don't have (the plan-named season
doesn't exist — see the step-1 note). On our substituted fixture, full-season tactile data
shows the same qualitative pattern (thumb/index mean |force| 2.5-4N vs middle/ring/pinky
0.02-0.17N) but per-channel `q99-q01` span with `eps=1e-3` masks only 1/60 dims — sensor noise
floor on the quiet fingers exceeds the span threshold even though their mean force is near
zero. `test_g8b_masking_mechanism_detects_degenerate_channels` proves the masking *logic*
works (deterministic flat-channel injection); `test_g8b_fixture_tactile_mask_on_real_data`
checks the direction that does hold (quiet fingers ≪ thumb/index in force) rather than the
exact finger-block claim, which is season-dependent.
