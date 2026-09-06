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
| 8 | verify.py | **done** (scoped to gates checkable now; see notes) | - |
| 9 | trex_patch.py + train_origami.py + delayed_lerobot_dataset.py + .sh | **done** (see notes; real training run itself is GPU-only, untested here) | - |
| 10 | Full prep run (user-executed) | not started | - |
| 10b | diagnose_shift.py + eval_offline.py --zero-shot | **done** (see notes; both run for real against the real T-Rex midtrain checkpoint + the `eef62_train_smalltest` fixture, not just fixture-scale) | G15 **FAIL**; zero-shot floor recorded, does not beat hold-position |
| 10c | refit_vqvae_stats.py (if needed) | not started -- **G15 failed, so this is next**, but stopping here per this session's instruction | - |
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
| G15 | VQ-VAE codebook health | **FAIL** | real midtrain checkpoint's `tacf6_vqvae_{min,max,mask}` applied to 200 000 real origami F6 windows: L-ring/L-pinky collapse to 1/64 codes (top-code frac ~1.0, normalized entropy ~0); L/R-middle, R-ring/R-pinky also near-collapsed (18-40/64 codes used but >97% mass on 1 code); only thumb/index (both hands) use the codebook meaningfully (43-63/64 codes, entropy 0.45-0.69). Clamp-saturation only 0.85% (not the failure mode -- collapse, not saturation, per §11.8's "either way" framing) |
| G16 | Token budget | not started | - |
| G17 | Delay-curriculum parity | **PARTIAL** | loader half (`tactile_f6s_delayed`/`tactile_f6_history` == the extended F6 window's slice at each configured delay) passes in `test_delayed_lerobot_dataset.py`, real fixture data; the deploy-parity half needs `serve_zenoh.py` (step 13), not implemented yet |
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

## Step 8 findings (verify.py) — scoped to gates checkable at this point in the build

* Gates G0, G1a, G1b, G2, G2b, G3, G3b, G8, G8b already live as fixture-scale pytest tests
  (steps 3/5); `verify.py unit` just re-runs those files via `pytest -q` rather than
  reimplementing them (§10: "every gate is a test... or a verify.py subcommand"). Confirmed
  working: `python -m origami.verify unit` -> 22 passed.
* Added root-scale subcommands (`verify.py all --root <merged_root>`, the exact call
  `prepare.py`'s phase 2 makes per §5.5) for the 4 gates meaningfully checkable **from a
  merged root alone**, i.e. after stream-and-delete has already dropped the original raw
  season videos: **G18** (recorded `origami_prep.json` seasons vs the parsed split list,
  exact match unless `--extra-seasons train`), **G8/G8b** (re-validates `trex_norm_stats.json`
  shapes + the tactile mask isn't all-`True`), **G6** (weakened to a *self-consistency* check
  -- `gray_to_3ch` replicates one luma channel into 3, so all 3 decoded channels of every
  deform tile must be exactly equal; this doesn't re-verify losslessness against the
  now-gone source video, which is `test_convert.py`'s job while the source is still on
  disk), **G4** (weakened to a *pose-reconstruction* check -- rebuilds the left/right FK
  matrices from the stored 9-D `observation.state`/`action_abs` and re-derives the k=0 chunk
  step via the real `build_action_chunk`, comparing to the stored chunk's k=0 step at
  `atol=1e-5`; a literal full-season bitwise check needs the original per-frame `action65`,
  which is also gone by merge time -- that's `test_convert.py`'s `test_g4_...` job).
  G7/G7b/G9-G17 are not yet implementable (need `train_origami.py`/`policy.py`/
  `serve_zenoh.py`, steps 9/13/14) -- `verify.py list` reports why for each.
* Manually confirmed `verify.py all --root ...` against a real 2-shard merged root: G8/G6/G4
  all pass (100% at `atol=1e-5`); G18 correctly **fails** when the merged root's seasons don't
  match the given split (expected -- the manual test merged one season twice, not a real
  101-season train split).
* `test_verify.py` (5 tests, all pass, real FK/decode/encode/merge -- same 2-shard fixture
  pattern as `test_merge.py`): G8/G6/G4 pass on real merged data; G18 both matches (mocked
  split == the fixture season) and correctly fails (mocked split != the fixture season).
* Full `origami/tests/` suite: **53/53 pass**, ~163s total.

## Step 9 findings (trex_patch.py + train_origami.py + delayed_lerobot_dataset.py + .sh)

* **Two real environment gaps found: T-Rex's own declared dependencies (`timm`, `wandb`,
  `h5py`, `accelerate`) were never actually installed in this dev venv**, because steps
  1-8 only ever imported `utils.lerobot_common` and `qwen_vla.lerobot_dataset` — neither
  touches `qwen_vla.modeling_vla` (needs `timm` via `diffusion.py`) or `scripts/midtrain.py`
  (needs `wandb`/`h5py`/`accelerate`). The moment `trex_patch.py` imports
  `qwen_vla.modeling_vla` (for patch 2's assertion) and `train_origami.py` imports the
  vendored script's own dependencies, both failed with `ModuleNotFoundError` on a fresh
  `uv sync --extra dev`. Fixed by adding `timm`, `wandb`, `h5py`, `accelerate` to
  `pyproject.toml` (T-Rex's own `pyproject.toml`/`requirements.txt` pins versions; ours
  uses floors, consistent with the existing `transformers>=4.53.0` convention) and
  re-running `uv sync`. `origami/train_origami.py` now imports cleanly stand-alone
  (`from origami import train_origami` succeeds, confirmed).
* **`trex_patch.py` (§7.1).** Added `qwen_vla/modeling_vla.py` and
  `qwen_vla/modeling_qwen3vl_mot.py` to `upstream_manifest.json` (they were never hashed
  before — only files direct-imported/vendored in steps 1-8 were) at f88e10c; this
  automatically extends `test_upstream_drift.py`'s parametrized hash checks to cover them,
  since that test parametrizes over every key already in the manifest.
  * Patch 1 (`Qwen3VLAttentionMoT.forward`'s cached-prefix read) is implemented exactly per
    the plan's `read_cached_kv` code, gated on `torch.is_grad_enabled()`. Real
    `transformers.cache_utils.DynamicCache` (the actual class T-Rex uses, confirmed via
    `pip show transformers`/direct inspection) uses the `.layers` list layout, not the older
    `key_cache`/`value_cache` top-level lists — `read_cached_kv`'s first branch is what
    actually executes on the installed version; the second branch is dead code on this
    version but kept per the plan's exact given code (harmless, and would matter on an
    older transformers).
  * Patch 2 is exactly what the plan describes: an assertion, not a functional patch.
    Verified by source inspection (`inspect.getsource` + locating the `else:` branch) that
    `modeling_vla.py`'s `forward_flow_action_{full,partial}` still omit `attention_mask=` in
    their `past_kv is not None` branch on the pinned commit.
  * `test_trex_patch.py` (8 tests, all pass, CPU-only, no GPU): idempotency of `apply()`;
    both upstream-hash and patch-2 assertions pass on the real pinned checkout;
    `read_cached_kv` returns `None` before any cache write and matches a manual
    `DynamicCache.update()` reference afterward without appending; a synthetic
    "no-grad real forward that appends once, then grad-enabled recompute that must
    reproduce the identical K/V without a second append" scenario (the exact bug patch 1
    fixes) passes bit-for-bit (`torch.equal`).
  * **Not verified for real** (needs GPU + a full model): that gradient checkpointing
    actually double-appends without the patch, end-to-end, inside a real
    `Qwen3VLVLAModel` forward/backward. The unit tests prove the patched read function is
    behaviorally equivalent to a correct single append using synthetic cache objects
    (matching the plan's ask), which is what's checkable without a GPU; the full
    integration is implicitly exercised whenever `--gradient_checkpointing 1` is first used
    for real (§11 pilot run, step 11).
* **`train_origami.py` (§7.2).** Vendored from `git -C T-Rex show
  origin/full-pipeline:scripts/midtrain.py` (b23eafe, already in `upstream_manifest.json`
  from step 1) with the 8 numbered deltas, each marked `# ORIGAMI-DELTA:`.
  * Two deltas needed judgment calls beyond the plan's literal wording, both documented
    inline: (1) item 6 (`train_flare`) only applies to `train()`'s loop — `run_validation`
    never computes a flare loss at all (only gates sequence shape via its `use_flare` arg),
    so there was nothing to change there. (2) item 7 (`--max_steps`) needed an actual loop
    break, not just truncating the LR schedule's period as the plan's one-line description
    literally says — without a break, `--max_steps 40000` with `--n_epochs 1` would run
    every batch in the epoch regardless (`steps_per_epoch` for the real 101-season root is
    unknown until step 10's real prep run, but there is no reason to assume it's exactly
    40000). Added a `stop_training` flag breaking both the batch and epoch loops, plus a
    final checkpoint save on early stop.
  * **`locked_config` in the checkpoint is honest about a real gap, not silently
    papered over**: `meta/origami_prep.json` (written by `convert.py`/`merge.py`, read by
    `save_checkpoint` here) only carries `LockedConfig.digest()` (a hash), never the full
    `lower_body`/`neck`/`left_hand`/`right_hand` arrays — those are computed in
    `prepare.py::phase0_locked_config` and passed to worker processes in-memory, but never
    written back out to the merged root. So `training_args.json`'s `locked_config` field is
    `{"digest": ...}` only. A full G1c cross-check (`serve_zenoh.py` reconstructing the
    identical `LockedConfig` a checkpoint was trained under) needs `prepare.py` to also
    persist the arrays at the merged root — flagged here, not fixed (out of scope for step
    9; `prepare.py` is step 7's deliverable).
  * `optim=adamw8bit` lazily imports `bitsandbytes`, **not installed** in this dev venv
    (GPU-only, training-only dependency) — confirmed the import is deferred to inside the
    `if` branch so nothing else breaks; the branch itself is untestable without it.
  * `test_train_vendor.py` (3 tests, all pass, CPU-only/network-free — `full-pipeline` was
    already fetched into the submodule in step 1, so `git show origin/full-pipeline:...` is
    a local ref lookup): every changed opcode block from `difflib.SequenceMatcher` against
    the real upstream text carries an `# ORIGAMI-DELTA:` marker within a small context
    window; vendored file is valid Python; vendored file is a small delta, not a rewrite
    (>90% of upstream's line count preserved).
  * **Not verified for real** (needs GPU + real data): that the script actually trains
    end-to-end. `python -m py_compile`/import-level checks pass; the full run is step 11
    (pilot, 2000 steps, user-executed).
* **`delayed_lerobot_dataset.py` (§7.5-B).** `DelayedTRexLeRobotDataset` subclasses
  `TRexLeRobotDataset`, overriding only `_f6_offsets` (extends the window from `W` to
  `W + max(tactile_delay_offsets)` entries), `_build_delta_timestamps` (adds 2-offset
  deform delta_timestamps only for `--tactile_delay_scope both`), and `collate_fn`
  (recomputes the delay-dependent output keys after calling `super().collate_fn()` for
  everything else — images/actions/state/flare are untouched by the delay curriculum and
  reusing the base implementation for them avoids re-deriving already-verified logic).
  * **`both` scope is a documented simplification, not an exact per-value fetch.** The plan
    states deform costs "20 video seeks instead of 10" for `both` scope — exactly 2
    timestamps/key × 10 keys. Since LeRobot's `delta_timestamps` are fixed once at dataset
    construction (not variable per `__getitem__` call the way `midtrain.py`'s own
    `rng.choice` per-item sampling is), there is no way to fetch "the one delay value this
    particular sample happened to draw" without either (a) fetching every configured delay
    value per key (more than 20 seeks for the default 4-value offset list) or (b) reaching
    into LeRobot's internal per-item video query API to seek an arbitrary runtime-chosen
    timestamp, bypassing `delta_timestamps` entirely. Implemented (a)-adjacent but bounded
    to the plan's literal "20" figure: fetch only `[anchor, max(delay_offsets)]`, and for
    any sampled nonzero `delay_k` use the max-delay frame as an approximation rather than an
    exact per-value fetch. F6 does NOT have this limitation (numeric parquet, cheap to
    extend to the full offset range) and gets an exact per-sample-`delay_k` value, which is
    what the loader-level G17 test below actually checks. Flagged in the module docstring;
    revisit once §8.1-D's real throughput measurement says whether `both` is worth using at
    all.
  * `test_delayed_lerobot_dataset.py` (4 tests, all pass, real fixture data via
    `origami.convert.convert_season(..., max_frames_per_episode=...)`, same truncation
    pattern as `test_convert.py`/`test_merge.py`): `_f6_offsets` extends by exactly
    `max(delay_offsets)` entries (and matches upstream exactly for `scope="none"`); for
    every configured delay `k`, forcing `delay_k=k` via a monkeypatched `np.random.default_rng`
    makes `tactile_f6s_delayed`/`tactile_f6_history` equal `f6_window[:, W-1+k]` /
    `f6_window[:, k:k+W]` (bf16-precision tolerance, since the model consumes bf16); `scope="none"`
    leaves `tactile_f6s_delayed == tactile_f6s` (upstream's delay-≡-0 behavior, unchanged).
    Needed a small `_FakeProcessor`/`_FakeAccelerator` to exercise the real (unmodified)
    `TRexLeRobotDataset.collate_fn` without a real Qwen checkpoint on disk; also had to set
    `--use_flare 1 --n_flare_steps 1` in the test config to avoid a **pre-existing upstream
    quirk** (not something this subclass introduces): a single-offset `delta_timestamps`
    entry is squeezed (no leading temporal dim, per the step-2 finding), so
    `TRexLeRobotDataset.collate_fn`'s `head_seq[0]`/`head_seq[fi]` frame-indexing for
    FLARE only works when `KEY_HEAD` has >=2 offsets — true of the real recipe too
    (`--use_flare 1 --n_flare_steps 8` always), just not something a `use_flare=0` test
    config would naturally hit.
* **`train_origami.sh` (§7.3).** Copied verbatim from the plan; no flags invented beyond
  what's listed there.
* Full `origami/tests/` suite: **70/70 pass**, ~187s total (this session added 17 new
  tests: `test_trex_patch.py` (8), `test_train_vendor.py` (3),
  `test_delayed_lerobot_dataset.py` (4), plus 2 more from the 2 newly-hashed files'
  parametrized cases in `test_upstream_drift.py`, on top of the existing 53).

## `prepare.py` review-fix pass (7 issues, user-reported code review)

A user code review of `prepare.py` found 7 real issues, all fixed:

1. **LockedConfig computed per `--split`, not once over train (§3.2/§3.3 violation).**
   `phase0_locked_config` was handed whatever `--split` resolved, so a val run silently
   computed its own median and lived in a different absolute-state frame than train's.
   Fixed: `LockedConfig` is now a persisted, shared artifact (`write_locked_config`/
   `load_locked_config`, JSON at `--locked-config-path`, default
   `<cache-root>/locked_config.json`). A train run bootstraps it fresh if absent; a val run
   with none yet **refuses** (`AssertionError`, tested in
   `test_main_refuses_to_bootstrap_locked_config_from_val_split`) instead of silently
   computing over val's own seasons.
2. **No standalone phase 0 / no persistence.** Added `--phase {locked, all}` (`locked` stops
   after writing/loading the shared config) and the persistence from (1). A resumed run no
   longer re-fetches all 126 seasons' meta+data or risks a digest clash discovered only after
   hours of conversion.
3. **Resume bookkeeping had two sources of truth** (manifest.jsonl for skip-decisions, a
   separate `DONE` marker file for the returned shard list, written in that order with a
   crash window between them). Fixed: `manifest.jsonl` is now the *only* source of truth for
   both (`_read_manifest`); the `DONE`-marker mechanism is gone entirely.
4. **Config-invalidation guard was vacuous.** `PrepConfig`'s `image_size`/`deform_size` had
   no CLI flags and were never threaded into `ConvertConfig`, so the comparison could only
   ever compare defaults to defaults; `--instruction` (which does vary and does invalidate)
   wasn't in the compared dict at all. Fixed: added `--image-size`/`--deform-size` CLI flags,
   wired through to `ConvertConfig`, and added `instruction` to `PrepConfig.as_dict()`.
   `deform_codec`/`frame_stride` stay as documented fixed constants (hardcoded in
   `convert.py`/fixed by D4 respectively) rather than fake-configurable fields.
5. **Phase 2 never called `verify.py`**, despite PROGRESS.md previously claiming it did.
   Fixed: `origami/verify.py` gained `run_all_gates(root, split, extra_seasons, ...)`
   (factored out of `cmd_all`), and `prepare.py`'s `main()` now calls it right after
   `merge_shards(...)` and **asserts** all gates pass before writing final metadata — a
   synthetic single-season smoke run confirmed this actually halts the run (G18 correctly
   fails on a 1-season "split") rather than silently completing.
6. **Required metadata missing from `meta/origami_prep.json`.** Added `instruction`,
   `urdf_sha256` (via `kinematics.urdf_sha256`), the **full** `LockedConfig` arrays (not just
   the digest — closes the gap step 9 flagged as blocking a full G1c cross-check), and the
   exact `total_frames` phase 1 actually measured (summed from `manifest_entries`, replacing
   the §1.1a estimate) — all written directly in `prepare.py`'s final metadata write, no
   changes needed to `convert.py`/`merge.py`.
7. **Failed seasons left no durable record.** `run_phase1` now writes a
   `{"status": "failed", "error": ...}` manifest entry for any season whose conversion
   raises (instead of only a log line), and logs which seasons were excluded from the merge.
   A failed season is retried on the next run (its manifest entry doesn't count as `"done"`).

**A deeper, previously-undetected bug surfaced while verifying fix (1)/(2) against real
data — `LockedConfig.digest()` was not actually dtype-invariant.** Hashing
`np.round(arr, 6).tobytes()` depends on the array's dtype: float32 and float64 arrays
holding the "same" 6-decimal value can hash differently, because float32's coarser
representable grid can round to a bit-different float64 value than the same decimal number
stored natively in float64 (confirmed empirically, not just reasoned about — a synthetic
`1.234567` reproduced it, and so did the real fixture's medians). This matters because
`phase0_locked_config`'s medians come out **float32** (float32 parquet columns), but
`json.loads` -> `np.array(python_floats)` always produces **float64** — so every real
locked-config write/reload round trip was hitting this and would have made the new
digest-consistency check in `load_locked_config` **always** fail. Fixed in
`kinematics.py::LockedConfig.digest()`: hash a fixed-precision **decimal string** rendering
(`f"{v:.6f}"`) instead of raw bytes — verified dtype-invariant both syntactically (a
synthetic float32-vs-float64 test) and against real data
(`test_locked_config_digest_is_dtype_invariant`, `test_kinematics.py`). This bug predates
this fix pass (nothing in steps 1-9 previously round-tripped a `LockedConfig` through JSON to
expose it) but would have silently broken G1c and the train/val locked-config-sharing fix
above without this catch.

Verification: `main()` end-to-end (real FK, real convert/merge/verify, `run_phase1`
monkeypatched to a fast truncated in-process conversion instead of real multiprocessing/
network) confirmed the fixed pipeline against real fixture data — correct halt on a
synthetic G18 mismatch, then a full clean pass writing all the new metadata fields
correctly and consistently (`origami_prep.json`'s `locked_digest` from `merge_shards` ==
`locked_config.digest` from `prepare.py`'s own final write). Full `origami/tests/` suite:
**77/77 pass** (+7 new: 6 in `test_prepare.py`, 1 in `test_kinematics.py`), ~185s.

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

## Step 10b findings (diagnose_shift.py + eval_offline.py) — 5 real transformers-version compatibility bugs found; G15 FAILS for real; zero-shot does not beat hold-position

**What "step 10" actually means here.** REDESIGN_PLAN.md §12 step 10 is "full prep run on the
large server, 126 seasons, both splits" -- **not done, and out of scope for this session**
(no such server/corpus here). Per this session's explicit instruction, `data_trex_origami/
eef62_train_smalltest/` (a real merged/post-`prepare.py` root -- 17 seasons, 180 episodes,
1 389 294 frames, `image_size=224`, NOT the plan's 384x384 -- someone else's prior prep run,
scope/provenance unknown beyond what's in its own `meta/origami_prep.json`) stood in as the
"val root" for exercising step 10b's two scripts. **This is a stand-in for testing the
scripts, not a claim that step 10's real 126-season run has happened.** Don't read "10b done"
as "10 done" in a future session.

* **`origami/diagnose_shift.py` (§11.9 step 2).** Implements chunk-delta magnitude per action
  block, F6 magnitude per finger, deform-tile occupancy per finger (new metric, not named
  precisely in the plan -- defined here as the fraction of a decoded tile's pixels deviating
  >8 gray levels from the tile's own median, a proxy for "this sensor is showing real contact
  deformation, not a flat/dead readout"), **G15** (frozen VQ-VAE codebook health), and
  frozen-ViT pooled-patch-token feature stats + cross-dataset cosine distance vs a
  `zekaiwang/trex_dataset` sample. The first three sections need only the merged root (read
  straight from `meta/trex_norm_stats.json` plus a handful of seek'd video frames -- no
  checkpoint); G15 and the ViT section need the T-Rex midtrain checkpoint.
  * **Perf bug caught before it shipped:** the first version of the frame-sampling helper
    picked one uniform-random frame index per video file, then decoded *sequentially from
    frame 0* to reach it. On real ~7 700-frame episodes this meant an average half-episode
    decode per file -- 20 samples/finger across all 180 real episodes took **~4 minutes**.
    Fixed by seeking to the target frame's approximate timestamp first (`container.seek`,
    `backward=True`) -- safe because these videos use `video.g: 2` (a keyframe every 2
    frames, per §5.2), so a seek lands within ~1 frame. Same 20-sample run: **~5 seconds**
    after the fix. `_seek_sample_frames` is shared by the deform-occupancy and ViT-feature
    frame sampling.
  * **G15 result, real checkpoint + 200 000 real origami F6 windows (`miniFranka/
    T-Rex_midtrain_mecka23k_ucb100_vqvae_epoch6`, downloaded from HF, 8.5 GB `model.pt`):
    FAILS, exactly the failure mode §11.8 predicted.** L-ring and L-pinky collapse to a
    single code each (top-code fraction ~1.0, normalized entropy ~0); L/R-middle and
    R-ring/R-pinky are effectively collapsed too (18-40/64 codes technically "used" but
    >97% of mass on one code); only the four live slots (L/R-thumb, L/R-index) use the
    codebook in a meaningful spread (43-63/64 codes, normalized entropy 0.45-0.69). Clamp
    saturation is only 0.85% overall -- **the failure is collapse (values sitting in a
    narrow sliver of T-Rex's own min/max range), not clamping**, confirming the "or
    origami forces occupy a sliver near -1" half of §11.8's "either way" framing, not the
    saturation half. This directly explains eval_offline's tactile-ablation result below.
  * **ViT feature cosine distance is ~0.0002 (near-identical) between origami and the
    trex_dataset sample -- flagged as likely NOT meaningful, not read as "no visual shift."**
    ViT activations are known to have a few very-large-magnitude "massive activation"
    channels present regardless of input content; those can dominate a raw cosine similarity
    between two mean feature vectors and mask real shift. `render_markdown` prints this
    caveat inline so the report doesn't get over-read. A more robust statistic (per-channel
    z-scored, or excluding the top-k magnitude channels) would be needed to actually test
    §11.9's qualitative claim (FOV/fisheye/photometry shift) -- not built here, out of scope.
  * Full real report (200k F6 windows, 60 deform + 60 ViT samples/dataset) saved this session
    at `reports/diagnose_shift_report.md` (not committed -- regenerate via `python -m
    origami.diagnose_shift --root <root> --checkpoint <ckpt> --trex-dataset-root <sample>`).
  * `test_diagnose_shift.py`: checkpoint-free sections always run against the real
    `eef62_train_smalltest` fixture (shape/sanity + directional checks, e.g. thumb/index
    force RMS > middle/ring RMS); the G15 and full-`run()` tests are `skipif`-gated on
    `checkpoints/midtrain/` existing on disk (gitignored, never committed -- ~8.5 GB) and
    were confirmed passing against the real downloaded checkpoint in this session.

* **`origami/eval_offline.py` (§8.1-A + a version of §8.1-D; §11.9 step 1).** Drives the
  **real, unmodified deploy-time inference path** -- `scripts.test.CascadedServer` and
  `model_load`, imported per §13's exact list, never reimplemented -- frame-by-frame (batch
  size 1, matching how the SDK actually calls it) over real held-out `eef62_train_smalltest`
  frames, across 4 configs: `cascaded` (deployed), `disable_tactile` (action-expert-only
  ablation), `tactile_zeroed` (cascaded path, tactile inputs zeroed -- a 4th config beyond the
  plan's named 3, cheap to add given the infra and directly useful given G15's finding), and
  `hold_position` (trivial baseline, no model call: zero delta9 + current hand state, no FK
  needed since `observation.state` is already in the same delta9+hand22 block layout as
  `action`). Predicted actions are denormalized using the **checkpoint's own** stats (from
  `stats_data.json` -- the honest zero-shot test: the model never saw origami's action scale),
  compared to the real un-normalized `action` ground truth in physical units.
  * **Real 100-sample result (§11.9 step 1's "run before step 11"):**
    `cascaded k=0 mean MAE=0.1675` **does not beat** `hold-position k=0 mean MAE=0.01462`.
    Breaking down by block clarifies *why*, and it isn't uniform: arm-pose blocks
    (`{L,R}_trans3`/`{L,R}_rot6d6`) are actually competitive with hold-position (MAE same
    order of magnitude, 0.006-0.04 vs 0.004-0.03; rotation `variance_share` 0.67-1.0, i.e.
    the model captures real rotational variance) -- the **entire gap is the `hand22` blocks**
    (cascaded MAE 0.44-0.47 vs hold-position 0.017-0.020, ~25x worse). This is the expected
    shape of the shift: `hand22` is an *absolute joint target* in a hand-specific convention
    (Sharpa Wave origami vs T-Rex midtrain's Dexmate-adjacent embodiment per §11.9's table),
    so it has no reason to transfer zero-shot at all, while EEF pose deltas are a much more
    embodiment-portable quantity that partially does. **Matches §11.9's warning exactly**
    ("if it is no better than hold-position, the pretrained init is worth less than assumed")
    for the hand-pose dimensions specifically, not uniformly across the whole action space --
    a more precise diagnosis than the plan's single aggregate-floor framing suggests.
  * **`cascaded` ≈ `disable_tactile` ≈ `tactile_zeroed` (all three nearly identical numbers,
    every block).** The tactile expert currently contributes ~nothing to the zero-shot
    prediction -- **directly cross-validates G15's finding** from `diagnose_shift.py`
    (near-collapsed codebook -> near-constant tactile token -> negligible influence on the
    cascaded output), found independently by two different mechanisms in the same session.
  * Full real 100-sample report saved this session at `reports/eval_offline_report.md` (not
    committed -- regenerate via `python -m origami.eval_offline --root <val_root> --checkpoint
    <ckpt> --zero-shot`).
  * `test_eval_offline.py`: checkpoint-free tests cover `hold_position_prediction`'s exact
    block-wise construction and `action_space_accuracy`'s MSE/MAE/variance-share arithmetic
    (zero-error and known-offset cases) without needing any model; one `skipif`-gated
    real-checkpoint smoke test (2 samples) confirms the real cascaded path end-to-end.

* **5 real, previously-undetected `transformers`-version compatibility bugs found and fixed
  in `origami/trex_patch.py` (patches 3, 4, 5 -- patches 1/2 are from step 9) while getting
  the FIRST real forward pass of this whole project to run.** Steps 1-9 only ever imported
  `utils.lerobot_common` / `qwen_vla.lerobot_dataset` / vendored `train_origami.py` at
  module level and unit-tested around real GPU forward passes (explicitly flagged in step 9's
  notes above as "not verified for real: needs GPU + real data"). This session's scripts are
  the first callers to actually run a real image through the real checkpoint's real
  `Qwen3VLVLAModel`/`Qwen3VLModelMoT`/vision tower on our pinned `transformers` (confirmed
  installed as a `5.16.x` release; T-Rex's own checkpoint was trained against
  `4.57.0.dev0` per its `config.json` -- a large version gap). Without all 5 patches
  (1/2 from step 9 plus these 3), **`model_load` and every real forward pass raise before
  touching a single pixel** -- this would have been the first thing to break at step 11's
  pilot train had it not surfaced here first. Each is hash-gated the same way as patches 1/2
  (`_EXPECTED_HASHES` in `trex_patch.py`, so upstream drift re-triggers a re-check) and has
  its own module-docstring section in `trex_patch.py` with the exact mechanism; summary:
  1. **Patch 3 -- vision-tower output unwrap** (`Qwen3VLVLAModel.prepare_inputs_embeds`).
     Upstream assumes `self.visual(...)` returns a plain `(merged_hidden_states,
     deepstack_features)` tuple (`out[0]` = merged). On our transformers, it always returns a
     `BaseModelOutputWithDeepstackFeatures` (an `OrderedDict` subclass -- NOT tuple/list, so
     upstream's `isinstance` check silently takes the wrong branch), whose `.last_hidden_state`
     is *pre-merge* (wrong token count) and whose `.pooler_output` is the actual merged
     sequence. Unpatched: crashes on `image_features.to(dtype)` the moment any real image
     goes through (`AttributeError` -- dicts have no `.to`).
  2. **Patch 4 -- rotary-embedding config schema** (`Qwen3VLRotaryEmbeddingWrapper.__init__`).
     Upstream builds a synthetic `_RopeCfg` with separate `rope_theta`/`rope_scaling` fields
     (T-Rex's transformers era). Ours reads a single consolidated `config.rope_parameters`
     dict instead. Unpatched: `AttributeError: '_RopeCfg' object has no attribute
     'rope_parameters'` -- raised at **model construction time**, before any data at all.
  3. **Patch 5 -- `get_rope_index` signature** (`Qwen3VLVLAModel.get_rope_index`). Our
     transformers' `Qwen3VLModel.get_rope_index` gained a required `mm_token_type_ids`
     positional arg upstream's call doesn't pass. Extra wrinkle: the `scripts/test.py`
     zero-shot path (`_build_qwen3vl_from_config`) binds a closure-local `_RopeStub` object
     that duck-types only `.config` -- the real method also calls `self.get_vision_position_
     ids(...)` internally, which the stub doesn't have either. Fixed by building
     `mm_token_type_ids` from `input_ids == image_token_id` (no video support needed) and,
     for the stub case, calling the real `Qwen3VLModel.get_rope_index` against an
     `object.__new__`'d real `Qwen3VLModel` instance (bypassing `__init__`/weight allocation
     entirely) with the stub's `.config` copied over -- since position-id computation never
     touches learned weights, this is safe and exact, not an approximation.
  * All 3 new patches have CPU-only synthetic unit tests in `test_trex_patch.py` (18 tests
    total in that file now, up from 8) that don't need the checkpoint; the real end-to-end
    integration (all 5 patches together, real 8.5 GB checkpoint, real fixture frames, real
    cascaded slow/fast inference producing a correctly-shaped `[16,62]` prediction) was
    manually verified in this session and is what `eval_offline.py`'s `skipif`-gated
    checkpoint smoke test now checks on any machine that has the checkpoint downloaded.
  * **Not yet known:** whether the *training* forward path (`train_origami.py`'s `train()`/
    `run_validation()`, which call `prepare_inputs_embeds`/`get_rope_index` the same way but
    also exercise gradient checkpointing + the cascaded tactile-expert training step) hits
    any *further* version-drift issues beyond these 5 -- only the CascadedServer inference
    path was actually exercised this session. Flagging for step 11's pilot train.

* **Added `pyzmq>=26.0.0` to `pyproject.toml`.** `scripts/test.py` (needed for `model_load`/
  `CascadedServer`, per §13's import list) imports `zmq` at module level for its ZMQ server,
  even though `model_load`/`CascadedServer` themselves never touch it -- wasn't previously a
  dependency because nothing had imported `scripts.test` before this session.

* Environment for this session: real GPU available (RTX 5060 Ti, 16 GB), 251 GB RAM, 221 GB
  free disk, network reachable -- unlike prior sessions' "GPU-only, untested here" notes.
  Downloaded (both gitignored, never committed): the full midtrain checkpoint to
  `checkpoints/midtrain/` (8.5 GB `model.pt` + small config/processor/stats files) and a
  6-video sample of `zekaiwang/trex_dataset`'s `head_left` camera to `trex_dataset_sample/`.
  Both `.gitignore`d this session (`/checkpoints/`, `/trex_dataset_sample/`).

* Full `origami/tests/` suite: **97/98 pass** (86/87 in the pre-existing suite + all 11 new
  `test_diagnose_shift.py`/`test_eval_offline.py` tests), ~875s + ~145s. The 87-test run was
  ~15x slower than the historical ~185-200s baseline for a similar-sized suite (14.5 min this
  time) despite no code-path changes to the slow tests -- likely shared-machine
  disk/GPU contention rather than a regression (nothing here uses more compute than before);
  re-time on a quiet machine before reading anything into that number.
  Checkpoint-loading tests are inherently slow regardless of contention: ~84s just to
  construct+load the full 2B-param model once on CPU before moving to GPU bf16.

* **Pre-existing test failure found, NOT caused by this session's changes (verified via `git
  status` -- `splits.json`/`test_splits.py` are untouched in this session's diff):
  `test_splits.py::test_fixture_season_absent_from_both_splits` now FAILS** --
  `season_POC22061_2026_05_23_19_21_25_train` (this whole project's "the fixture season",
  substituted in step 1 specifically *because* it was outside both splits) **is now inside
  `splits["train"]`**. `splits.json` is regenerated fresh from `dataset.md` and
  `test_splits_json_matches_fresh_parse_of_dataset_md` still passes, so this isn't a stale
  `splits.json` -- **`dataset.md` itself changed** (commit `a5005d7 "update split"`, already
  on `main` before this session started) to now include this season in the real train split.
  Everything downstream still works because nothing else actually depended on the
  exclusion -- `data_trex_origami/eef62_train_smalltest/` (this session's stand-in val root)
  is named "train_smalltest" and its `origami_prep.json` already lists this season among its
  17, consistent with it now genuinely being a train-split season -- but the test's premise
  is stale and someone should either pick a new genuinely-excluded fixture season or update
  the test/PROGRESS.md's framing. **Not fixed here** -- out of step 10b's scope and the
  right fix (which season, and whether anything else quietly assumed the old exclusion) is a
  judgment call, not a mechanical one.
