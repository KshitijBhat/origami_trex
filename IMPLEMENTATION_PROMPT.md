# Implementation prompt — origami × T-Rex eef-62 redesign

Implement the codebase specified in `REDESIGN_PLAN.md`, at the repo root
`/home/kshitij/origami_trex`, on the current branch `dev/redesign`.

**Read `REDESIGN_PLAN.md` in full before writing any code.** It is the single source of
truth: ~1440 lines, §0–§13, with 26 numbered verification gates (G0–G18, incl. sub-gates) and a 15-step build
order in §12. Every design decision, tensor definition, threshold and file:line citation you
need is already in it. Do not re-derive facts it states — they were verified against the
actual data and the actual upstream source. Where this prompt and the plan disagree, the plan
wins; tell me about the conflict.

## Scope

Deliver **all code and tests**: §12 steps 1–9 and the code for steps 13–15
(`policy.py`, `retarget.py`, `serve_zenoh.py`, `bench_latency.py`, `eval_offline.py`,
`eval_shadow.py`, `diagnose_shift.py`, `refit_vqvae_stats.py`, `docker/`).

Steps **10, 10b, 10c, 11, 11b, 12** are *runs* (full data prep, zero-shot eval, training) —
I execute those. Build them so they are launchable, resumable and correct; do not attempt to
run them.

## Environment

- Python via **uv + the workspace venv at `.venv`** — never bare `pip`, never `pip --user`.
  Add deps to `pyproject.toml`, then `uv sync` / `uv pip install -e .`.
- Local GPU is an **RTX 4060 Laptop, 8 GB** — training happens elsewhere (A100). Local work
  is CPU-only: prep, tests, gates. Do not add code that assumes a big GPU is present.
- **23 GB free local disk** on this laptop. **The full prep run happens on a separate
  large-RAM/large-disk server** — ≈ 107 GB output and 126–390 GB streamed. Nothing may write
  large artifacts under the repo by default; `--out-root` / `--cache-root` are required
  arguments with a `statvfs` precondition (§5.6).
- **The HF dataset is gated.** File reads need `HF_TOKEN`; the tree listing is public.
  `prepare.py` validates the token before phase 0 (§5.1).
- Test fixture: `season_POC22061_2026_07_23_10_20_33_train/lerobot3.0/` (22 episodes,
  95 533 frames, h264). It is the only local season, it is **not on the hub, and it is in
  neither split list** — a test-only 127th season. It must never enter train or val (G18).
- Robot asset: `north_poc2_2_urdf_usd/north_poc2_2_v3_1.urdf` (complete, 65 revolute joints).
- Competition SDK: `origami-inference-kit-participant/sharpa_north_ces_lite_sdk-main/`.

## Hard invariants — violating any of these silently breaks the project

1. **`T-Rex/` is read-only.** It is a git submodule pinned at `f88e10c` with **zero** local
   diff. Never edit a file under it. Import from it (§13 lists exactly what); monkeypatch via
   `origami/trex_patch.py` where behaviour must change. `origami/tests/test_upstream_drift.py`
   must fail if any depended-on upstream file changes.
2. **Reuse, don't reimplement.** The pose/chunk/stats math is imported verbatim from
   `T-Rex/utils/lerobot_common.py`. If you find yourself typing
   `R_curr.T @ (t_targ - t_curr)`, you are reimplementing something you should import.
   §13 lists the exact import set and the three functions to *copy* (not import) because
   their modules pull in hardware dependencies.
3. **Prep and deploy must be bit-identical from the 224×224 wire image onward.** Any
   resize, crop, colour or normalization step added to one path must be added to the other,
   from the same recorded config. Gates G12/G14/G16 exist for this.
4. **One kinematics module.** `origami/kinematics.py` is the only FK/IK in the codebase, used
   by prep, eval and deploy. The `LockedConfig` digest must match across dataset meta,
   checkpoint and server (G1c).
5. **Gate discipline.** Each §12 step ends at named gates. Do not start step *n+1* until step
   *n*'s gates pass. If a gate cannot pass, stop and report — do not weaken the threshold.
6. **Tests are CPU-only and network-free.** Use the fixture season and a tiny stub Qwen config
   for the forward-pass gate (G7b). No test may download anything.

## Traps I already hit — do not rediscover these

- **Chunk-base, not frame-to-frame deltas.** All 16 chunk steps are relative to *one* base
  pose at frame `i`, and targets come from the **commanded** `action` joints, not the next
  `state`. `NEW_DESIGN.md` says otherwise and is wrong; §6 has the correct algebra with a
  callout.
- **`observation.state.tcp` is identically zero.** FK is mandatory; there is no EEF data.
- **The URDF is complete**, but the SolidWorks exporter puts `name=` on its own line. Parse
  the XML; a `grep '<joint name='` will convince you the arms are missing. Cast joints to
  **float64** before FK.
- **Video files are chunked per *key*, not per episode.** In the fixture, `head_left` spans 6
  files, `tactile_deform` 1, `tactile_raw` 22. You *must* use the per-key
  `(chunk_index, file_index, from_timestamp, to_timestamp)` columns in
  `meta/episodes/**.parquet`. Seek lands on the preceding keyframe, so pre-roll and drop
  frames until the PTS reaches `from_timestamp`.
- **Deform must be exactly 240×240.** `deform_proj = ActionEmbedder(28800)` = 128·15·15. Any
  other size breaks the head with no useful error.
- **Decode the luma (Y) plane directly** (`format="gray"`), never YUV→RGB→gray.
- **`NormStatsAccumulator` OOMs** — ≈ 47 GB for the action block at the real corpus size. Use
  the streaming + reservoir replacement in §5.4, API-compatible, and prove it with G8.
- **`dataset.md`'s statistics table is stale** (51 seasons / 4.76 M frames). The real scope is
  **143 seasons on the hub, 126 in the split (101/25), ≈ 11.8 M frames ≈ 109 h** — §1.1a.
  Parse the split lists from `dataset.md`'s `## My Split` block; never hard-code counts;
  replace the frame estimate with the exact sum of per-season `info.json` in phase 0.
- **An all-`True` tactile mask is a bug.** Four fingers are dead (mean |F| 0.002–0.065 N);
  G8b requires them masked out.
- **Two different F6 normalizations.** The per-frame vector uses *our* q01/q99 from the
  loader; the VQ-VAE window is normalized *inside the model* by the **T-Rex checkpoint's**
  `tacf6_vqvae_{min,max,mask}`. Fixing the first does not fix the second — that is §11.8.
- **At deploy use Gram–Schmidt `rot6d_to_matrix`** (`eval_trex_async.py:283`), not
  `lerobot_common.get_rot_mat` — policy output is only approximately orthonormal.
- **`--tactile_delay_offsets` is a no-op on the LeRobot path** as shipped; §7.5-B specifies
  the loader subclass that implements it.
- **Motor dims 58:65 are held at the observed value**, never predicted, never stale.
- **`tactile_f6` is `[B, 10, 6]`** — `tacf6_embedder` is `ActionEmbedder(6, H)`.
- **`train_origami.py` is vendored from `origin/full-pipeline:scripts/midtrain.py`**, not from
  `main:scripts/train.py`. `full-pipeline` is fetched but not checked out; read it with
  `git -C T-Rex show origin/full-pipeline:scripts/midtrain.py`.
- **Cite upstream by `path::symbol`, not by line number.** Line numbers in the plan were
  verified once against the pinned refs but drift; symbols do not. If a cited line looks
  wrong, grep for the symbol and trust that.

## Probe before designing (§12 steps 1–2)

Two probes gate downstream decisions. Run them first and report the results before building
on them:

1. `origami/tests/test_lerobot_probe.py` — write a 3-frame dataset with the **full**
   `build_trex_features` schema, reopen with `delta_timestamps`, assert every key round-trips.
   Resolves §11.4: `add_frame`/`save_episode`/`finalize` signatures, whether non-`observation.images.*`
   keys are accepted as `dtype: "video"` (the 10 deform keys are `observation.tactile_deform.*`),
   and whether `lerobot.datasets.aggregate.aggregate_datasets` exists — **if it does, use it and
   delete the hand-rolled merger in §5.3.**
2. `origami/tests/test_processor_probe.py` — print
   `processor.image_processor.patch_size * merge_size` and the resolved grid for each candidate
   `--image_size`, then pick the square size whose token count is closest to upstream's
   `384 288` (§4.5-A).

If either probe contradicts the plan, stop and tell me before adapting.

## Working agreement

- Work **step by step through §12**, one step per turn where practical. After each step,
  report: what you built, which gates ran, their measured numbers vs thresholds, and anything
  that surprised you.
- Maintain `origami/PROGRESS.md`: a checklist of §12 steps and G0–G17 gates with
  status + measured values. Update it as you go — it is the handoff artifact if context is
  compacted.
- **Decide routine things yourself** (naming, file splits, logging, argparse ergonomics,
  error messages). **Ask me** only when the plan is genuinely ambiguous *and* the two readings
  produce materially different code. Do not ask permission to proceed between steps.
- Prefer clear, typed, docstring'd code that matches the surrounding style. Cite the plan
  section in a module docstring (`Implements REDESIGN_PLAN.md §5.2`) and cite upstream
  file:line whenever you copy or mirror upstream logic.
- Commit per §12 step, on `dev/redesign`, with the gates that passed in the message. Do not
  push. Do not commit data, checkpoints or logs.
- **Report failures honestly.** If a gate fails, say so with the numbers. If you skipped
  something, say what and why. Do not report a step complete on unrun tests.

## Definition of done

- `origami/` is complete per §2 layout, importable, and `uv run pytest origami/tests` is green
  on CPU with no network.
- `python -m origami.verify --help` exposes every gate G1–G17 as a subcommand, and all gates
  runnable against the fixture season pass at their stated thresholds.
- `python -m origami.prepare --split train --out-root … --dry-run` validates config, prints
  the storage/time budget banner, and refuses to start on insufficient disk.
- `origami/train_origami.sh` runs 5 smoke steps (G9a) and the vendor diff test (§7.2) is green.
- `origami/serve_zenoh.py` passes the SDK's `check_zenoh_policy.py` and metadata validation
  against a locally launched server with a stub checkpoint.
- `origami/PROGRESS.md` records every gate with its measured value.
- T-Rex submodule diff is empty: `git -C T-Rex status --porcelain` prints nothing.

Start by reading `REDESIGN_PLAN.md`, then `origami/PROGRESS.md` if it exists, then do §12
step 1. Report back before step 3.
