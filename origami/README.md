# Origami dataset preparation

`origami/prepare.py` converts the Robotic Origami Challenge dataset
(`SharpaIT/Robotic_Origami_Challenge`, gated on the Hub) into the T-Rex eef-62 LeRobot format
consumed by `T-Rex/qwen_vla/lerobot_dataset.py::TRexLeRobotDataset`. See REDESIGN_PLAN.md §5
for the full design; this doc covers setup end-to-end plus the prepare command and its
arguments.

## 1. Clone the repo (with the T-Rex submodule)

```bash
git clone --recurse-submodules <this-repo-url> origami_trex
cd origami_trex
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

This must check out `T-Rex/` at the pinned commit (`f88e10c`) with **zero local diff** —
`T-Rex/` is never modified in this repo (gate G0 in `origami/tests/test_upstream_drift.py`
enforces this).

## 2. Install `uv` and sync the environment

This project uses [`uv`](https://docs.astral.sh/uv/) for Python/venv/dependency management —
never bare `pip` or `pip --user`.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # skip if uv is already installed
uv sync --extra dev
```

`uv sync` reads `pyproject.toml`/`uv.lock`, installs a matching Python (>=3.12) if you don't
already have one, creates `.venv/` at the repo root, and installs everything: `pinocchio`
(`pin`) + `pin-pink` (**not** the PyPI package literally named `pink` — that's an unrelated
code-formatting tool that shadows the real inverse-kinematics library) + `qpsolvers[daqp]`
for FK/IK, `av` for video decode, `lerobot[dataset]`, `transformers`/`accelerate`/`timm` for
the T-Rex model, plus `pytest` (the `dev` extra).

Use `.venv/bin/python` or `uv run <cmd>` for everything below — never the system Python.

Verify the venv works:

```bash
.venv/bin/python -c "import pinocchio, pink, av, lerobot; print('ok')"
```

## 3. Get Hub access and set `HF_TOKEN`

`SharpaIT/Robotic_Origami_Challenge` is a **gated** dataset — request access on its Hub page
first, then create a read-scoped token at https://huggingface.co/settings/tokens.

Put it in a `.env` file at the repo root (already `.gitignore`d — never commit a token):

```bash
echo 'HF_TOKEN=hf_...' > .env
```

Load it into your shell before running anything that talks to the Hub:

```bash
set -a && source .env && set +a
```

(or just `export HF_TOKEN=hf_...` directly, or pass `--hf-token hf_...` to `prepare.py`
below instead of relying on the env var).

## 4. Sanity-check the install

```bash
.venv/bin/python -m pytest origami/tests/ -q
```

Most of these tests are CPU-only and network-free, but several (`test_convert.py`,
`test_merge.py`, `test_verify.py`, `test_delayed_lerobot_dataset.py`) need a local fixture
season at `season_POC22061_2026_05_23_19_21_25_train/lerobot3.0/` at the repo root (gitignored,
~2.8 GB) — fetch it once with `HF_TOKEN` set:

```bash
.venv/bin/python -c "
from origami.fetch import download_season
download_season('season_POC22061_2026_05_23_19_21_25_train', '.', '$HF_TOKEN')
"
```

The whole suite takes a few minutes (real FK/IK, real video decode/encode against that
fixture — nothing here is mocked out). Expect it to pass fully before moving on.

## 5. Prepare the dataset

`prepare.py` runs three phases per split (train/val are prepared separately, into separate
`--out-root`s, but **share one `LockedConfig`** — see below):

1. **Phase 0** — fetches `meta/` + `data/` only (no video, ~66 MB/season) for every season in
   the split, to get the exact frame count and compute `LockedConfig` (§3.2/§3.3). Only ever
   runs over the **train** split (see "Locked config is shared across train/val" below);
   loaded from disk instead when it already exists.
2. **Phase 1** — per season: download (video included) → `convert_season` (FK/IK retarget +
   deform-video encode) → drop the raw download. Resumable: seasons recorded
   `"status": "done"` in `<out-root>_shards/manifest.jsonl` are skipped on a re-run; seasons
   that failed are recorded `"status": "failed"` and retried on the next run.
3. **Phase 2** — merges all per-season shards into one dataset root at `--out-root`
   (`origami/merge.py`), then runs `origami/verify.py`'s root-scale gates (G18, G8/G8b, G6,
   G4) against it and **fails the run** if any gate doesn't pass.

Run it from the repo root with the project venv (`uv run` or `.venv/bin/python`).

### Command

```bash
export HF_TOKEN=hf_...   # or pass --hf-token

python -m origami.prepare \
  --split train \
  --out-root /mnt/big/origami/eef62_train \
  --cache-root /mnt/big/origami/_src \
  --workers 3 \
  --disk-budget 3
```

Run again with `--split val` and a different `--out-root` for the validation set (each split
is its own dataset root — that's what `--lerobot_val_root` in `train_origami.sh` points at).

### Locked config is shared across train/val — run train first

§3.2/§3.3: `LockedConfig` (the frozen lower-body/neck posture the FK/IK retargeting is
computed against) must be computed **once, over the train split**, and reused unchanged for
val — recomputing it per split would put train and val in different absolute-state frames.
`prepare.py` enforces this: it persists the config to `--locked-config-path` (default
`<cache-root>/locked_config.json`), a **train** run bootstraps it if missing, and a **val**
run refuses to start if it's missing (rather than silently computing its own). So:

```bash
# 1. train first (bootstraps locked_config.json)
python -m origami.prepare --split train --out-root .../eef62_train --cache-root .../_src

# 2. val reuses the same --cache-root (and therefore the same locked_config.json)
python -m origami.prepare --split val --out-root .../eef62_val --cache-root .../_src
```

If you'd rather compute (or refresh) the shared config without converting anything yet, use
`--phase locked` — it stops right after phase 0:

```bash
python -m origami.prepare --split train --out-root .../eef62_train --cache-root .../_src \
  --phase locked
```

### Quick smoke test (one season)

To sanity-check the pipeline end-to-end before committing to a full split, cap it to the
first season with `--limit-seasons 1` and point both roots somewhere scratch/disposable:

```bash
python -m origami.prepare \
  --split train \
  --out-root /tmp/origami_smoke \
  --cache-root /tmp/origami_smoke_cache \
  --workers 1 \
  --disk-budget 1 \
  --limit-seasons 1
```

Don't reuse that `--out-root` for a real run afterward — `prep_config` locking (§5.5) only
guards the conversion *settings*, not which seasons were included, so a later full run would
happily merge into a partially-populated root.

### Disk and cache space

- **`--cache-root`** (raw per-season downloads, deleted right after each season converts):
  needs `--disk-budget * 3.5 GB` free — ~3.5 GB per season resident at once, not the whole
  corpus. At the default `--disk-budget 3` that's **~10.5 GB**.
- **`--out-root`** (final merged dataset, accumulates and is never deleted): needs
  `n_seasons * 0.85 GB` free. For the full split lists that's **~86 GB** for train (101
  seasons) and **~21 GB** for val (25 seasons); add ~0.85 GB/season more if using
  `--extra-seasons train` (17 extra seasons, ~14 GB).
- Both numbers are enforced at startup via `shutil.disk_usage` — the run fails fast with the
  exact shortfall if either path doesn't have enough free space, rather than partway through
  phase 1.
- `--cache-root` and `--out-root` can point at the same filesystem or different ones; there's
  no requirement they be on the same disk.

### Arguments

| flag | default | meaning |
|---|---|---|
| `--split` | *(required)* | `train` or `val` — which season list from `dataset.md`'s `## My Split` block to prepare. |
| `--out-root` | *(required)* | Where the final merged LeRobot dataset root is written. |
| `--cache-root` | *(required)* | Scratch space for per-season raw downloads; each season is deleted from here right after it converts (stream-and-delete), so this only needs to hold `--disk-budget` seasons at once, not the whole corpus. |
| `--workers` | `3` | Number of seasons converted in parallel (`ProcessPoolExecutor`). |
| `--disk-budget` | `3` | Max seasons' raw video resident on disk at once (independent of `--workers` — a semaphore around each worker's download+convert+drop). |
| `--urdf` | `north_poc2_2_urdf_usd/north_poc2_2_v3_1.urdf` | Robot URDF used for FK/IK retargeting. |
| `--extra-seasons` | `none` | `train` folds in the hub's 17 seasons that sit outside both the train/val split lists (train split only — keeps val frozen; see §1.1a). |
| `--instruction` | `origami.constants.INSTRUCTION` | The language instruction written as every frame's `task` (§5.7). Part of the config locked into `--out-root` on first write (see below) — changing it on a later run into the same root is refused. |
| `--image-size H W` | `224 224` | Camera wire resolution (D7). Locked into `--out-root` on first write. |
| `--deform-size H W` | `240 240` | Deform-tile resolution. Locked into `--out-root` on first write. |
| `--hf-token` | `$HF_TOKEN` env var | Hub token with read access to the gated dataset repo. Validated once before phase 0 starts. |
| `--phase` | `all` | `locked`: only compute/refresh the shared `LockedConfig` and exit (see above). `all`: locked (if needed) + convert + merge + verify. |
| `--locked-config-path` | `<cache-root>/locked_config.json` | Where the shared `LockedConfig` (§3.3) is read from / written to. |
| `--limit-seasons` | *(none)* | Restrict the resolved split to the first N seasons — for a quick local smoke test only; use a separate `--out-root` from any real run (see above). |

### Preflight checks (fail fast, before any download)

- `--hf-token` is validated against the Hub before phase 0 runs.
- Every season in the requested split must resolve on the Hub, and the split lists
  themselves are re-validated (101 train / 25 val, no overlap) — G18.
- Disk space on both `--out-root` and `--cache-root` must cover
  `n_seasons * 0.85 GB + disk_budget * 3.5 GB` (output estimate + raw-download headroom),
  checked via `shutil.disk_usage`.
- Re-running into an existing `--out-root` with different settings (`--instruction`,
  `--image-size`, `--deform-size`, `--extra-seasons`) is refused — those are recorded in
  `meta/origami_prep.json` on first write and locked in. (`--deform-size`'s codec and the
  frame stride are fixed constants, not CLI flags, so they can't drift silently.)

### After it finishes

Phase 2 already ran `origami/verify.py`'s root-scale gates (G18, G8/G8b, G6, G4) against the
merged root and would have raised if any failed, so nothing further is required. To re-check
a root later (e.g. after inspecting it), run the same gates manually:

```bash
python -m origami.verify all --root /mnt/big/origami/eef62_train --split train
```

`meta/origami_prep.json` at the merged root also records the exact `instruction`,
`urdf_sha256`, the full `locked_config` (arrays + digest), and the exact `total_frames`
converted (not the §1.1a estimate) — everything `train_origami.py`'s checkpoint metadata
(G1c, G14) needs to cross-check against.
