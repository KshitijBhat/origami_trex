# Cross-timestep memory for T-Rex (branch: `dev/memory`)

## Context

T-Rex (the Qwen3-VL Mixture-of-Transformers VLA policy, `T-Rex/qwen_vla/`) has
zero memory across environment timesteps today: every inference tick is a
fully independent forward pass, and the model can't tell "the paper slipped
so I'm re-grasping" from "I'm grasping for the first time" -- this is the
first step toward a later failure-recovery/reasoning piece (a frozen critic
module that injects corrective text into `input_ids`; that needs the model to
actually have memory of what it just did, which is what this builds).

Explicitly out of scope here: a Video-Depth-Anything-style small per-layer
temporal attention module inserted into specific decoder layers (the "DPT
head" approach) -- deferred as a separate, more surgical follow-up. This is
specifically about extending the *existing* KV-cache mechanism across
timesteps, not adding a new module type.

Scope decisions:
- Memory spans all three streams -- latent/vision, action, AND tactile --
  not restricted to a cheaper action+tactile-only option.
- Memory rides the existing async slow/fast split already in
  `eval_trex_async.py`/`test.py`'s `CascadedServer`, with a real K-step
  window on **both** sides, rather than inventing a new rate structure.
- **The slow/fast split in this codebase is NOT "vision vs tactile."**
  Confirmed in `origami_dataset.py:487-491` (`pil_slow = [head]`,
  `pil_fast = [wrist_right, wrist_left]`) and `modeling_vla.py:901-950`'s
  `split_slow_fast_embeds` docstring ("slow (latent) and fast (action)
  portions"; line 29, `tactile_flow_continue` is explicitly "fast tick
  (tactile expert, lower segment)"). It's **head camera + text = slow**
  (the latent expert's infrequently-refreshed context) vs **wrist_left +
  wrist_right + action expert + tactile expert = fast** (recomputed
  together, frequently -- where a slip would actually show up, since the
  wrist cameras watch the grasp up close).
- **Slow memory uses exponential lookback in seconds, not a fixed row
  count.** Task-level context is useful much further back than it's useful
  densely -- curr + 0.25/0.5/1/5s ago (4 log-spaced past samples, config
  `memory_slow_seconds`) reaches 5s back for the same token cost 4
  *consecutive* rows would spend reaching only ~0.13s back at 30Hz. Fast
  (wrist+action+tactile) memory stays a short **linear** window
  (`memory_fast`, an int count) -- it's meant to catch a near-immediate
  reflex, where seconds of lookback is dead weight.
- **Train/inference timing mismatch, and the fix.** Training computes row
  offsets as `round(dt_seconds * sample_fps)` -- exact, because parquet data
  was captured at a known uniform rate. Inference ticks do NOT land on a
  uniform grid (compute/network jitter) -- a fixed-position buffer ("Kth
  most recent entry") would silently drift from the intended
  0.25/0.5/1/5s targets. Fix: the inference-side slow memory buffer must
  store `(timestamp, cache)` pairs capped by a **time window**, and each
  tick picks whichever buffered entry's timestamp is nearest to
  `now - target` per offset, not by position. To match this at training
  time, each target offset gets jittered before conversion to a row index:
  `dt_k = nominal_k + Uniform(-j, j)`, with `j` a FIXED absolute jitter (not
  a percentage of `dt_k`) -- the nearest-timestamp snap's error at inference
  is bounded by roughly half the tick interval, ~constant in absolute terms
  regardless of how far back the target is, so a percentage-based jitter
  would overshoot for the 5s target and undershoot for the 0.25s one.
  Training-time cost of this: none -- jitter only changes which row index
  gets read, still one parquet row + one JPEG decode per lookback step.

## Confirmed facts

- **Inference has no cross-timestep memory today.** Rollout loop:
  `eval_trex_async.py:844` (slow) + `:1019` (fast), over ZMQ to `test.py`'s
  `CascadedServer`. `CascadedServer._run_slow` (`test.py:778-796`) calls
  `forward_flow_action_partial` then does `self.cached_kv = cached_kv`
  (line 792), **clobbering** the previous slow tick's cache every time.
  `_run_fast` (`test.py:803-852`) reuses that cache via
  `_clone_dynamic_cache`, but only within one slow tick's chunk, never
  across slow ticks. `forward_flow_action_partial` (`modeling_vla.py:644`)
  and `forward_flow_action_full` (`modeling_vla.py:557`) both hardcode
  `past_kv = None` (lines 690, 596), no external-cache parameter.
  `tactile_flow_continue` (`modeling_vla.py:769`) DOES take a required
  positional `cached_kv` -- the closest existing precedent for the "seed
  with prior cache" API shape.
- **The low-level plumbing for cross-call KV already exists, no new
  weights needed.** `Qwen3VLModelMoT.forward` (`modeling_qwen3vl_mot.py:621-635`)
  already accepts `past_key_values` in/out. `Qwen3VLAttentionMoT`
  (`modeling_qwen3vl_mot.py:222-328`, `read_cached_kv` at 143-161) already
  reads/concats cached K/V generically. Its Q/K/V/O/MLP projections are
  sequence-length-agnostic. The real risk is distributional (train/inference
  mismatch), not architectural.
- **Training has no multi-timestep windowing except tactile.** Every
  example is one timestep (`origami_dataset.py` `__getitem__`): current-frame
  images only, one `action_chunk` (future horizon, not memory), one
  `tacf6_hist` (the only genuine past window, tactile-only). Episode
  boundaries respected in indexing, but batches are NOT temporally ordered
  (`BlockShuffleSampler` interleaves episodes at random) -- so memory MUST
  be built inside a single `__getitem__` call, not via batch-to-batch
  adjacency.
- **Row granularity varies per dataset -- do not hardcode.** `src_fps=30.0`
  is fixed, but `sample_stride` is read from that dataset's own prep config
  (`origami_dataset.py:188`, `cfg.get("sample_stride", 1)`). Verified two
  different real values in practice: `prepare.py`'s own default is 5
  (-> 6Hz), but the actual `drakedrake/origami_preprocessed_stride_1` HF
  dataset used for testing was prepared with `sample_stride=1` (-> native
  30Hz) and `action_chunk=16` (not the file's own default constant of 25).
  Always read `self.sample_fps`, never assume a rate.
- **Token budget**: head image alone ~64 tokens (slow side); wrist_left +
  wrist_right together ~128 tokens (fast side, plus action/tactile tokens).
  A K_slow-step slow memory window costs ~64*K_slow tokens (cheap -- 4 steps
  is only +192 tokens). Fast memory is more expensive per step, which is
  fine since that's the side real memory matters most for.

## Design

**A) Training-side (`origami_dataset.py`) -- DONE, verified (see Progress)**

New config: `memory_slow_seconds` (list of floats, e.g. `[0.25, 0.5, 1.0,
5.0]`, empty = off), `memory_slow_jitter_sec` (float, absolute jitter, 0 =
off), `memory_fast` (int count, 0 = off).

In `__getitem__`:
- Slow: for each `dt` in `memory_slow_seconds` (processed furthest-first for
  oldest->newest ordering), jitter it (`dt + Uniform(-j, j)`, clamped >= 0),
  convert to a row offset via `round(dt * self.sample_fps)`, clamp at
  episode start (`max(row - offset, 0)`, same left-pad convention
  `tacf6_hist` already uses), read that row's **head image + task text
  only**. Returns `item["memory_slow"] = [...]`.
- Fast: for `k` in `1..memory_fast` (consecutive, linear), same
  clamp-at-episode-start convention, read that row's **wrist_left +
  wrist_right + action_abs + tacf6_hist**. Returns
  `item["memory_fast"] = [...]`.
- Neither depends on batch ordering -- built entirely inside one
  `__getitem__` call.

In `collate_fn`: each past row (for both tiers) gets the same
`apply_chat_template` + Qwen processor treatment the live row already gets
(not the simpler raw-pixel path `flare` uses, since each memory row needs
its own tokenized sequence to be forwarded separately later). Output:
`memory_slow` / `memory_fast` = `None` when off, else a list of per-step
dicts with batched tensors (`input_ids`, `attention_mask`, `pixel_values`,
`image_grid_thw`, plus `action_abs`/`tacf6_hist` for the fast tier).

**B) Model-side (`modeling_qwen3vl_mot.py`, `modeling_vla.py`) -- SLOW + FAST TIERS DONE, verified on real GPU with real weights (2026-09-18)**

**Decided: raw per-token KV reuse (zero new weights), not latent-compressed
memory.** Considered compressing each past row into a small number of
learned "memory slot" latents (mirroring `flare_queries`' existing
cross-attention-compression pattern, just consuming the past instead of
predicting the future) -- would directly fix the fast-tick token-cost risk
below (K_slow=4 rows x ~64 head tokens = 256 raw KV tokens/layer could
shrink 8-16x to ~16-32 compressed slots), but needs a new trainable
component trained end-to-end through the action-prediction loss. Decided to
ship the simpler, already-mostly-designed zero-new-weights plumbing first;
revisit compression later if the token cost proves to matter in practice
once real numbers are measured (see Risks).

**Position-id design, refined after tracing where `position_ids` actually
comes from (`train.py:869`, `raw_model.get_rope_index(input_ids,
image_grid_thw, attention_mask)`).** FLARE turned out NOT to be a usable
template for this, despite being the closest existing "extra temporal
content" precedent: `flare_queries` are learned placeholder query tokens (no
real image patches), so `_extend_position_ids`'s flat scalar-per-token
extension is fine for them. Memory rows are different -- they carry real
multi-patch image content (head image for slow, wrist images for fast) that
needs genuine internal 2D spatial (height/width) M-RoPE structure, not a
flat scalar per token. Correct approach: for each memory row, call
`get_rope_index` independently on *that row's own* `input_ids`/
`image_grid_thw` (exactly as the live row already does), preserving correct
internal spatial structure, then shift the *entire resulting block*
backward by an amount proportional to that row's `dt_k` (slow) or `k` (fast,
uniform), landing it at **negative** positions relative to the current
step's 0-baseline. Mathematically sound -- rotary embeddings work fine with
negative angles, and it's the correct semantic for "before now" on a
monotonic time axis. This replaces the original "flat `k*S` for the whole
row" idea from the initial design pass.

**Slow tier -- implemented:**

1. `Qwen3VLModelMoT.shift_position_ids_for_memory(position_ids, time_offset)`
   (staticmethod, `modeling_qwen3vl_mot.py`, next to `_extend_position_ids`):
   takes one memory row's own fresh `get_rope_index()` output (`[3,B,L]`,
   0-based -- confirmed by reading the real HF `get_rope_index` source
   directly: every fresh call starts `current_pos` at 0 per batch item) and
   a per-batch `time_offset: [B]` **already in RoPE position units**
   (`dt_actual_seconds * rope_stride`), returns the block translated
   backward by that amount. Pure uniform subtraction -- internal 2D
   spatial structure within the row is untouched, only the whole block's
   baseline moves. `rope_stride` is a still-unvalidated hyperparameter
   (needs empirical tuning against how many position units a live tick
   actually spans -- deferred to GPU time, see Progress).
2. `Qwen3VLVLAModel.build_memory_kv_slow(memory_rows, rope_stride=32.0)`
   (`modeling_vla.py`, next to `_clone_dynamic_cache`): loops the K
   `memory_slow` rows oldest-first, for each: `prepare_inputs_embeds` +
   `get_rope_index` (both existing helpers, reused as-is) + the shift
   above, then one `self.model(...)` forward with `use_cache=True`,
   `latent_indexes=arange(0,L)` (memory_slow rows are homogeneously
   head+text content by construction -- no `split_slow_fast_embeds` call
   needed, unlike the live row), threading `past_key_values` across
   iterations so the K rows accumulate into one `DynamicCache`. Returns
   `None` for an empty list (matches today's default-off behavior exactly).
3. **Mask-alignment bug found and fixed before it could bite silently:**
   `build_causal_mask` always pads a shorter `attention_mask` at the TAIL
   of the `(past_len + seq_len)` axis. That convention is correct for this
   codebase's *existing* past-KV use (denoising steps: cached content is
   `[live latent (described by attention_mask) | previously-appended action
   tokens (always real)]` -- attention_mask already sits at the front of the
   cache). Memory KV breaks it: the cache becomes `[memory (always real) |
   live latent (described by attention_mask) | action tokens]`, so a
   naively-reused `attention_mask` would end up describing the *memory*
   block instead of the live latent block. Fixed with
   `Qwen3VLVLAModel._front_pad_attention_mask(attention_mask, batch_size,
   seq_len, past_kv, device)`: front-pads with 1s (memory is never padding)
   by `past_kv`'s length, called **once** before the denoising loop starts
   in both `forward_flow_action_full`/`_partial` (not per-iteration --
   `past_kv.crop(-n_act)` always restores `past_len` back to exactly
   `memory_len + latent_len` between iterations, so one pre-pad is
   sufficient and keeps every iteration's reused `attention_mask` correct).
4. `forward_flow_action_full`/`forward_flow_action_partial` gained a
   `memory_kv_slow: Optional[DynamicCache] = None` param. Loop init changed
   from `past_kv = None` to `past_kv = memory_kv_slow`; the "first
   iteration does a full latent+action forward, later ones reuse KV"
   branch now keys off an explicit step index (`step_idx == 0` / `i == 0`)
   instead of `past_kv is None`, since `past_kv` now legitimately starts
   non-None when memory is supplied. `position_ids` for the live row is
   passed through completely unchanged (RoPE handles the live-vs-memory
   relative distance automatically via the position values themselves;
   only memory's own position_ids needed shifting). Default (`None`) is a
   strict no-op, verified by the mask/position CPU tests below reducing to
   identity in that case.

**Fast tier -- implemented:**

`memory_fast` rows are NOT text+image content routed purely through
`prepare_inputs_embeds` the way `memory_slow` rows are -- their images
(wrist_right + wrist_left) go through that same helper for the vision
part, but the row also carries `action_abs` (that row's executed action)
and optionally `tacf6_hist` (raw tactile window), which need the
ACTION/TACTILE embedding paths (`x_embedder`, `_embed_tactile_observations`
-- the existing helper `tactile_flow_continue`/`tactile_flow_train_step`
already use for live tactile embedding).

`Qwen3VLVLAModel.build_memory_kv_fast(memory_rows, past_kv=None,
rope_stride=8.0)` (`modeling_vla.py`, next to `build_memory_kv_slow`):
loops the memory_fast rows oldest-first, for each:
1. `prepare_inputs_embeds` on that row's wrist-image `input_ids`/
   `pixel_values`/`image_grid_thw` -> `wrist_embeds`.
2. `get_rope_index` on the same -> that row's own fresh 0-based
   position_ids (2D spatial structure for the wrist images).
3. `x_embedder(action_abs.unsqueeze(1))` -> one action token
   (`n_action=1`) -- the row's actually-executed past action, not a noisy
   sample (there's no flow-matching timestep for a past, already-resolved
   tick).
4. `_embed_tactile_observations(..., tactile_f6_history=tacf6_hist)` when
   present AND the model has an on-the-fly VQ-VAE (`use_tactile_vqvae`) --
   otherwise tactile is silently skipped for that row (known limitation:
   `tacf6_hist` is a history window shaped for the VQ-VAE path only, not
   the plain-vector `tactile_f6` path some configs use instead).
5. Content is concatenated as `[wrist | action | tactile]` -- this order
   is not arbitrary: it lets `Qwen3VLModelMoT._extend_position_ids`'s
   existing `(n_action, n_tactile)` convention build correct position_ids
   for the appended tokens directly, no new position-extension logic
   needed beyond what slow tier already added.
6. The extended position block is shifted by `shift_position_ids_for_memory`
   using `(K - i) * rope_stride` for row `i` of `K` (oldest first) -- fast
   memory has no per-row `dt_actual` in the dataset (it's a short LINEAR
   window, ticks assumed uniformly spaced -- see Context), unlike slow
   memory's exponential/jittered real-time offsets.
7. `past_kv` threads across iterations exactly like `build_memory_kv_slow`
   -- critically, **`build_memory_kv_fast` is meant to be called with
   `build_memory_kv_slow`'s output cache as its own `past_kv` input**, so
   both tiers land in ONE combined cache: `[slow memory | fast memory |
   live tokens]`. That combined cache is what gets passed as
   `forward_flow_action_full`/`_partial`'s `memory_kv` argument (renamed
   from `memory_kv_slow` now that it holds both tiers -- no other caller
   existed yet to break).

`tactile_flow_continue` needs no direct change either way -- it receives
`cached_kv` from `forward_flow_action_partial`'s output, so both memory
tiers are already baked in transitively once that call is fed `memory_kv`.

**C) Inference-side (`test.py` `CascadedServer`) -- DONE, verified via a
CPU-only stubbed-model/processor test (real GPU/checkpoint integration test
still open, see Progress).**

**Design correction made while implementing (important, changes what's
actually stored vs. the original plan above): the buffers hold RAW
CONTENT (PIL images + text / action + timestamp), NOT pre-computed
`DynamicCache` snapshots.** Confirmed by reading `Qwen3VLAttentionMoT.forward`
directly: `apply_rotary_pos_emb_1d` runs on Q/K BEFORE `cache.update()` --
RoPE rotation is baked into a cached K the moment it's stored, permanently.
A KV snapshot built once therefore cannot be "re-shifted" later for a
different tick's `now` without recomputing from scratch -- which is exactly
what `build_memory_kv_slow`/`_fast` already do (fresh forward pass every
call, using that call's own `dt_actual`). So Part C's buffers store what to
FEED those methods, mirroring what `OrigamiDataset` keeps per episode, and
recompute the memory KV fresh on every slow tick (not every fast tick --
`_run_fast`'s `tactile_flow_continue` inherits it for free via `cached_kv`,
already proven on GPU in Part B's verification, so memory-KV construction
cost is bounded to the slow-tick cadence, not the faster fast-tick one).

1. `self.memory_buf_slow: List[(timestamp, PIL.Image head, str task)]`,
   evicted by **time window** (`max(memory_slow_seconds) +
   memory_buffer_margin_sec`), not fixed count (ticks aren't uniformly
   spaced). `_select_memory_slow_rows(now)`: for each configured target
   (oldest first), the buffered entry whose timestamp is nearest
   `now - target` (nearest-timestamp snap, not positional), tokenized
   fresh via the same `apply_chat_template`+processor path the live row
   uses, `dt_actual = now - that_entry's_real_timestamp` (the REAL gap,
   matching training's `dt_actual` semantics exactly, including
   irregular-tick drift from the nominal target).
2. `self.memory_buf_fast: List[(timestamp, [wrist_right, wrist_left],
   action_raw)]`, fixed-count eviction (linear window, fast ticks close
   enough together that count~=time -- matches training). `action_raw` is
   `self._prev_command(state)` (the server's existing best-estimate of
   "what did we just tell the robot to do" -- already used for the old
   anchor reconstruction, repurposed here since it's exactly the right
   concept for "the executed action at a past tick"), normalized via the
   SAME `_normalize`/`statistic["action_min/max"]` the live tick's
   `state_embeds` and training's `norm_actions` use (see the collate_fn
   normalization bugfix below -- this server-side path needed the matching
   fix). `_select_memory_fast_rows()`: the last `memory_fast` entries,
   oldest first, each tokenized via chat template + processor (wrist
   images only, no text -- matches `origami_dataset.py`'s memory_fast
   convention).
   - **Known limitation, matching the model-side one**: `tacf6_hist` is
     NOT captured for server-side fast memory. `_rolling_f6_window`
     (single-frame client input) has a side effect -- it appends to
     `self.f6_buffer` -- and `_run_fast` already calls it once per real
     request; calling it again from `_run_slow` to snapshot a memory row
     would double-append the same tick's frame into the rolling window,
     corrupting it. Fixing this properly needs restructuring so the F6
     window is computed once per request and threaded to both call sites,
     deferred rather than risking a subtle rolling-buffer corruption bug.
3. `_build_memory_kv(now)`: calls `build_memory_kv_slow` on the selected
   slow rows, then feeds ITS OUTPUT as `build_memory_kv_fast`'s `past_kv`
   -- both tiers land in one combined cache, passed as `_run_slow`'s
   `memory_kv` argument to `forward_flow_action_full`/`_partial`. Returns
   `None` (true no-op) when both tiers are off or nothing is buffered yet
   (e.g. an episode's first slow tick).
4. Ordering in `_run_slow`: `memory_kv = self._build_memory_kv(now)` is
   computed FIRST (from prior ticks' buffered content only), and this
   tick's own content is recorded into the buffers (`_remember_slow`/
   `_remember_fast`) only AFTER -- so a tick can never see itself as its
   own memory. `_run_fast` needs no changes at all (memory reaches it
   transitively via `cached_kv`, exactly as designed in Part B).
5. Both buffers reset in `reset_episode` alongside the existing phase
   clock / last-command chain reset -- memory must not leak across fold
   attempts, matching the training-side index never crossing an episode
   boundary.
6. `CascadedServer` confirmed to be a single shared instance (one ZMQ REP
   socket, `self.lock` serializes all calls, single active session) -- not
   per-episode, so no per-episode-id keying is needed; `reset_episode`
   alone is sufficient (resolves the "unverified" item from the original
   plan).
7. New CLI flags (all default off/0, strict no-op): `--memory_slow_seconds`
   (comma-separated string), `--memory_fast` (int),
   `--memory_rope_stride_slow` / `--memory_rope_stride_fast` (must match
   whatever the checkpoint was actually trained with once Part D exists),
   `--memory_buffer_margin_sec`.

**Separate, pre-existing bug found and fixed while wiring this in (not a
memory bug, but it blocked even importing `test.py` to test Part C):**
`CascadedServer` still imported `trex_origami.anchoring`, a module deleted
in commit `227b516` ("Strip anchoring... for the all-absolute rewrite") --
present on `dev/memory` AND its parent `dev/all_absolute` alike, so this
wasn't introduced by this branch. That commit's own message names
`policy.py`/`verify.py`/`preflight.py` as "still coupled to anchoring, not
yet addressed" but missed `test.py`, which was in the identical broken
state -- the production inference server could not be imported at all,
independent of memory. Per explicit user direction, rewrote
`CascadedServer`'s reconstruction to match the all-absolute target instead
of restoring the deleted module: `_reconstruct` now just denormalizes the
model's output directly (no per-dim anchor addition -- the model already
predicts absolute actions), `_prev_command`/`self.last_chunk` tracking is
kept (repurposed as the memory buffer's fast-tier `action_abs` source,
see above) but decoupled from the deleted anchor-spec machinery,
`--anchor_source` CLI flag removed (no longer meaningful), `--action_output
delta` kept as a documented legacy escape hatch for pre-all-absolute
checkpoints. Verified by actually importing the module locally
(`from test import CascadedServer`) -- confirmed broken before the fix,
confirmed working after.

**Separate normalization bug found and fixed while designing Part C's
fast-memory action proxy:** `origami_dataset.py`'s `collate_fn` built
`memory_fast`'s `action_abs` from the RAW (unnormalized) dataset field,
but `build_memory_kv_fast` feeds it straight into `x_embedder`, which
(via the live tick's `norm_actions`/flow-target construction,
`origami_dataset.py:526-527`) only ever sees NORMALIZED `[-1,1]`-ish
values elsewhere -- a real train/inference input-distribution mismatch,
not just a style inconsistency, that a random-weight GPU smoke test
couldn't have caught (random Gaussian noise looks equally plausible
normalized or not). Fixed by normalizing `action_abs` in `collate_fn` the
same way `norm_actions` is, using the dataset's own `action_mask`/
`action_min`/`action_max`. Verified with a dedicated regression test
(`gpu_smoke_test_memory_fast_normalization.py`, real processor, raw
values with amplitude ~3.0 deliberately far outside `[-1,1]`) confirming
normalized output actually lands in `[-1,1]` post-fix.

## Risks

- **Fast-tick compute isn't free, just cheaper.** Skipping vision/action
  *recomputation* on a tactile-only fast tick (confirmed: length-0
  `latent_indexes`/`action_indexes` skip those projections) does NOT skip
  attending over the cached KV already in the sequence -- per-fast-tick
  attention cost grows linearly with accumulated slow-memory tokens.
  Measure real wall-clock before committing to `K_slow`/`K_fast` values.
- **Backward compatibility**: all new params default `None`/`0`/`[]`, no
  new `nn.Parameter`/buffers -- existing checkpoints load unchanged.
- **Train/inference mismatch**: `memory_slow_seconds`/`memory_fast` and the
  padding/jitter convention must match between training and serving, or
  RoPE offsets and attention statistics go out-of-distribution.
- **Unverified before Part B/C can be finished**: real `command_hz`/
  `fetch_hz` slow/fast rate values (not found in this checkout's configs);
  whether `CascadedServer` is one-instance-per-episode or shared.

## Progress so far

- **Part A: done, verified twice.**
  - Local CPU-only synthetic tests (episode-boundary padding, exponential
    offset math, jitter distribution/bounds, `memory_slow_seconds=[]`/
    `memory_fast=0` true no-op) -- all pass.
  - Real-data smoke test on a rented GPU box: real `Qwen/Qwen3-VL-2B-Instruct`
    processor + 2 real episodes from `drakedrake/origami_preprocessed_stride_1`
    (this dataset: `sample_stride=1`, `action_chunk=16` -- NOT the file's own
    defaults, confirming the "don't hardcode" warning above). Both
    memory-off (regression, byte-identical structure to current behavior)
    and memory-on (`memory_slow_seconds=[0.25,0.5,1,5]`, `memory_fast=2`,
    `memory_slow_jitter_sec=0.1`) pass with correct tensor shapes throughout.
  - Follow-up fix (post-commit, same session): added `dt_actual` (the REAL
    elapsed time of the row actually fetched, post-jitter, post-episode-
    clamp) to both `__getitem__` and `collate_fn` -- needed because episode-
    start clamping can make the real gap disagree with the nominal jittered
    target, and Part B's position shift must reflect what the content
    actually is. New test case in the local CPU-only suite confirms clamped
    rows report the correct real gap (not the nominal target).
- **Part B, slow + fast tiers: code done, verified both locally (CPU-only)
  and on a real GPU with real weights (see below).**
  - Slow: `shift_position_ids_for_memory` + `build_memory_kv_slow` +
    `_front_pad_attention_mask`, `memory_kv` wired into
    `forward_flow_action_full`/`_partial` (param renamed from
    `memory_kv_slow` now that it holds both tiers combined -- no other
    caller existed yet to break).
  - Fast: `build_memory_kv_fast` (wrist images via `prepare_inputs_embeds`
    + past action via `x_embedder` + tactile via
    `_embed_tactile_observations`, concatenated `[wrist|action|tactile]`,
    positions via `_extend_position_ids` + `shift_position_ids_for_memory`
    with a `(K-i)*rope_stride` linear offset), meant to be chained after
    `build_memory_kv_slow` (fed its output cache as `past_kv`) so both
    tiers land in one combined cache.
  - Verified via `py_compile` (both files) and a pure-logic CPU-only test
    suite (no real 2B model, no GPU -- `T-Rex/scripts/
    smoke_test_memory_position_mask.py`, 10 cases, all passing): the slow-
    tier position/mask cases from before, plus fast-tier-specific cases --
    `_extend_position_ids` + `shift_position_ids_for_memory` compose
    correctly for the `[wrist|action|tactile]` layout (relative ordering
    survives the uniform shift), the `(K-i)*rope_stride` offset formula is
    monotonically decreasing oldest->newest, and `_front_pad_attention_mask`
    chains correctly across two sequential accumulation steps (simulating
    slow-then-fast growing one combined cache).
  - **GPU verification (2026-09-18, real hardware, real weights):**
    `T-Rex/scripts/gpu_smoke_test_memory.py` builds a real
    `Qwen3VLVLAModel` via `from_pretrained_qwen3vl("Qwen/Qwen3-VL-2B-
    Instruct", ...)` (real pretrained text+vision weights; the new MoT
    expert/VLA-specific weights -- `x_embedder`, `tacf6_embedder`,
    `tactile_code_embedder`, etc. -- are randomly initialized, so this
    verifies wiring/shapes/numerics, not trained policy quality), a real
    `AutoProcessor`, and a live tick + `memory_slow`/`memory_fast` rows
    built through the same `apply_chat_template`+processor path
    `collate_fn` uses. Ran on an A100-40GB. Results: `build_memory_kv_slow`
    and `build_memory_kv_fast` both populate real caches without error
    (seq_len 156 slow-only, 442 combined slow+fast); `memory_kv=None`
    (regression) is finite and byte-reproducible across repeated calls,
    for both `forward_flow_action_full` and the cascaded
    `forward_flow_action_partial`+`tactile_flow_continue` path;
    `memory_kv=<slow+fast>` is finite for both paths and MEASURABLY changes
    the output vs. the no-memory baseline (max abs diff 0.775 for the full
    flow, 0.498 for the cascaded/tactile path) -- confirming the memory
    tiers are genuinely wired into attention, not a silent no-op, and that
    `tactile_flow_continue` really does inherit both tiers transitively
    through `cached_kv` (it never receives `memory_kv` directly, as
    designed). All 9 checks (7 explicit + 2 no-op checks) pass.
    - Two test-script setup bugs found and fixed along the way (both in the
      test script, not the memory-feature code): (1) `vqvae_config={}`'s
      default `codebook_size=1024` didn't match the model's own
      `vqvae_codebook_size=64` `tactile_code_embedder` table, so the
      randomly-initialized VQ-VAE could emit out-of-range codes -> CUDA
      `indexSelectSmallIndex` assert; fixed by passing
      `vqvae_config={"codebook_size": 64}` to match. (2)
      `from_pretrained_qwen3vl`'s `torch_dtype` arg only covers weights
      loaded from the pretrained base model -- the new VLA-specific
      modules are constructed at default float32 and need an explicit
      `model.to(torch.bfloat16)` afterward (confirmed this is exactly what
      the real production loader, `trex_origami/loading.py`, already does)
      -- fixed by adding that cast to the test script.
    - Separate, pre-existing, unrelated finding while setting up the GPU
      box: a bare `pip install transformers` pulls whatever is newest
      (5.16.1 / 5.17.0 both tried), which breaks even plain
      `Qwen3VLVLAModel` construction -- `Qwen3VLTextRotaryEmbedding.__init__`
      in current transformers reads `config.rope_parameters`, which this
      repo's `Qwen3VLRotaryEmbeddingWrapper`'s `_RopeCfg` shim
      (`modeling_qwen3vl_mot.py`) does not set (it was written against an
      older transformers API using `rope_scaling`/`rope_theta` directly).
      This repo's own `pyproject.toml` already correctly pins
      `transformers==4.57.3` -- the fix was simply installing that exact
      pin rather than latest. Not a memory-feature bug and nothing was
      changed in the shipped code for it, but worth remembering: any
      fresh GPU box for this repo needs `transformers==4.57.3` specifically,
      not "whatever's current."
    - `rope_stride` for both tiers (slow default 32.0, fast default 8.0)
      remain UNTUNED placeholders -- this session confirmed they produce
      finite, sane-shaped, non-degenerate output, not that the specific
      values are good ones. Empirical tuning (e.g. sweeping rope_stride and
      checking downstream loss/behavior once a trained checkpoint exists)
      is still open.
  - **Known limitation**: fast-tier tactile memory only supports the
    on-the-fly VQ-VAE path (`use_tactile_vqvae=True`) -- a row with
    `tacf6_hist` but no VQ-VAE silently skips tactile for that memory row
    rather than raising, since the history-window shape doesn't match the
    plain-vector `tactile_f6` path some configs use instead. Not yet
    needed by any tested config.
  - **`memory_fast` normalization bug found and fixed** (see Design C
    above for detail): `action_abs` was unnormalized before hitting
    `x_embedder`. Fixed in `origami_dataset.py`'s `collate_fn`; verified
    with a dedicated real-processor regression test.
- **Part C: done, verified via a CPU-only stubbed test
  (`T-Rex/scripts/smoke_test_cascaded_server_memory.py`, 8 cases, all
  passing against the real `test.py` -- nearest-timestamp slow selection
  over irregular ticks, time-window eviction, linear fast-window eviction,
  slow->fast KV chaining, a tick never seeing its own content as memory,
  `reset_episode` clearing both buffers, and true no-ops when memory is
  off or the buffer is still empty).**
  - Along the way, found and fixed (per explicit user direction) a
    separate pre-existing bug that blocked even importing `test.py`: a
    dead `trex_origami.anchoring` import left over from an incomplete
    "strip anchoring for the all-absolute rewrite" migration, present on
    both `dev/memory` and its parent `dev/all_absolute`. Rewrote
    `CascadedServer`'s action reconstruction to match all-absolute
    (denormalize only, no anchor addition) instead of restoring the
    deleted module -- see Design C above for the full list of what
    changed (`_reconstruct`, `_prev_command`, `--anchor_source` removed,
    `--action_output delta` kept as a legacy escape hatch). Verified by
    actually importing the module (confirmed broken before, working
    after) -- this was a real blocker for the ENTIRE inference server,
    independent of memory.
  - **NOT yet verified**: a real end-to-end run with an actual trained
    checkpoint (`CascadedServer.__init__` + `predict()` against real
    `training_args.json`/`processor/`/`model.pt`/`stats_data.json`). No
    checkpoint with memory support exists yet -- blocked on Part D
    (training-loop wiring, see below), not on anything in Part C itself.

## Part D: training-loop wiring -- NOT STARTED, discovered while doing Part C

`origami_dataset.py`'s `collate_fn` produces `memory_slow`/`memory_fast`
keys (Part A), and the model has `build_memory_kv_slow`/`build_memory_kv_fast`
plus the `memory_kv` parameter on `forward_flow_action_full`/`_partial`
(Part B) -- but **`T-Rex/scripts/train.py`'s actual training loop never
calls any of this**. It doesn't build a memory KV from the batch's
`memory_slow`/`memory_fast` entries, and doesn't pass `memory_kv` into its
forward call. Today, turning `memory_slow_seconds`/`memory_fast` on in
training would compute the extra dataset fields (image decode + tokenize
for each memory row, on every batch) and then silently throw them away --
no gradient ever flows through the memory path, so no checkpoint has
learned to use it. `run_all_abs_ori.sh`/`run_trex_job.sh` also have no
memory-related CLI flags, for the same reason -- there's nothing to wire
them to yet.

This is required before Part C can be verified end-to-end (needs a real
trained checkpoint) and before memory can do anything useful at all.
Scope: add CLI args (`--memory_slow_seconds`, `--memory_fast`,
`--memory_rope_stride_slow/fast`) to `train.py`'s argparse and
`run_all_abs_ori.sh`'s env-var pass-through, thread them into the
`OrigamiDataset` construction (already supported, Part A), and in the
training forward pass build `memory_kv` from `batch["memory_slow"]`/
`batch["memory_fast"]` via the same `build_memory_kv_slow`/
`build_memory_kv_fast` chaining Part C's server uses, passing it into
whatever training calls `forward_flow_action_full`/`_partial`
(`train.py`'s loss-computation forward, not yet located/read in detail).

## Critical files

- `T-Rex/qwen_vla/origami_dataset.py` (Part A -- done)
- `T-Rex/qwen_vla/lerobot_dataset.py` (Part A, LeRobot path -- not touched
  yet, only the origami-flat path above is done)
- `T-Rex/qwen_vla/modeling_vla.py` (Part B -- slow+fast tiers done:
  `build_memory_kv_slow`, `build_memory_kv_fast`,
  `_front_pad_attention_mask`, `memory_kv` param on
  `forward_flow_action_full`/`_partial`)
- `T-Rex/qwen_vla/modeling_qwen3vl_mot.py` (Part B -- done:
  `shift_position_ids_for_memory`)
- `T-Rex/scripts/test.py` (Part C -- done: `_build_memory_kv`,
  `_select_memory_slow_rows`, `_select_memory_fast_rows`, `_remember_slow`,
  `_remember_fast`, `memory_kv` wired into `_run_slow`; also carries the
  unrelated anchoring-removal rewrite that was blocking imports entirely)
- `T-Rex/hardware_code/eval/eval_trex_async.py` (not touched -- the ZMQ
  client sends the same payload shape as before; memory is entirely
  server-side state, no protocol change needed)
- `T-Rex/scripts/train.py` (Part D -- not started, the actual gap: never
  calls `build_memory_kv_slow`/`_fast` or passes `memory_kv`)
- `T-Rex/scripts/run_all_abs_ori.sh`, `T-Rex/run_trex_job.sh` (Part D --
  no memory CLI flags yet, nothing to wire them to until train.py is done)
- `T-Rex/scripts/smoke_test_memory_windowing.py` (Part A CPU-only test)
- `T-Rex/scripts/smoke_test_memory_position_mask.py` (Part B CPU-only test,
  slow + fast tier pure-logic cases)
- `T-Rex/scripts/gpu_smoke_test_memory.py` (Part B real-GPU/real-weights
  test -- needs a GPU box + `transformers==4.57.3` per pyproject.toml)
- `T-Rex/scripts/gpu_smoke_test_memory_fast_normalization.py` (Part A/B
  regression test for the `action_abs` normalization bugfix -- needs
  network + `AutoProcessor.from_pretrained`, no GPU compute required)
- `T-Rex/scripts/smoke_test_cascaded_server_memory.py` (Part C CPU-only
  test, stubbed processor/model -- no GPU or network needed)
