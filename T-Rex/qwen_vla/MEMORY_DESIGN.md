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

**B) Model-side (`modeling_qwen3vl_mot.py`, `modeling_vla.py`) -- SLOW TIER DONE (code+CPU tests), FAST TIER NOT STARTED**

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

**Fast tier -- not started**, and flagged as needing separate treatment:
`memory_fast` rows are NOT text+image content routed through
`prepare_inputs_embeds` -- they're wrist images + `action_abs` +
`tacf6_hist` that need the ACTION-expert embedding path (`x_embedder`,
`tacf6_embedder`/`encode_tactile_f6_history`), matching how live
`fast_embeds` get built and fed into `act_parts` inside
`forward_flow_action_full`/`_partial`. `tactile_flow_continue` needs no
direct change either way -- it receives `cached_kv` from
`forward_flow_action_partial`'s output, so slow memory is already baked in
transitively once that call is fed `memory_kv_slow`.

**C) Inference-side (`test.py` `CascadedServer`, `eval_trex_async.py`) -- NOT STARTED**

1. Replace `self.cached_kv = cached_kv` (`test.py:792`, clobbers every
   tick) with `self.memory_buf_slow: List[Tuple[float, DynamicCache]]` --
   timestamp + cropped snapshot, evicted by **time window**
   (`max(memory_slow_seconds) + margin`), not fixed count (ticks aren't
   uniformly spaced). Each new slow tick: for each target offset, pick the
   buffered entry nearest `now - target`, concatenate oldest->newest.
2. `self.memory_buf_fast`: fixed-count is fine here (linear window, fast
   ticks close enough together that count~=time).
3. Reset both on `reset_episode` (`test.py:626`).
4. Verify: is `CascadedServer` one instance per episode, or shared across
   concurrent episodes? If shared, memory needs per-episode-id keys.

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
- **Part B, slow tier: code done, CPU-only tests pass (no GPU available this
  session -- see below).**
  - `shift_position_ids_for_memory` + `build_memory_kv_slow` +
    `_front_pad_attention_mask` implemented as described above,
    `memory_kv_slow` wired into `forward_flow_action_full`/`_partial`.
  - Verified via `py_compile` (both files) and a new pure-logic CPU-only
    test suite (no real 2B model, no GPU): uniform per-batch position
    translation preserves each row's internal M-RoPE structure and lands
    at <=0; zero-offset is a true no-op; the mask front-pad is a true no-op
    when there's no memory or a zero-length cache, correctly prepends
    all-1s for the memory block otherwise, and preserves a real per-batch
    padding mask exactly at the tail; an end-to-end check against the real
    `build_causal_mask` confirms the intended semantics (full memory block
    always attendable, causal structure intact across the whole
    `[memory|latent|act]` axis). All 7 cases pass.
  - **NOT yet verified**: an actual forward pass through the real 2B model
    (needs GPU + real weights) -- this is pure math/plumbing verification
    only. `rope_stride` (RoPE position units per real second) is a
    placeholder default (32.0) and needs empirical calibration once a GPU
    is available, against however many position units a real live tick
    actually spans.
- **Part B, fast tier: not started** (needs the action-expert embedding
  path, not `prepare_inputs_embeds` -- see Design B above).
- **Part C: not started.**

## Critical files

- `T-Rex/qwen_vla/origami_dataset.py` (Part A -- done)
- `T-Rex/qwen_vla/lerobot_dataset.py` (Part A, LeRobot path -- not touched
  yet, only the origami-flat path above is done)
- `T-Rex/qwen_vla/modeling_vla.py` (Part B -- slow tier done:
  `build_memory_kv_slow`, `_front_pad_attention_mask`, `memory_kv_slow`
  param on `forward_flow_action_full`/`_partial`)
- `T-Rex/qwen_vla/modeling_qwen3vl_mot.py` (Part B -- slow tier done:
  `shift_position_ids_for_memory`)
- `T-Rex/scripts/test.py` (Part C)
- `T-Rex/hardware_code/eval/eval_trex_async.py` (Part C)
- `T-Rex/scripts/smoke_test_memory_windowing.py` (Part A CPU-only test)
- `T-Rex/scripts/smoke_test_memory_position_mask.py` (Part B slow-tier
  CPU-only test)
