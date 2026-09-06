# Fresh origami branch — plan

Goal: stay as close to original T-Rex (`/home/sai/Desktop/ORI/T-Rex`) as reasonably possible,
while cutting origami_trex-fork-specific complexity — starting with anchoring — and defining a
dataset-prep approach for the real 65D origami robot that's evidenced, not guessed.

## Status

Branch `dev/all_absolute` (off `dev/attempt2_server` @ `641b189`). Commits so far: plan doc;
anchoring.py + Zenoh stack deleted; `prepare.py`/`prepare_fast.py` restored to pre-anchoring form
(`85ab48e`) and converted to all-absolute (`action_chunk = action[t+k]`, no delta) with
`action_chunk` changed 25→16; season lists moved from ~130 hardcoded lines in `seasons.py` into
`splits_main.json`/`splits_competition_paper_set.json` (one per HF revision, `--revision` flag
added to both prepare scripts).

**Smoke-tested end to end and verified correct** on the real local season
(`season_POC22061_2026_07_09_16_23_46_train`, on `main`'s train split), `sample_stride=3`, both
`--split train` and `--split val` code paths (same season used for both — no val season available
locally, no HF token to fetch one, clearly labeled as a substitution, not a real val run): 14
episodes, 30,451 samples each, ~1.5-4 min on CPU-only. Verified in the actual output data, not
just "it ran": `action_chunk[0]` exactly equals `action_abs`, and is close-but-distinct from
`state` (a real tracking offset, not an accidental collapse to delta-like behavior);
`meta/dataset.json`'s config has zero anchor-related fields.

**Also done since**: `--instruction` is now a required CLI flag on both prepare scripts (no more
silent default, was "fold the paper into a paper airplane" which `DEPLOY.md` found underperforms),
recorded once in `meta/dataset.json`'s top-level config, not per-row. `origami_dataset.py` (the
training loader) stripped of anchoring entirely -- `action` is exactly `action_chunk`, no
reconstruction; `frozen_action_dims`/`clamp_frozen_absolute` kept (they were already generic,
mask-driven) but now fed by the norm-stats variance mask instead of an anchor spec, computed as
`self.frozen_dims` and printed at startup. `action_eval`/`prev_command` kept as plain aliases so
`collate_fn`'s eval-only extras don't `KeyError` -- real semantics land with the eval rewrite.
Verified with a standalone import (bypassing an unrelated missing `cv2` dep), no leftover
references to the deleted module.

**Full training-relevant pipeline is now anchoring-free and verified end to end**:
`preflight.py` -- `check_anchoring` removed entirely (nothing left to check under all-absolute;
`check_frozen_dims` needed no changes, was already generic). Ran against the real smoke-test
output: correctly reports its two *expected* gaps (stand-in val season, `stats.py` never run) with
no import crash. `verify.py` -- gate 6 (was "anchor + action_chunk reconstructs action_abs")
simplified to a direct `action_chunk[0] == action_abs` check; gate 7's src-root round-trip drops
the anchor subtraction (`want_chunk = action_all[idx]`, not `- build_anchor(...)`). Ran against
the real smoke-test output **with `--src-root` pointed at the actual source data**: reports
"round-trip against raw season ... exact" -- a genuine confirmation against ground truth, not just
"imports without crashing."

That covers the full training path (prepare → loader → preflight → verify). `policy.py` (deploy)
is untouched on purpose -- explicitly staged for after training works, per your own note.

**Still not done, in rough order**: streaming/reservoir q01/q99 stats for full 126-season scale
(`stats.py` not yet checked for whether it already handles this or would OOM). `final_layer` init
script (hand warm-start + fresh arms, §6). VQ-VAE buffer re-fit script (§7). Training-time logging
beyond the frozen-dims print already added (confirmed in scope, e.g. recording which flags
actually got applied at launch). `accel.py`'s NVDEC improvements from `dev/attempt2_server`'s tip
were kept as-is (already anchor-free) but not yet diffed line-by-line against `85ab48e` to confirm
nothing else worth keeping was missed in `prepare.py`/`prepare_fast.py` specifically -- worth a
real diff pass before the full run. `policy.py` still imports the deleted `anchoring.py` --
deferred on purpose, not a bug.

Two things below are flagged as **open decisions**, not settled — I have evidence and a leaning
on each, but they're value calls that are yours to make, not mine to assume.

**Deliberately separate from the parallel `dev/redesign` effort** (`someother_REDESIGN_PLAN.md`,
`someother_IMPLEMENTATION_PROMPT.md`): that plan converts origami's data into original T-Rex's
own 62D EEF-pose format via forward kinematics and imports `T-Rex/` completely unmodified as a
submodule. Ours stays in origami's **native 65D raw-joint space** and adapts original's code
*style* (plain, hardcoded, no generic abstraction layers) rather than converting the
*representation*. Confirmed as the deliberate choice, not a gap to reconcile — see that plan for
the EEF-62 alternative if it's ever worth revisiting.

**Scope confirmed**: the full 126-season split (101 train / 25 val, ≈11.8M frames, ≈109h — per
the other plan's own accounting of the hub, which is real data worth trusting) will be used, not
just the single local season analyzed so far. This changes the sample_stride/epoch-budget
discussion in §5 from "our one season" to real scale — revisit those numbers before committing.

**Planned next phase (not this plan): historical-state representation.** Once the native-65D
pipeline above is working, extend `observation.state` from a single current-frame snapshot to a
short history window (~1-5 seconds) — analogous to how tactile F6 history already works (native
frame rate, per-episode-sliced, boundary-safe), but for the full joint-state vector rather than
just tactile. Not designed yet; flagged here so it isn't lost, and so §5's episode-boundary
patterns (already proven correct for tactile) are the template to reuse rather than re-deriving
boundary safety from scratch a second time.

---

## 1. Branch strategy — SETTLED

**Base: `dev/attempt2_server` at commit `641b189`** (not `main`, not `checks/attn_maps`) —
because this is the branch actually run on the real server. `checks/attn_maps` only adds this
session's diagnostic tooling (`attention_analysis/`) on top of the same tip; nothing
training-relevant is lost by not using it as the base. New branch name: **`dev/all_absolute`**
(deliberately not reusing "attempt3" — that name is taken by the hybrid-anchoring effort in
`REIMPLEMENTATION_PLAN.md`, and this is a different representation, not a continuation of it).

**Kept as-is from `dev/attempt2_server`** (Category A, general infra, dataset-agnostic):
1. DDP-unwrap fix
2. LR-scheduler-not-checkpointed fix
3. Cosine-schedule resume-clamp fix
4. DataLoader `RLIMIT_NOFILE` fix
5. Checkpoint disk-space preflight + atomic write
6. `output_attentions` plumbing (model-level, no training-behavior change)
7. Cluster launcher scripts (`run_trex_job.sh`, `setup_server_env.sh`, `cmd_trex.sh`)
8. `train_origami.sh`'s `NUM_GPUS`/prompt-default fixes

**Deleted outright, not edited** — all assumed anchoring/hybrid reconstruction, which no longer
exists under all-absolute:
- `T-Rex/trex_origami/anchoring.py`
- `origami_dataset.py`'s `reanchor_chunk`/`clamp_frozen_absolute`
- Anchoring-aware parts of `eval_origami.py`, `eval_diagnostics.py`, `eval_robustness.py`,
  `verify.py`, `preflight.py`, `smoke_test_origami.py`, `scripts/test.py` — rewritten against the
  all-absolute target, not surgically patched
- Zenoh deployment stack (`serve_origami_zenoh.py`, `policy.py`, `DEPLOY.md`,
  `bench_policy_latency.py`, `eval_deploy.py`, `replay_origami_real.py`) — coupled to anchoring
  reconstruction, rebuilt later once training actually works

**Reference for the rewrite — found in git history, not written from scratch**: `prepare.py`,
`prepare_fast.py`, and `accel.py` all exist in a clean, **pre-anchoring form at commit `85ab48e`**
(the direct parent of `98c029c`, the commit that introduced `anchoring.py`). Use `85ab48e`'s
versions as the structural base — they already have no anchoring coupling — then hand-port
forward the *non*-anchoring improvements made between `85ab48e` and `dev/attempt2_server`'s tip
(mainly `accel.py`'s NVDEC mixed-codec-probing hardening, ~170 lines; smaller non-anchoring
diffs in the other two files — check `git diff 85ab48e dev/attempt2_server -- <file>` line by
line when implementing, don't guess which parts are anchoring vs not). `prepare_fast.py` at
`dev/attempt2_server`'s tip is only 275 lines with just 5 lines touching anchoring — the
anchoring coupling here was always thin, not deeply structural.

**Logging — in scope for this pass.** Add clear console/log output during training that states
which flags actually got applied (freeze status, which layers are trainable, any config that
`training_args.json` doesn't fully capture) so it can be confirmed by eye during a run, not
reconstructed after the fact from an incomplete saved config — this is the direct fix for the
ambiguity that made confirming `checkpoint-2-7000`'s freeze status require a weight diff instead
of just reading a file.

---

## 2. What transfers, organized (numbers match the earlier keep/remove list)

**Keep regardless of branch choice (Category A — general infra, dataset-agnostic):**
1. DDP-unwrap fix (only matters if you keep a bare-DDP launcher rather than switching to the
   original's DeepSpeed configs — see §5)
2. LR-scheduler-not-checkpointed fix
3. Cosine-schedule resume-clamp fix
4. DataLoader `RLIMIT_NOFILE` fix
5. Checkpoint disk-space preflight + atomic write
6. NVDEC mixed-codec probing
7. `output_attentions` plumbing (model-level, no training-behavior change)
8. Cluster launcher scripts
9. Mid-epoch resume via `skip_first_batches`

**Drop (Category B — origami-fork-specific):**
11. Action anchoring, all of it — **confirmed you want this gone**
12. `trex_origami/` data-prep package as it stands (rebuilding regardless, see §4)
13. Training-prompt experiment
14. Zenoh deployment/serving stack (unless you want to keep the general flow-matching
    inference-loop *pattern* and rebuild just the origami-specific wiring — your call, separate
    from the branch decision)
15. `use_robot_state=1` default — revisit in §4

**Keep, orthogonal to the above (Category C):**
16. `attention_analysis/` toolkit — read-only, doesn't touch training behavior either way

---

## 3. Does `eef62_mini` help? Checked directly, and the answer is nuanced

Fetched `meta/info.json` from `huggingface.co/datasets/kshitij-hf/eef62_mini`:

| | `eef62_mini` | origami's real robot data |
|---|---|---|
| robot_type | `north_poc2_2` | (same platform family — `north_poc2_2_urdf_usd/` already sits in this repo) |
| state/action dims | **62** | **65** |
| arm representation | 9D EEF/task-space pose per arm (3D trans + 6D rot) | **7 raw joint angles per arm** |
| motor/torso-neck group | none | **7D**, present |
| action chunk length | 16 | 25 |
| cameras | head, wrist_left, wrist_right | same names, same convention |
| tactile_f6 | 10×6 | same shape |
| tactile_deform | 10 feeds, 240×240 | same |
| fps | 30 | 30 (confirmed native rate) |
| tasks | 1 (not origami) | origami paper-folding |
| scale | 180 episodes, 1.39M frames | (your season, smaller) |

**Verdict: not a drop-in for origami fine-tuning.** Different action-space dimensionality *and*
representation (task-space EEF pose vs. raw joints — origami's data has no forward-kinematics
step to derive the EEF pose this dataset uses), different chunk length, and it's a different task
entirely (single-task dataset, not paper-folding).

**What it's actually good for**: it's real data on the same robot hardware family, with a
matching camera/tactile setup, in the *original* 62D EEF convention. Two legitimate uses:
- A ready-made way to stand up and sanity-check an original-format (62D EEF) data loader, if you
  want that code path to exist and be tested even if you don't train on it for origami.
- Possible extra pretraining/midtrain-style data before origami-specific fine-tuning, since the
  hardware and sensing setup genuinely match — worth a separate cost/benefit look, not assumed
  here.

Not something I'd treat as origami training data itself.

---

## 4. Action representation — SETTLED: all-absolute, action_chunk=16

**Decision: option (c), all-absolute.** No delta anywhere, for any of the 65 dims — arms, hands,
and motor are all predicted as plain absolute joint values. This matches the raw LeRobot data's
own native representation exactly (`modality.json` already marks the whole 65-D block absolute),
sidesteps both documented delta failure modes in one move (original's own delta-from-state
tracking-offset problem, and the near-static-dim normalization-amplifies-noise problem), and needs
no anchoring module, no per-dim `meta/dataset.json` bookkeeping, no `build_anchor`/reconstruction
step at all — the model's raw output *is* the target, full stop. This is also the simplest
possible scheme to implement and to reason about, which matters given the explicit goal of
avoiding origami-fork abstraction machinery.

Trade-off accepted knowingly: this gives up whatever motion-prior benefit a
delta-from-recent-command target would have provided (the model has to learn absolute joint
targets from scratch rather than a small correction on top of its own last command). Not
revisiting unless real training data says otherwise.

**`action_chunk = 16`**, matching original T-Rex's own value (not origami's current checkpoint's
25). `chunk_stride` stays independent of `sample_stride` regardless (§5) — this decision only
changes how many steps the chunk covers, not how it's indexed.

For context, this was weighed against two other options that are no longer live: (a)
delta-from-current-state, literally original's own code, but reintroduces the exact failure origami's
own `checkpoint-2-8000` experiments found (arms can't beat hold-state; the tracking offset isn't
observable from any model input) — the parallel `dev/redesign` plan is taking this option for its
EEF-62 representation and has flagged it as their #1 ranked risk, worth watching how that plays
out; (b) delta-from-previous-command (origami's own validated fix, implemented without the
`anchoring.py` abstraction) — this was my earlier lean, superseded by (c).

---

## 5. Dataset feature set and preprocessing (settled — from evidence gathered this session)

**Per-episode data needed:**
- 3 RGB streams: head (slow), wrist_left + wrist_right (fast) — same convention both repos use
- Tactile F6 (10 fingers × 6D force/torque)
- Tactile deform images (10 fingers) — both repos default this on
- 65D state + action target (7D/arm + 22D/hand ×2 + 7D motor/torso-neck)
- Language instruction
- Explicit episode row boundaries (critical for the boundary-safety pattern below)

**Preprocessing, in dependency order:**
1. Pick `sample_stride` based on your task's actual motion speed, not a default — see the
   stride discussion: it controls dataset density and how well short/fast transitions get
   sampled, but does *not* change the action chunk's real-time span (see next point)
2. Keep `chunk_stride=1` (or whatever native-frame value), independent of `sample_stride` —
   confirmed this is how original avoids coupling chunk horizon to sampling density; worth
   keeping exactly as-is
3. Slice every per-signal array to `[episode_row_from:episode_row_to]` *before* any windowing
   logic, so boundary-clipping can only repeat-pad within the same episode
4. Tactile F6 history at native frame rate, independent of `sample_stride` (matches what the
   embedded VQ-VAE was trained on)
5. q01/q99 (not raw min/max) normalization stats for action/state/tactile_f6
6. Verify camera synchronization against actual capture hardware guarantees — don't assume
   naive same-index frame pairing is safe without checking
7. Decide the instruction/prompt convention early and test it empirically — the paper-airplane
   rewrite underperformed the plain task label in origami's own benchmark

**Still open, smaller decisions:**
- `use_robot_state`: origami's fork enables it (`1`), original's own `train.sh` default is `0` —
  worth an explicit choice, not inherited silently either way
- How many decoder layers to leave trainable — separate discussion already had; original's own
  reference recipe is a full 28-layer fine-tune, origami's `train_latent_last_n=4` was a
  single-GPU compute compromise, not a demonstrated-better architecture

---

## 6. `final_layer` (65D action head) — per-segment plan, evidenced per-dim

Not a single freeze/unfreeze decision — the 65 output dims split into three groups that each need
different treatment, based on what actually transfers and what the real data shows.

**Hands (44 dims: `left_hand[7:29]`, `right_hand[36:58]`) — reuse, stay fully trainable.**
Same Sharpa Wave hardware, same joint order/count confirmed against original T-Rex's own
`joint_names_Left_Sharpa_HA4.yaml` (thumb/index/middle/ring/pinky, 5/4/4/4/5). Copy the pretrained
`final_layer`'s hand-output weight rows into the new model at init — no gradient hook, no
freezing, ordinary backprop from there. Origami's task genuinely needs different finger behavior
than whatever the pretrained checkpoint learned, so this is a warm start, not a lock.

**Arms (14 dims: `left_arm[0:7]`, `right_arm[29:36]`) — no transplant possible, fresh init.**
Confirmed dimension *and* representation mismatch, not just a count difference: original T-Rex's
arm output is 9D task-space pose (3D translation + 6D rotation), origami's is 7 raw joint angles.
A pretrained row that outputs "end-effector Z translation" has no corresponding row to copy into
one that outputs "joint 4's angle" — there's nothing to transplant. Random init, train from
scratch.

**Motor (7 dims: `motor[58:65]`, torso+neck) — checked against real data, not uniform.**
No original T-Rex equivalent exists at all (bimanual-arm+hand only, no torso/neck), so this group
is fresh-init regardless — the question is which of the 7 dims are worth learning at all:

| dim | index | variance (91,338 frames, full season) | cross-episode shape | verdict |
|---|---|---|---|---|
| `lower_body_j1` | 58 | range 0.0019 rad — static | — | **exclude**, hold at measured state (matches existing deployment: `"frozen action dims [58, 59]"`) |
| `lower_body_j2` | 59 | range 0.0010 rad — static | — | **exclude**, same as above |
| `lower_body_j3` | 60 | range 0.226 rad, std 0.038 | corr 0.74 (0.32–0.94) | learn — real movement, some shared shape but not scripted-clean |
| `lower_body_j4` | 61 | range 0.088 rad, std 0.017 | corr 0.51 (-0.24–0.99) | learn — inconsistent shape, looks task-coupled |
| `lower_body_j5` | 62 | range 0.122 rad, std 0.020 | corr 0.005 (-0.86–0.89) | **learn — clearest genuinely task-reactive dim of the 7** |
| `neck_j1` | 63 | range 0.274 rad, std 0.038 | corr 0.59 | learn, moderate confidence |
| `neck_j2` | 64 | range 0.347 rad, std 0.041 | corr 0.86 (up to **0.998**), fwd/rev symmetry 0.68 | **suspect scripted/preprogrammed** — near-identical trajectory shape across all 14 episodes regardless of task content; treat like 58/59 (hold at a fixed/mean trajectory) rather than learn contextually, unless further evidence says otherwise |

So of the 7 motor dims: 2 confirmed static (exclude), 1 confirmed likely-scripted (`neck_j2`,
exclude pending more evidence), 4 genuinely worth learning (`lower_body_j3/j4/j5`, `neck_j1`).
Net: model learns 4 motor dims + 14 arm dims (fresh) + 44 hand dims (warm-started) = 62 of 65
active outputs, with 3 (58, 59, 64) held at fixed/measured values instead of predicted.

**Does motor being absolute affect the reused hand weights? No, for two independent reasons.**
First, this isn't actually new — motor was already `absolute` under the *previous* hybrid scheme
too (confirmed from the real submitted checkpoint's own `action_anchor`, §4 note); only the arms
changed representation. Second, even if it were new: `final_layer` is a plain
`nn.Linear(hidden_size, 65)` — each output dimension is an independent row of the weight matrix,
with zero mathematical coupling between rows at initialization. Copying pretrained values into
the hand rows and randomly initializing the motor rows are separate slices of the same matrix;
the only interaction is through shared upstream gradients during *training*, which is a property
of any multi-output linear head and isn't introduced or worsened by this choice.

---

## 7. Tactile VQ-VAE calibration — check this on the *existing* checkpoint too, not just the new plan

Surfaced by reviewing the parallel `dev/redesign` plan (§11.8 there), but it applies to
`checkpoint-2-7000` right now, independent of which dataset-format plan gets built.

**The bug shape**: there are two separate F6 normalizations. The per-frame tactile vector into
`tacf6_embedder` is normalized by the loader using origami's own q01/q99 — correct by
construction. The raw F6 *history window* feeding the frozen embedded VQ-VAE
(`modeling_vla.py::encode_tactile_f6_history`) is normalized *inside the model* by
`tacf6_vqvae_min/max/mask` buffers baked into the checkpoint from **T-Rex's own pretraining
calibration** — nothing origami-aware touches those. If they don't match origami's actual force
range, the VQ-VAE either saturates at ±1 or origami's whole range sits in a sliver near one end —
either way the tactile expert gets a near-degenerate discrete code regardless of real contact,
and nothing errors.

**Checked directly against real data from both sides** (not assumed):

```
                    T-Rex (zekaiwang/trex_dataset, 122,652 frames)   origami (this season)
L/R thumb,index,mid p99 ≈ 13-25N, all 6 clearly active              p99 up to ~42N, active — comparable order of magnitude
L/R ring, pinky     p99 ≈ 13-34N — fully active, L_pinky is         mean 0.002-0.065N — dead
                    T-Rex's single highest-signal finger
```

**Corrected on review — the dead-finger half is actually not a real risk, the saturation half
is the only one that can genuinely destroy information:**
- **Dead channels (both ring, both pinky) are fine regardless of calibration.** A constant input
  always finds the same nearest codebook entry under a fixed linear map, so a genuinely-dead
  sensor produces a *constant code* whether or not T-Rex's calibration matches origami's — that's
  the correct representation of "no information here," not a failure mode. Even if the specific
  code index chosen isn't the one T-Rex's own data would have picked for "zero," the downstream
  `tactile_code_embedder` is trainable (not frozen), so origami fine-tuning can still learn what
  that index means. Nothing to fix here.
- **Saturation/clamping on the 6 active channels is the one real risk**, and specifically only
  where it matters: if origami's genuine, time-*varying* force signal exceeds T-Rex's calibrated
  max, distinct real values (e.g. 30N vs 42N) both clamp to +1 and become the *same* code — a
  real collapse of information that no amount of downstream fine-tuning recovers, unlike the
  dead-channel case. This is the only scenario worth calling a "bug."
- The magnitude comparison above argues this specific risk is currently **low but unconfirmed** —
  ranges are roughly comparable between origami and T-Rex's public release, not an
  order-of-magnitude mismatch that would obviously saturate. Still worth checking against the
  checkpoint's actual baked-in buffer values directly (I've only compared public dataset samples
  from both sides, not the exact numbers `checkpoint-2-7000` carries) before concluding it's a
  non-issue.

**Action item, cheap and checkpoint-only**: read `checkpoint-2-7000`'s own
`tacf6_vqvae_min/max/mask` buffer values directly, compare against origami's real per-finger q01/
q99, and check codebook usage/entropy on a real batch of origami F6 windows through
`encode_tactile_f6_history` — before assuming the tactile expert is contributing anything useful
in the current checkpoint. If buffers are mismatched, the fix is a checkpoint edit (overwrite the
three buffers with origami-fit values, encoder/codebook weights untouched), not a data-pipeline
change — passing pre-normalized values through the existing code path doesn't work, since
`encode_tactile_f6_history` unconditionally reapplies its own fixed transform to whatever it's
given.
