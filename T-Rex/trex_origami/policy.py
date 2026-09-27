"""The T-Rex origami policy behind the competition's `TeamPolicy` boundary.

One class, `TRexOrigamiPolicy`, with the two methods the kit's
`policy_server_template.py` calls -- `reset()` and `infer(observation)` -- and
nothing else that the robot side needs to know about.  It takes the *full
public observation* of `docs/robot_io_spec.md` (four RGB cameras, state, torque,
tactile wrench, tactile deform grid, prompt) and returns `float32[25, 65]`
absolute joint targets in radians.  The offline replay
(`scripts/replay_origami_real.py`) and the Zenoh server
(`scripts/serve_origami_zenoh.py`) both call exactly this method, which is what
the participant guide asks for: one preprocessing / normalisation / joint
mapping implementation, not two.

What happens inside `infer`, and why
------------------------------------
* **Images.** `head_left` is the slow expert's image; `[wrist_right, wrist_left]`
  are the fast images, in that order, because that is the order
  `OrigamiDataset.collate_fn` fed the model.  `head_right` is accepted and
  ignored (the checkpoint never saw it).  The wire already delivers 224x224
  squashed frames, matching the training prep.
* **Prompt.** The client's `prompt` is ignored.  The policy trained on one
  constant string (the dataset's own task label, or `--instruction` if the
  checkpoint recorded one) and serving an unseen prefix measurably hurts.
* **Tactile.**  The 60-D wrench is normalised into the `tacf6` token; the
  embedded VQ-VAE wants a 16-frame *30 Hz* history, which no `infer` call
  carries.  The policy keeps a time-stamped buffer of the wrenches it has seen
  and resamples it onto a 30 Hz grid ending at the current observation
  (hold-previous).  With one observation this degenerates to tiling the
  current frame -- which `eval_robustness` showed costs the checkpoint nothing
  (a frozen or strided history moved MAE by 0.000 deg).  The deform grid is
  converted to grayscale exactly as the prep's JPEG decode did (`PIL` "L").
* **Mean of K draws.** K flow integrations from independent noise share one
  vision/language prefix, and their mean is returned.  Sampling variance was
  ~40% of the checkpoint's MSE; K=8 removes most of it at almost no latency.
* **All 65 dims are absolute joint radians** -- no delta, no anchor
  reconstruction, straight off `denormalize()`.  Frozen/near-static dims
  (torso 58/59 on the pilot split) are held at the measured state instead of
  the flow head's output, per `frozen_action_dims`/`clamp_frozen_absolute`
  (`qwen_vla.origami_dataset`), driven by the norm-stats mask, not a spec.
* **Safety projection.** A sequential clamp into the Shadow evaluator's
  feasible set (URDF position limits within tolerance, per-group step-jump and
  velocity budgets at 30 Hz).  The limits come from `joint_limits.json`, a
  65-row table exported from the URDF, because the organizer forbids shipping
  the URDF asset itself in the submission image.

Episode-scoped state (tactile buffer, last chunk) is cleared by `reset()`.
"""
from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
from dataclasses import dataclass, asdict
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import PIL.Image
import torch

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_PKG_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from qwen_vla import extend_position_ids_for_flare, split_slow_fast_embeds  # noqa: E402
from qwen_vla.origami_dataset import (N_FINGERS, F6_PER_FINGER,  # noqa: E402
                                      clamp_frozen_absolute, denormalize,
                                      frozen_action_dims, split_deform_strip)
from .loading import load_args_from_checkpoint, model_load  # noqa: E402
from .seasons import ACTION_DIM, DATASET_TASK_STRING, JOINT_NAMES  # noqa: E402

CONTROL_HZ = 30.0
OBS_HEAD_LEFT = "observation/image/head_left"
OBS_HEAD_RIGHT = "observation/image/head_right"
OBS_WRIST_LEFT = "observation/image/wrist_left"
OBS_WRIST_RIGHT = "observation/image/wrist_right"
OBS_STATE = "observation/state"
OBS_TORQUE = "observation/state/joint_torque"
OBS_TACTILE = "observation/tactile"
OBS_DEFORM = "observation/image/tactile_deform"
OBS_TACTILE_RAW = "observation/image/tactile_raw"
DEFAULT_LIMITS = os.path.join(_PKG_DIR, "joint_limits.json")


def _normalize(values, mask, vmin, vmax):
    return np.where(mask, np.clip(2 * (values - vmin) / (vmax - vmin + 1e-8) - 1, -1, 1),
                    values)


# ── configuration ─────────────────────────────────────────────────────────────
@dataclass
class PolicyConfig:
    checkpoint_path: str
    mode: str = "cascaded"            # "cascaded" (tactile expert) | "blind"
    total_steps: int = 10             # Euler steps over tau in [0, 1]
    split_step: int = 6               # action-expert steps before the tactile expert
    n_draws: int = 8                  # K flow draws averaged
    max_flow_batch: int = 64
    safety: str = "tol"               # "tol" | "urdf" | "off" | "none" (disable)
    rate_margin: float = 0.999
    limits_path: str = DEFAULT_LIMITS
    instruction: str = ""             # "" = what the checkpoint / dataset declares
    tactile_history: str = "resample" # "resample" | "tile"
    device: str = "cuda"
    seed: Optional[int] = None        # fixed noise seed per request (None = fresh)
    warmup: bool = True
    compile: bool = False             # torch.compile the flow functions (see _compile)
    compile_mode: str = "default"     # "default" only; CUDA graphs deadlock off-thread
    tactile_refine_every: int = 1     # 1 = always full slow+fast (current/validated behavior);
                                       # N>1 = reuse the last slow-tick's KV for (N-1) of every N
                                       # ticks, running only the tactile expert on those -- blind
                                       # to any new camera images in the skipped ticks' observations
                                       # (tactile_flow_continue takes no vision input at all).
    tactile_refine_max_stale_s: float = 0.5  # force a fresh slow tick if the cache is older than
                                              # this, regardless of the counter -- bounds how stale
                                              # the reused vision context can get under irregular
                                              # call timing.
    # cross-timestep memory (see qwen_vla/MEMORY_DESIGN.md). "" / 0 / None
    # auto-detects from training_args.json, matching every other
    # architecture flag above -- a memory-trained checkpoint must be served
    # with the exact tiers/strides it trained with, these are not free
    # serving knobs.
    memory_slow_seconds: str = ""             # comma-separated seconds-back targets, e.g. "0.25,0.5,1,5"
    memory_fast: int = 0                      # linear fast-memory window, in ticks
    memory_rope_stride_slow: Optional[float] = None
    memory_rope_stride_fast: Optional[float] = None
    memory_buffer_margin_sec: float = 2.0     # extra slow-buffer retention past the oldest target
    disable_memory: bool = False              # force memory off, overriding training_args.json
                                               # auto-detect -- for A/B diagnosis only, not a
                                               # normal serving knob


# ── safety projection (self-contained copy of eval_smoothed.SafetyProjector) ──
class SafetyProjector:
    """Sequential clamp of an absolute chunk into the evaluator's feasible set."""

    def __init__(self, limits_path: str, position_mode: str = "tol",
                 rate_margin: float = 0.999):
        with open(limits_path) as handle:
            table = json.load(handle)
        joints = table["joints"]
        if [j["name"] for j in joints] != list(JOINT_NAMES):
            raise ValueError(f"{limits_path} joint order does not match the contract")
        lower = np.array([j["lower"] for j in joints], dtype=np.float64)
        upper = np.array([j["upper"] for j in joints], dtype=np.float64)
        vel = np.array([j["velocity"] for j in joints], dtype=np.float64)
        jump = np.array([j["jump"] for j in joints], dtype=np.float64)
        tol = float(table.get("position_tolerance_rad", math.radians(2)))
        hz = float(table.get("control_hz", CONTROL_HZ))
        if position_mode == "tol":
            self.lo, self.hi = lower - rate_margin * tol, upper + rate_margin * tol
        elif position_mode == "urdf":
            self.lo, self.hi = lower, upper
        elif position_mode == "off":
            self.lo, self.hi = np.full_like(lower, -np.inf), np.full_like(upper, np.inf)
        else:
            raise ValueError(position_mode)
        self.max_step = rate_margin * np.minimum(jump, vel / hz)

    def project(self, seed_abs: np.ndarray, traj_abs: np.ndarray) -> np.ndarray:
        prev = np.array(seed_abs, dtype=np.float64, copy=True)
        out = np.empty_like(traj_abs, dtype=np.float64)
        for k in range(traj_abs.shape[-2]):
            target = np.clip(traj_abs[..., k, :], self.lo, self.hi)
            prev = prev + np.clip(target - prev, -self.max_step, self.max_step)
            out[..., k, :] = prev
        return out


# ── the policy ────────────────────────────────────────────────────────────────
class TRexOrigamiPolicy:
    def __init__(self, config: PolicyConfig):
        self.cfg = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self._lock = threading.Lock()
        self.model, self.processor, self.statistic, self.train_args = self._load_model()
        self.action_chunk = int(self.model.action_chunk)
        self.action_dim = int(self.model.action_dim)
        if self.action_dim != ACTION_DIM:
            raise ValueError(f"checkpoint action_dim {self.action_dim} != {ACTION_DIM}")
        if not (0 < config.split_step < config.total_steps) and config.mode == "cascaded":
            raise ValueError("split_step must be in (0, total_steps)")

        st = self.statistic
        self.action_mask = np.asarray(st["action_mask"], dtype=bool)
        self.action_min, self.action_max = st["action_min"], st["action_max"]
        self.tacf6_mask = np.asarray(st["tacf6_mask"], dtype=bool)
        self.tacf6_min, self.tacf6_max = st["tacf6_min"], st["tacf6_max"]
        self.frozen_dims = frozen_action_dims(self.action_mask)

        self.instruction = (config.instruction or self.train_args.get("instruction")
                            or DATASET_TASK_STRING)
        size = self.train_args.get("image_size") or [224, 224]
        self.image_size = (int(size[0]), int(size[1]))
        self.vqvae_window = int(getattr(getattr(self.model, "tactile_vqvae", None), "cfg",
                                        SimpleNamespace(window=16)).window)
        self.use_state = bool(self.train_args.get("use_robot_state", 1))
        self.use_f6 = bool(self.train_args.get("use_tactile_vec", 1))
        self.use_deform = bool(self.train_args.get("use_tactile_deform", 1))

        # cross-timestep memory: two raw-content buffers (not pre-computed
        # KV -- RoPE bakes each row's position into its cached K before
        # storage, so a snapshot can't be re-shifted for a later, differently
        # -timed tick without recompute; build_memory_kv_slow/_fast recompute
        # fresh from these every slow tick). See qwen_vla/MEMORY_DESIGN.md.
        if config.disable_memory:
            raw_slow = ""
            memory_fast_raw = 0
        else:
            raw_slow = (config.memory_slow_seconds
                        or self.train_args.get("memory_slow_seconds", "") or "")
            memory_fast_raw = (config.memory_fast
                               or self.train_args.get("memory_fast", 0) or 0)
        self.memory_slow_seconds = sorted(
            (float(s) for s in str(raw_slow).split(",") if s.strip()), reverse=True)
        self.memory_fast = int(memory_fast_raw)
        self.memory_rope_stride_slow = float(
            config.memory_rope_stride_slow if config.memory_rope_stride_slow is not None
            else self.train_args.get("memory_rope_stride_slow", 32.0))
        self.memory_rope_stride_fast = float(
            config.memory_rope_stride_fast if config.memory_rope_stride_fast is not None
            else self.train_args.get("memory_rope_stride_fast", 8.0))
        self.memory_buffer_margin_sec = float(config.memory_buffer_margin_sec)
        # memory_buf_slow: List[(timestamp, PIL.Image head, str task_text)]
        # memory_buf_fast: List[(timestamp, [wrist_right, wrist_left], action_raw_np)]
        self.memory_buf_slow: List[Tuple[float, PIL.Image.Image, str]] = []
        self.memory_buf_fast: List[Tuple[float, List[PIL.Image.Image], np.ndarray]] = []
        # Wrist images from the most recent slow tick -- the wire protocol
        # never sends fresh wrist images on every tick under
        # tactile_refine_every>1 (fast ticks only carry tactile), so a fast
        # tick has no new image to remember; reusing the last-seen images
        # paired with THAT tick's own freshly-refined action still captures
        # real fast-tick-rate temporal density for the part that actually
        # changes that fast -- the action.
        self._last_fast_images: Optional[List[PIL.Image.Image]] = None
        # Best-known executed action, for the fast-memory buffer's
        # action_abs proxy (_prev_command) -- not read anywhere else.
        self.last_chunk: Optional[np.ndarray] = None
        self.last_chunk_time: float = 0.0
        if self.memory_slow_seconds:
            print(f"[policy] slow memory enabled: targets {self.memory_slow_seconds}s back, "
                  f"rope_stride={self.memory_rope_stride_slow}")
        if self.memory_fast > 0:
            print(f"[policy] fast memory enabled: window={self.memory_fast} ticks, "
                  f"rope_stride={self.memory_rope_stride_fast}")

        # loading.model_load only writes state_mask/min/max when the checkpoint
        # was trained with use_robot_state=1 -- this checkpoint has it at 0, so
        # these keys are genuinely absent, not just unused. Reading them
        # unconditionally (as this did before) is a KeyError waiting to happen
        # on any use_robot_state=0 checkpoint; every checkpoint used so far had
        # use_robot_state=1, so it never fired.
        if self.use_state and "state_mask" in st:
            self.state_mask = np.asarray(st["state_mask"], dtype=bool)
            self.state_min, self.state_max = st["state_min"], st["state_max"]
        else:
            self.use_state = False
            self.state_mask = self.state_min = self.state_max = None

        self.projector = None
        if config.safety not in ("none", None):
            self.projector = SafetyProjector(config.limits_path, config.safety,
                                             config.rate_margin)
        self.generator = None
        if config.seed is not None:
            self.generator = torch.Generator(device=self.device).manual_seed(config.seed)

        # episode-scoped state
        self._f6_buffer: List[Tuple[float, np.ndarray]] = []
        self.last_timing: Dict[str, float] = {}
        self.n_infer = 0

        # cross-tick slow-pass cache (tactile_refine_every > 1 only)
        if config.tactile_refine_every > 1 and config.n_draws > config.max_flow_batch:
            raise ValueError(
                "tactile_refine_every > 1 requires n_draws <= max_flow_batch "
                f"(got n_draws={config.n_draws}, max_flow_batch={config.max_flow_batch}); "
                "a K-batch split across sub-batches would only cache the last one")
        self._tick = 0
        self._cache_kv = None
        self._cache_kv_exclude_mask = None
        self._cache_pos = None
        self._cache_mask = None
        self._cache_x_split = None
        self._cache_n_action = None
        self._cache_tau_split = None
        self._cache_built_at = None

        print(f"[policy] frozen dims {self.frozen_dims.tolist()}; "
              f"prompt {self.instruction!r}; "
              f"mode={config.mode} steps={config.total_steps}/{config.split_step} "
              f"K={config.n_draws} safety={config.safety} compile={config.compile} "
              f"tactile_refine_every={config.tactile_refine_every}")
        if config.compile:
            self._compile()
        if config.warmup:
            self._warmup()

    def _compile(self) -> None:
        """torch.compile the three flow entry points (the only GPU-heavy calls
        after the vision prefix).

        Every shape here is fixed for the life of the process -- fixed prompt,
        fixed image sizes, fixed K, fixed horizon -- so Dynamo's automatic
        dynamic-shape promotion is switched off: otherwise the first *live*
        call after warm-up is taken as evidence that a dimension varies and
        triggers a second, dynamic graph compile, which cannot happen inside a
        read-only container.  `mode="default"` on purpose: CUDA-graph modes
        capture on the warm-up thread and deadlock when Zenoh's callback
        thread replays them (observed on the previous submission).
        `tau_split = float(time.item())` inside forward_flow_action_partial
        is a known, benign graph break.
        """
        import torch._dynamo
        torch._dynamo.config.automatic_dynamic_shapes = False
        torch._dynamo.config.cache_size_limit = max(
            torch._dynamo.config.cache_size_limit, 64)
        m = self.model
        m.forward_flow_action_partial = torch.compile(
            m.forward_flow_action_partial, mode=self.cfg.compile_mode)
        m.tactile_flow_continue = torch.compile(
            m.tactile_flow_continue, mode=self.cfg.compile_mode)
        m.forward_flow_action_full = torch.compile(
            m.forward_flow_action_full, mode=self.cfg.compile_mode)
        print(f"[policy] torch.compile({self.cfg.compile_mode!r}) armed for "
              f"forward_flow_action_partial / tactile_flow_continue / "
              f"forward_flow_action_full; inductor cache "
              f"{os.environ.get('TORCHINDUCTOR_CACHE_DIR', '<default>')}")

    # ── loading ───────────────────────────────────────────────────────────────
    def _load_model(self):
        """`trex_origami.loading.model_load` -- the one checkpoint loader
        (lifted from scripts/test.py so serving needs no scripts/ or pyzmq)."""
        ta_path = os.path.join(self.cfg.checkpoint_path, "training_args.json")
        train_args = json.load(open(ta_path)) if os.path.exists(ta_path) else {}
        load_args = load_args_from_checkpoint(self.cfg.checkpoint_path)
        model, processor, statistic = model_load(load_args)
        model = model.to(self.device).eval()
        return model, processor, statistic, train_args

    def _warmup(self) -> None:
        """Run every observation *pattern* the process will meet before READY.

        Without compile one call suffices (CUDA context, allocator, processor
        caches).  With compile each pattern below has produced its own graph
        on the previous submission (random content, the validator's all-zero
        tactile + gradient images, and differently-valued real content), so
        all of them are exercised here -- at build time this is what fills
        the inductor cache, at start-up it is what makes the first live query
        a cache hit.  A repeat of the first pattern is timed so the log shows
        the steady-state cost.
        """
        patterns = [("random", synthetic_observation())]
        if self.cfg.compile:
            patterns += [("validator_zero_tactile", validator_observation()),
                         ("varied_1", varied_observation(1)),
                         ("varied_2", varied_observation(2))]
        t_all = time.time()
        for name, obs in patterns:
            t0 = time.time()
            self.reset()
            self.infer(obs)
            self.infer(obs)      # second call: with compile this must be a cache hit
            print(f"[policy] warm-up pattern {name}: first+second call "
                  f"{1000 * (time.time() - t0):.0f} ms, second call timing "
                  f"{self._fmt_timing()}")
            if self.cfg.tactile_refine_every > 1:
                # The two calls above can both land as "slow" if compiling took
                # long enough to blow the staleness budget before the second
                # call -- that would mean _flow_cached's call to
                # tactile_flow_continue (a *reused*, prior-call DynamicCache,
                # not a same-call fresh one) never gets exercised/compiled
                # here, and the first real "fast" tick in the container would
                # hit it live. Force it directly, bypassing the tick/staleness
                # gate, so it is always compiled before READY.
                t1 = time.time()
                with self._lock, torch.inference_mode():
                    f6 = np.asarray(obs[OBS_TACTILE], dtype=np.float32).reshape(
                        N_FINGERS, F6_PER_FINGER)
                    tactile = self._tactile_tensors(f6, obs.get(OBS_DEFORM), time.time())
                    self._flow_cached(tactile, {})
                print(f"[policy] warm-up pattern {name}: forced cached-fast call "
                      f"{1000 * (time.time() - t1):.0f} ms")
        self.reset()
        print(f"[policy] warm-up complete in {time.time() - t_all:.1f} s")

    def _fmt_timing(self) -> str:
        return " ".join(f"{k}={v:.0f}" for k, v in self.last_timing.items())

    # ── protocol ──────────────────────────────────────────────────────────────
    def metadata(self) -> Dict[str, Any]:
        return {"action_dim": self.action_dim, "action_horizon": self.action_chunk,
                "joint_names": list(JOINT_NAMES), "policy": asdict(self.cfg)}

    def reset(self) -> None:
        with self._lock:
            self._f6_buffer.clear()
            self._tick = 0
            self._cache_kv = None
            self._cache_kv_exclude_mask = None
            self._cache_pos = None
            self._cache_mask = None
            self._cache_x_split = None
            self._cache_n_action = None
            self._cache_tau_split = None
            self._cache_built_at = None
            # Memory must not leak across episodes, exactly like the
            # training-side index never lets a lookback window cross an
            # episode boundary.
            self.memory_buf_slow = []
            self.memory_buf_fast = []
            self._last_fast_images = None
            self.last_chunk = None
            self.last_chunk_time = 0.0

    def infer(self, observation: Dict[str, Any], now: Optional[float] = None) -> np.ndarray:
        """Full public observation -> float32[T, 65] absolute radians.

        `now` is the observation time in seconds; it defaults to wall-clock.
        An offline replay passes the frame timestamp so the tactile-history
        resampling sees replay time, not CPU time.
        """
        with self._lock, torch.inference_mode():
            return self._infer(observation, time.time() if now is None else float(now))

    # ── pieces ────────────────────────────────────────────────────────────────
    def _infer(self, obs: Dict[str, Any], now: float) -> np.ndarray:
        timing: Dict[str, float] = {}
        t0 = time.time()
        state = np.asarray(obs[OBS_STATE], dtype=np.float64).reshape(self.action_dim)
        if not np.isfinite(state).all():
            raise ValueError("observation/state is not finite")
        f6 = np.asarray(obs[OBS_TACTILE], dtype=np.float32).reshape(N_FINGERS, F6_PER_FINGER)
        self._push_f6(now, f6)
        tactile = self._tactile_tensors(f6, obs.get(OBS_DEFORM), now)
        timing["prep_ms"] = 1000 * (time.time() - t0)

        every = max(1, int(self.cfg.tactile_refine_every))
        stale = (self._cache_kv is None or self._cache_built_at is None
                 or now - self._cache_built_at > self.cfg.tactile_refine_max_stale_s)
        do_slow = every == 1 or stale or (self._tick % every == 0)
        self._tick += 1
        timing["slow_tick"] = 1.0 if do_slow else 0.0

        if do_slow:
            # This tick's camera images are used and the slow-pass KV is
            # cached (if tactile_refine_every > 1) for later "fast" ticks to
            # reuse -- see _flow_cached, which never looks at new images at
            # all (tactile_flow_continue takes no vision input).
            t1 = time.time()
            # Built from buffered PRIOR ticks only -- this tick's own content
            # is recorded into the buffers further down, after this read, so
            # a tick never becomes its own memory.
            memory_kv, kv_exclude_mask = self._build_memory_kv(now)
            slow_imgs = [self._pil(obs[OBS_HEAD_LEFT])]
            fast_imgs = [self._pil(obs[OBS_WRIST_RIGHT]), self._pil(obs[OBS_WRIST_LEFT])]
            slow, pos, fast, mask, state_emb = self._embed(slow_imgs, fast_imgs, state)
            self._sync()
            timing["embed_ms"] = 1000 * (time.time() - t1)
            self._remember_slow(now, slow_imgs[0], self.instruction)
            self._remember_fast(now, fast_imgs, self._prev_command(state))
            self._last_fast_images = fast_imgs
            t2 = time.time()
            norm = self._flow_slow(slow, pos, fast, mask, state_emb, tactile, timing,
                                   cache=(every > 1), now=now, memory_kv=memory_kv,
                                   kv_exclude_mask=kv_exclude_mask)
            timing["flow_ms"] = 1000 * (time.time() - t2)
        else:
            timing["embed_ms"] = 0.0
            timing["cache_age_ms"] = 1000 * (now - self._cache_built_at)
            t2 = time.time()
            norm = self._flow_cached(tactile, timing)
            timing["flow_ms"] = 1000 * (time.time() - t2)

        t3 = time.time()
        actions = self._reconstruct(norm.mean(axis=0), state)
        timing["post_ms"] = 1000 * (time.time() - t3)
        if not do_slow and self._last_fast_images is not None:
            # Fast-tick-rate fast-memory capture: wrist images are reused
            # from the last slow tick (nothing fresher over the wire), but
            # the action genuinely is fresh -- _reconstruct just wrote this
            # tick's own output into self.last_chunk, so _prev_command here
            # (elapsed~=0) returns THIS tick's command, matching training's
            # fast-tick-rate memory density instead of only updating at
            # slow-tick rate.
            self._remember_fast(now, self._last_fast_images, self._prev_command(state))
        timing["total_ms"] = 1000 * (time.time() - t0)
        self.last_timing = timing
        self.n_infer += 1
        return actions

    def _pil(self, image: np.ndarray) -> PIL.Image.Image:
        arr = np.asarray(image)
        if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"camera image must be uint8 HWC RGB, got {arr.dtype}{arr.shape}")
        img = PIL.Image.fromarray(np.ascontiguousarray(arr), mode="RGB")
        if img.size != self.image_size:
            img = img.resize(self.image_size, PIL.Image.LANCZOS)
        return img

    def _push_f6(self, now: float, f6: np.ndarray) -> None:
        self._f6_buffer.append((now, f6.copy()))
        horizon_s = (self.vqvae_window + 2) / CONTROL_HZ
        while len(self._f6_buffer) > 1 and now - self._f6_buffer[0][0] > horizon_s:
            self._f6_buffer.pop(0)

    def _f6_history(self, now: float) -> np.ndarray:
        """[W, 10, 6] raw wrench window on a 30 Hz grid ending at `now`."""
        W = self.vqvae_window
        if self.cfg.tactile_history == "tile" or len(self._f6_buffer) == 1:
            return np.repeat(self._f6_buffer[-1][1][None], W, axis=0)
        times = np.array([t for t, _ in self._f6_buffer])
        grid = now - (W - 1 - np.arange(W)) / CONTROL_HZ
        idx = np.searchsorted(times, grid, side="right") - 1      # latest sample <= t
        idx = np.clip(idx, 0, len(times) - 1)
        return np.stack([self._f6_buffer[i][1] for i in idx], axis=0)

    def _tactile_tensors(self, f6: np.ndarray, deform: Optional[np.ndarray],
                         now: float) -> Dict[str, Optional[torch.Tensor]]:
        if self.cfg.mode != "cascaded":
            return {}
        out: Dict[str, Optional[torch.Tensor]] = {"tactile_f6": None, "tactile_deform": None,
                                                  "tactile_f6_history": None}
        if self.use_f6:
            norm = _normalize(f6.reshape(-1), self.tacf6_mask, self.tacf6_min, self.tacf6_max)
            out["tactile_f6"] = torch.tensor(norm.reshape(1, N_FINGERS, F6_PER_FINGER),
                                             dtype=torch.bfloat16, device=self.device)
        if getattr(self.model, "tactile_vqvae", None) is not None:
            out["tactile_f6_history"] = torch.from_numpy(
                self._f6_history(now).astype(np.float32)).unsqueeze(0).to(self.device)
        if self.use_deform:
            if deform is None:
                raise ValueError(f"{OBS_DEFORM} is required by this checkpoint")
            arr = np.asarray(deform)
            if arr.dtype != np.uint8 or arr.shape != (480, 1200, 3):
                raise ValueError(f"{OBS_DEFORM} must be uint8(480,1200,3), got {arr.dtype}{arr.shape}")
            # Same conversion as the prep's JPEG decode: PIL "L" (ITU-R 601 luma).
            gray = np.asarray(PIL.Image.fromarray(arr, mode="RGB").convert("L"),
                              dtype=np.float32) / 255.0
            tiles = split_deform_strip(gray)                          # [10, 240, 240]
            out["tactile_deform"] = torch.from_numpy(tiles).unsqueeze(0).unsqueeze(2).to(
                self.device, dtype=torch.bfloat16)                    # [1, 10, 1, 240, 240]
        return out

    # ── cross-timestep memory (see qwen_vla/MEMORY_DESIGN.md) ─────────────────
    def _prev_command(self, state: np.ndarray) -> np.ndarray:
        """Best estimate of the executed action for the frame just before
        now -- the cross-timestep memory buffer's fast-tier action_abs
        proxy. Before any chunk has been emitted, falls back to the
        measured state."""
        if self.last_chunk is None:
            return state
        elapsed = max(0.0, time.time() - self.last_chunk_time)
        k = int(round(elapsed * CONTROL_HZ)) - 1
        k = max(0, min(k, self.last_chunk.shape[0] - 1))
        return self.last_chunk[k]

    def _remember_slow(self, now: float, head_image: PIL.Image.Image, task_text: str) -> None:
        """Record this slow tick's own content so a LATER tick can use it as
        memory. Must only be called with content from a tick already fully
        processed -- never the tick currently selecting memory for itself."""
        if not self.memory_slow_seconds:
            return
        self.memory_buf_slow.append((now, head_image.copy(), task_text))
        cutoff = now - (max(self.memory_slow_seconds) + self.memory_buffer_margin_sec)
        self.memory_buf_slow = [e for e in self.memory_buf_slow if e[0] >= cutoff]

    def _remember_fast(self, now: float, fast_images: List[PIL.Image.Image],
                       action_raw: np.ndarray) -> None:
        """Record this tick's fast-content snapshot (wrist images + best-
        known executed action) for a later tick's fast memory."""
        if self.memory_fast <= 0:
            return
        self.memory_buf_fast.append(
            (now, [img.copy() for img in fast_images], np.asarray(action_raw, dtype=np.float64)))
        if len(self.memory_buf_fast) > self.memory_fast:
            self.memory_buf_fast = self.memory_buf_fast[-self.memory_fast:]

    def _select_memory_slow_rows(self, now: float) -> List[Dict[str, Any]]:
        """For each configured lookback target, the buffered entry whose
        timestamp is nearest `now - target` (buffered entries aren't
        uniformly spaced -- inference ticks aren't uniform -- so this is a
        nearest-timestamp snap, not a fixed-position lookup). Returns [] (a
        true no-op) when memory is off or the buffer is still empty (e.g.
        the first slow tick of an episode)."""
        if not self.memory_slow_seconds or not self.memory_buf_slow:
            return []
        rows = []
        for dt_nominal in self.memory_slow_seconds:            # oldest target first
            target_ts = now - dt_nominal
            ts, img, task = min(self.memory_buf_slow, key=lambda e: abs(e[0] - target_ts))
            content = [{"type": "image"}, {"type": "text", "text": task}]
            text = self.processor.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=False,
                add_generation_prompt=True)
            inp = self.processor(text=text, images=[img], return_tensors="pt", padding=False)
            rows.append({
                "input_ids": inp.input_ids.to(self.device),
                "attention_mask": inp.attention_mask.to(self.device),
                "pixel_values": (inp.pixel_values.to(self.device, dtype=torch.bfloat16)
                                if getattr(inp, "pixel_values", None) is not None else None),
                "image_grid_thw": (inp.image_grid_thw.to(self.device)
                                   if getattr(inp, "image_grid_thw", None) is not None else None),
                "dt_actual": torch.tensor([now - ts], dtype=torch.float32, device=self.device),
            })
        return rows

    def _select_memory_fast_rows(self) -> List[Dict[str, Any]]:
        """The last `memory_fast` buffered fast-tick snapshots, oldest first
        -- a linear window, not exponential, matching training (fast ticks
        are close enough together that count ~= time). Returns [] when
        memory is off or nothing buffered yet."""
        if self.memory_fast <= 0 or not self.memory_buf_fast:
            return []
        rows = []
        for ts, fast_images, action_raw in self.memory_buf_fast:
            content = [{"type": "image"} for _ in fast_images]
            text = self.processor.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=False,
                add_generation_prompt=True)
            inp = self.processor(text=text, images=fast_images,
                                 return_tensors="pt", padding=False)
            # action_raw is a single instantaneous absolute action, not a
            # chunk -- action_min/max are kept per-CHUNK-step [T,65], so
            # normalizing this [65] vector against them needs a step index;
            # use step 0's calibration, matching origami_dataset.py's
            # collate_fn convention for the identical fast-memory input.
            action_norm = _normalize(action_raw, self.action_mask,
                                     self.action_min[0], self.action_max[0])
            rows.append({
                "input_ids": inp.input_ids.to(self.device),
                "attention_mask": inp.attention_mask.to(self.device),
                "pixel_values": (inp.pixel_values.to(self.device, dtype=torch.bfloat16)
                                if getattr(inp, "pixel_values", None) is not None else None),
                "image_grid_thw": (inp.image_grid_thw.to(self.device)
                                   if getattr(inp, "image_grid_thw", None) is not None else None),
                "action_abs": torch.tensor(action_norm, dtype=torch.float32,
                                           device=self.device).unsqueeze(0),
            })
        return rows

    def _build_memory_kv(self, now: float):
        """Combined slow+fast memory cache for this tick, or (None, None) (a
        true no-op) when both tiers are off/empty -- forward_flow_action_full/
        _partial's memory_kv=None path is then identical to current
        behavior. Batch size 1 (one set of memory rows regardless of
        n_draws) -- see _expand_memory_kv_batch/_expand_kv_exclude_mask_batch.
        Second return value, kv_exclude_mask, hides redundant memory-row task
        text from attention -- see Qwen3VLVLAModel.build_memory_kv_slow."""
        memory_kv, kv_exclude_mask = None, None
        slow_rows = self._select_memory_slow_rows(now)
        if slow_rows:
            memory_kv, kv_exclude_mask = self.model.build_memory_kv_slow(
                slow_rows, rope_stride=self.memory_rope_stride_slow)
        fast_rows = self._select_memory_fast_rows()
        if fast_rows:
            memory_kv, kv_exclude_mask = self.model.build_memory_kv_fast(
                fast_rows, past_kv=memory_kv, kv_exclude_mask=kv_exclude_mask,
                rope_stride=self.memory_rope_stride_fast)
        return memory_kv, kv_exclude_mask

    @staticmethod
    def _expand_memory_kv_batch(memory_kv, batch_size: int):
        """Broadcast a batch=1 memory cache (build_memory_kv_slow/_fast
        always process one set of memory rows, independent of how many
        parallel noise draws the live tick uses) to match a K-draws batch.

        Not exercised anywhere else in this codebase: the reference server
        (scripts/test.py's CascadedServer) never batches multiple draws, so
        this batch-1-vs-batch-K interaction between memory and
        PolicyConfig.n_draws has no existing implementation to port from --
        written fresh here, mirroring modeling_vla.Qwen3VLVLAModel's own
        `_clone_dynamic_cache` layer-walking so it stays correct across
        both the current (`.layers`/`DynamicLayer`) and pre-4.55
        (`key_cache`/`value_cache`) transformers cache APIs. `.clone()`
        materializes the `.expand()` view into a real batch_size-sized
        tensor (expand alone is a stride-0 view, unsafe to mutate)."""
        if memory_kv is None or batch_size == 1:
            return memory_kv
        from transformers.cache_utils import DynamicLayer
        expanded = type(memory_kv)()
        if hasattr(memory_kv, "layers") and isinstance(memory_kv.layers, list):
            for layer in memory_kv.layers:
                new_layer = DynamicLayer()
                if getattr(layer, "is_initialized", False):
                    new_layer.dtype = layer.dtype
                    new_layer.device = layer.device
                    new_layer.keys = layer.keys.expand(
                        batch_size, *layer.keys.shape[1:]).clone()
                    new_layer.values = layer.values.expand(
                        batch_size, *layer.values.shape[1:]).clone()
                    new_layer.is_initialized = True
                expanded.layers.append(new_layer)
        elif hasattr(memory_kv, "key_cache"):
            expanded.key_cache = [k.expand(batch_size, *k.shape[1:]).clone()
                                  for k in memory_kv.key_cache]
            expanded.value_cache = [v.expand(batch_size, *v.shape[1:]).clone()
                                    for v in memory_kv.value_cache]
            expanded._seen_tokens = getattr(memory_kv, "_seen_tokens", 0)
        else:
            raise NotImplementedError(
                "Unrecognized DynamicCache layout; cannot expand memory_kv batch.")
        return expanded

    @staticmethod
    def _expand_kv_exclude_mask_batch(kv_exclude_mask, batch_size: int):
        """Sibling of `_expand_memory_kv_batch`: broadcast a batch=1
        `kv_exclude_mask` ([1, past_len] bool) to match a K-draws sub-batch,
        the same way `memory_kv` itself gets expanded -- must stay in lockstep
        with it or `build_causal_mask`'s batch dim (kv_exclude_mask) and the
        cache's own batch dim (memory_kv) would disagree."""
        if kv_exclude_mask is None or batch_size == 1:
            return kv_exclude_mask
        return kv_exclude_mask.expand(batch_size, -1).clone()

    def _embed(self, slow_imgs, fast_imgs, state):
        """Slow/fast embedding split, M-RoPE ids, state token (train.py layout)."""
        model, processor = self.model, self.processor
        content = [{"type": "image"} for _ in slow_imgs]
        content.append({"type": "text", "text": self.instruction})
        content += [{"type": "image"} for _ in fast_imgs]
        text = processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        inp = processor(text=text, images=slow_imgs + fast_imgs, return_tensors="pt",
                        padding=False)
        input_ids = inp.input_ids.to(self.device)
        attention_mask = inp.attention_mask.to(self.device)
        pixel_values = inp.pixel_values.to(self.device, dtype=torch.bfloat16)
        grid_thw = inp.image_grid_thw.to(self.device)

        embeds = model.prepare_inputs_embeds(input_ids=input_ids, pixel_values=pixel_values,
                                             image_grid_thw=grid_thw)
        merge = getattr(model.visual, "spatial_merge_size", 2)
        n_slow_tokens = sum(int(g[0] * (g[1] // merge) * (g[2] // merge))
                            for g in grid_thw[:len(slow_imgs)])
        slow, fast = split_slow_fast_embeds(embeds, input_ids, model.image_token_id,
                                            n_slow_tokens)
        position_ids, _ = model.get_rope_index(input_ids=input_ids, image_grid_thw=grid_thw,
                                               attention_mask=attention_mask)
        position_ids = position_ids[:, :, :slow.shape[1]]
        if model.n_flare_tokens > 0:
            flare = model.flare_queries.expand(1, -1, -1).to(device=slow.device, dtype=slow.dtype)
            slow = torch.cat([slow, flare], dim=1)
            position_ids = extend_position_ids_for_flare(position_ids, model.n_flare_tokens)
        state_emb = None
        if model.use_robot_state and self.use_state:
            norm = _normalize(state.astype(np.float32), self.state_mask,
                              self.state_min, self.state_max)
            vec = torch.tensor(norm, dtype=slow.dtype, device=self.device).unsqueeze(0)
            state_emb = model.state_embedder(vec).unsqueeze(1)
        return slow, position_ids, fast, attention_mask, state_emb

    def _flow_slow(self, slow, pos, fast, mask, state_emb, tactile, timing, cache: bool,
                   now: float, memory_kv=None, kv_exclude_mask=None) -> np.ndarray:
        """K draws from a fresh vision-language prefix -> normalised [K, T, D].

        When `cache` is True (tactile_refine_every > 1), stashes the slow
        pass's KV + prefix position ids/mask so later ticks can skip straight
        to `_flow_cached` -- only valid when K fits in one `max_flow_batch`
        (enforced in __init__), since caching mid-loop would only keep the
        last sub-batch's state.

        `memory_kv` (from `_build_memory_kv`) is batch=1; each sub-batch
        below expands it to that sub-batch's own width via
        `_expand_memory_kv_batch` before use -- forward_flow_action_full/
        _partial clone it again internally (their own defensive copy against
        cross-call mutation), so handing them a fresh per-sub-batch expansion
        each iteration is safe and never mutates the shared `memory_kv`.
        """
        K = self.cfg.n_draws
        model = self.model
        rep = lambda t, dim=0: None if t is None else t.repeat_interleave(K, dim=dim)
        slow_r, fast_r, mask_r, state_r = rep(slow), rep(fast), rep(mask), rep(state_emb)
        pos_r = rep(pos, dim=1)
        tac_r = {k: rep(v) for k, v in tactile.items()}
        noise = torch.randn(K, self.action_chunk, self.action_dim, dtype=torch.bfloat16,
                            device=self.device, generator=self.generator)
        outs = []
        for s in range(0, K, self.cfg.max_flow_batch):
            sl = slice(s, min(s + self.cfg.max_flow_batch, K))
            mem_sl = self._expand_memory_kv_batch(memory_kv, sl.stop - sl.start)
            excl_sl = self._expand_kv_exclude_mask_batch(kv_exclude_mask, sl.stop - sl.start)
            if self.cfg.mode == "blind":
                outs.append(model.forward_flow_action_full(
                    inputs_embeds=slow_r[sl], position_ids=pos_r[:, sl],
                    attention_mask=mask_r[sl], noise=noise[sl],
                    state_embeds=None if state_r is None else state_r[sl],
                    fast_embeds=fast_r[sl], num_steps=self.cfg.total_steps,
                    memory_kv=mem_sl, kv_exclude_mask=excl_sl))
                continue
            ts = time.time()
            x_split, kv, n_act, tau = model.forward_flow_action_partial(
                inputs_embeds=slow_r[sl], position_ids=pos_r[:, sl],
                attention_mask=mask_r[sl], noise=noise[sl],
                state_embeds=None if state_r is None else state_r[sl],
                fast_embeds=fast_r[sl], num_steps_total=self.cfg.total_steps,
                split_step=self.cfg.split_step, refresh_clean_kv=True,
                memory_kv=mem_sl, kv_exclude_mask=excl_sl)
            self._sync()
            timing["slow_ms"] = timing.get("slow_ms", 0.0) + 1000 * (time.time() - ts)
            tf = time.time()
            outs.append(model.tactile_flow_continue(
                cached_kv=kv, latent_position_ids=pos_r[:, sl], n_action_in_cache=n_act,
                x_split=x_split, tau_split=tau, attention_mask=mask_r[sl],
                num_steps_total=self.cfg.total_steps, split_step=self.cfg.split_step,
                kv_exclude_mask=excl_sl,
                **{k: (None if v is None else v[sl]) for k, v in tac_r.items()}))
            self._sync()
            timing["fast_ms"] = timing.get("fast_ms", 0.0) + 1000 * (time.time() - tf)
            if cache:
                self._cache_kv, self._cache_pos, self._cache_mask = kv, pos_r, mask_r
                self._cache_kv_exclude_mask = excl_sl
                self._cache_x_split, self._cache_n_action = x_split, n_act
                self._cache_tau_split, self._cache_built_at = tau, now
        return torch.cat(outs, dim=0).float().cpu().numpy().astype(np.float64)

    def _flow_cached(self, tactile, timing) -> np.ndarray:
        """K draws continuing the tactile expert from a cached slow-tick KV.

        Reuses `self._cache_*` from the last `_flow_slow(..., cache=True)`
        call verbatim -- no vision-language forward pass, and blind to any
        new camera images in this tick's own observation, since
        `tactile_flow_continue` takes no vision input at all (only cached
        KV + fresh tactile). `cached_kv` is cloned inside that call, so the
        same stored object can be reused across many separate ticks safely.
        """
        K = self.cfg.n_draws
        model = self.model
        tac_r = {k: (None if v is None else v.repeat_interleave(K, dim=0))
                for k, v in tactile.items()}
        tf = time.time()
        out = model.tactile_flow_continue(
            cached_kv=self._cache_kv, latent_position_ids=self._cache_pos,
            n_action_in_cache=self._cache_n_action, x_split=self._cache_x_split,
            tau_split=self._cache_tau_split, attention_mask=self._cache_mask,
            num_steps_total=self.cfg.total_steps, split_step=self.cfg.split_step,
            kv_exclude_mask=self._cache_kv_exclude_mask,
            **tac_r)
        self._sync()
        timing["fast_ms"] = 1000 * (time.time() - tf)
        return out.float().cpu().numpy().astype(np.float64)

    def _reconstruct(self, norm_mean: np.ndarray, state: np.ndarray) -> np.ndarray:
        absolute = denormalize(norm_mean, self.action_mask, self.action_min, self.action_max)
        absolute = clamp_frozen_absolute(absolute, self.action_mask, state)
        if self.projector is not None:
            absolute = self.projector.project(state, absolute)
        actions = np.ascontiguousarray(absolute, dtype=np.float32)
        if actions.shape != (self.action_chunk, self.action_dim) or not np.isfinite(actions).all():
            raise RuntimeError("policy produced an invalid action chunk")
        # Best-known executed chunk, for the fast-memory buffer's
        # action_abs proxy (_prev_command) -- the actual, safety-projected
        # values, since that's what's really sent to the robot.
        self.last_chunk = actions.astype(np.float64)
        self.last_chunk_time = time.time()
        return actions

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)


# ── helpers shared by the server and the replay ───────────────────────────────
def synthetic_observation(seed: int = 0) -> Dict[str, Any]:
    """A schema-complete observation with plausible values, for warm-up/tests."""
    rng = np.random.default_rng(seed)
    img = lambda h, w: rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    return {
        OBS_HEAD_LEFT: img(224, 224), OBS_HEAD_RIGHT: img(224, 224),
        OBS_WRIST_LEFT: img(224, 224), OBS_WRIST_RIGHT: img(224, 224),
        OBS_STATE: np.zeros(ACTION_DIM, dtype=np.float32),
        OBS_TORQUE: np.zeros(ACTION_DIM, dtype=np.float32),
        OBS_TACTILE: np.zeros(60, dtype=np.float32),
        OBS_DEFORM: img(480, 1200),
        OBS_TACTILE_RAW: np.zeros((480, 1600, 3), dtype=np.uint8),
        "prompt": "fold the plane",
    }


def validator_observation() -> Dict[str, Any]:
    """Byte-for-byte the pattern `check_zenoh_policy.py` sends: gradient images,
    linspace state, all-zero torque / tactile / deform / raw."""
    rows = np.arange(224, dtype=np.uint8)[:, None]
    cols = np.arange(224, dtype=np.uint8)[None, :]
    base = np.empty((224, 224, 3), dtype=np.uint8)
    base[..., 0] = rows
    base[..., 1] = cols
    base[..., 2] = rows ^ cols
    return {
        OBS_HEAD_LEFT: np.ascontiguousarray(base),
        OBS_HEAD_RIGHT: np.ascontiguousarray(np.roll(base, 11, axis=0)),
        OBS_WRIST_LEFT: np.ascontiguousarray(np.roll(base, 17, axis=1)),
        OBS_WRIST_RIGHT: np.ascontiguousarray(np.flip(base, axis=1)),
        OBS_STATE: np.linspace(-0.25, 0.25, ACTION_DIM, dtype=np.float32),
        OBS_TORQUE: np.zeros(ACTION_DIM, dtype=np.float32),
        OBS_TACTILE: np.zeros(60, dtype=np.float32),
        OBS_DEFORM: np.zeros((480, 1200, 3), dtype=np.uint8),
        OBS_TACTILE_RAW: np.zeros((480, 1600, 3), dtype=np.uint8),
        "prompt": "origami synthetic protocol check",
    }


def varied_observation(seed: int) -> Dict[str, Any]:
    """Random *content* everywhere (images, state, wrench, deform) -- the
    pattern a real, differently-valued query presents."""
    rng = np.random.default_rng(seed)
    img = lambda h, w: rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    return {
        OBS_HEAD_LEFT: img(224, 224), OBS_HEAD_RIGHT: img(224, 224),
        OBS_WRIST_LEFT: img(224, 224), OBS_WRIST_RIGHT: img(224, 224),
        OBS_STATE: rng.uniform(-0.25, 0.25, ACTION_DIM).astype(np.float32),
        OBS_TORQUE: np.zeros(ACTION_DIM, dtype=np.float32),
        OBS_TACTILE: rng.uniform(0, 2, 60).astype(np.float32),
        OBS_DEFORM: img(480, 1200),
        "prompt": "fold the plane",
    }


def add_policy_arguments(parser) -> None:
    """CLI flags for `PolicyConfig`, shared by the server and the replay."""
    parser.add_argument("--checkpoint_path", default=os.environ.get("TREX_CKPT_PATH", ""))
    env = os.environ.get
    parser.add_argument("--mode", choices=["cascaded", "blind"], default=env("TREX_MODE", "cascaded"))
    parser.add_argument("--total_steps", type=int, default=int(env("TREX_TOTAL_STEPS", "10")))
    parser.add_argument("--split_step", type=int, default=int(env("TREX_SPLIT_STEP", "6")))
    parser.add_argument("--n_draws", type=int, default=int(env("TREX_N_DRAWS", "8")))
    parser.add_argument("--max_flow_batch", type=int, default=64)
    parser.add_argument("--safety", choices=["tol", "urdf", "off", "none"],
                        default=env("TREX_SAFETY", "tol"))
    parser.add_argument("--rate_margin", type=float, default=0.999)
    parser.add_argument("--limits_path", default=DEFAULT_LIMITS)
    parser.add_argument("--instruction", default="")
    parser.add_argument("--tactile_history", choices=["resample", "tile"], default="resample")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no_warmup", action="store_true")
    parser.add_argument("--compile", type=int, default=int(os.environ.get("TREX_COMPILE", "0")),
                        help="1 = torch.compile the flow functions (default from $TREX_COMPILE)")
    parser.add_argument("--compile_mode", default="default")
    parser.add_argument("--tactile_refine_every", type=int,
                        default=int(env("TREX_TACTILE_REFINE_EVERY", "1")),
                        help="1 = always full slow+fast (default); N>1 = reuse cached slow-tick "
                             "KV for (N-1)/N ticks, running the tactile expert only")
    parser.add_argument("--tactile_refine_max_stale_s", type=float,
                        default=float(env("TREX_TACTILE_REFINE_MAX_STALE_S", "0.5")))
    parser.add_argument("--memory_slow_seconds", default=env("TREX_MEMORY_SLOW_SECONDS", ""),
                        help="comma-separated seconds-back targets, e.g. '0.25,0.5,1,5'; "
                             "'' auto-detects from training_args.json (default)")
    parser.add_argument("--memory_fast", type=int, default=int(env("TREX_MEMORY_FAST", "0")),
                        help="linear fast-memory window in ticks; 0 auto-detects from "
                             "training_args.json (default)")
    parser.add_argument("--memory_rope_stride_slow", type=float,
                        default=(float(env("TREX_MEMORY_ROPE_STRIDE_SLOW"))
                                if env("TREX_MEMORY_ROPE_STRIDE_SLOW") else None))
    parser.add_argument("--memory_rope_stride_fast", type=float,
                        default=(float(env("TREX_MEMORY_ROPE_STRIDE_FAST"))
                                if env("TREX_MEMORY_ROPE_STRIDE_FAST") else None))
    parser.add_argument("--memory_buffer_margin_sec", type=float,
                        default=float(env("TREX_MEMORY_BUFFER_MARGIN_SEC", "2.0")))
    parser.add_argument("--disable_memory", action="store_true",
                        default=bool(int(env("TREX_DISABLE_MEMORY", "0"))),
                        help="force memory off regardless of training_args.json/checkpoint "
                             "auto-detect -- for A/B diagnosis only, not a normal serving knob")


def config_from_args(args) -> PolicyConfig:
    return PolicyConfig(
        checkpoint_path=args.checkpoint_path, mode=args.mode, total_steps=args.total_steps,
        split_step=args.split_step, n_draws=args.n_draws, max_flow_batch=args.max_flow_batch,
        safety=args.safety, rate_margin=args.rate_margin,
        limits_path=args.limits_path, instruction=args.instruction,
        tactile_history=args.tactile_history, device=args.device, seed=args.seed,
        warmup=not args.no_warmup, compile=bool(args.compile), compile_mode=args.compile_mode,
        tactile_refine_every=args.tactile_refine_every,
        tactile_refine_max_stale_s=args.tactile_refine_max_stale_s,
        memory_slow_seconds=args.memory_slow_seconds, memory_fast=args.memory_fast,
        memory_rope_stride_slow=args.memory_rope_stride_slow,
        memory_rope_stride_fast=args.memory_rope_stride_fast,
        memory_buffer_margin_sec=args.memory_buffer_margin_sec,
        disable_memory=args.disable_memory)
