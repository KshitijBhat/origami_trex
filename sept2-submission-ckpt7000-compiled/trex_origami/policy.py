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
* **Anchoring.** The 14 arm dims are deltas from the *previous command*.  On the
  robot that is the measured state (`anchor_source="state"`, what the 15%
  anchor-dropout in training taught the policy to accept), the state plus the
  prep's mean command-minus-state tracking offset (`"state_offset"`, the
  expected previous command -- the default, see eval_deploy), or this policy's
  own last chunk indexed by elapsed time (`"self"`).  Everything else is
  absolute.  Frozen dims (torso 58/59 in this checkpoint) are held at the
  measured state.
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
from .anchoring import (ANCHOR_PREV_COMMAND, build_anchor,  # noqa: E402
                        describe as describe_anchor, masks as anchor_masks,
                        spec_from_meta)
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
    anchor_source: str = "state_offset"  # "state" | "state_offset" | "self"
    stale_anchor_s: float = 2.0       # fall back to state if own chunk is older
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
        self.state_mask = np.asarray(st["state_mask"], dtype=bool)
        self.state_min, self.state_max = st["state_min"], st["state_max"]
        self.tacf6_mask = np.asarray(st["tacf6_mask"], dtype=bool)
        self.tacf6_min, self.tacf6_max = st["tacf6_min"], st["tacf6_max"]
        self.frozen_dims = frozen_action_dims(self.action_mask)
        self.te_mean = st["tracking_error_mean"]
        self.anchor_spec = tuple(st["action_anchor"])
        self.needs_prev = bool(anchor_masks(self.anchor_spec)[ANCHOR_PREV_COMMAND].any())

        self.instruction = (config.instruction or self.train_args.get("instruction")
                            or DATASET_TASK_STRING)
        size = self.train_args.get("image_size") or [224, 224]
        self.image_size = (int(size[0]), int(size[1]))
        self.vqvae_window = int(getattr(getattr(self.model, "tactile_vqvae", None), "cfg",
                                        SimpleNamespace(window=16)).window)
        self.use_state = bool(self.train_args.get("use_robot_state", 1))
        self.use_f6 = bool(self.train_args.get("use_tactile_vec", 1))
        self.use_deform = bool(self.train_args.get("use_tactile_deform", 1))

        self.projector = None
        if config.safety not in ("none", None):
            self.projector = SafetyProjector(config.limits_path, config.safety,
                                             config.rate_margin)
        self.generator = None
        if config.seed is not None:
            self.generator = torch.Generator(device=self.device).manual_seed(config.seed)

        # episode-scoped state
        self._f6_buffer: List[Tuple[float, np.ndarray]] = []
        self._last_chunk: Optional[np.ndarray] = None
        self._last_obs_time: Optional[float] = None
        self.last_timing: Dict[str, float] = {}
        self.n_infer = 0

        print(f"[policy] {describe_anchor(self.anchor_spec)}; frozen dims "
              f"{self.frozen_dims.tolist()}; prompt {self.instruction!r}; "
              f"mode={config.mode} steps={config.total_steps}/{config.split_step} "
              f"K={config.n_draws} anchor={config.anchor_source} safety={config.safety} "
              f"compile={config.compile}")
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
        if not train_args.get("action_anchor"):
            statistic["action_anchor"] = list(spec_from_meta(train_args))
        # Mean command-minus-state offset the prep measured, per dim: the
        # expected previous command given only the measured state.
        with open(os.path.join(self.cfg.checkpoint_path, "stats_data.json")) as handle:
            raw = json.load(handle)
        block = raw[next(iter(raw))]
        statistic["tracking_error_mean"] = np.asarray(
            block.get("tracking_error", {}).get("mean", np.zeros(load_args.action_dim)),
            dtype=np.float64)
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
            self._last_chunk = None
            self._last_obs_time = None

    def infer(self, observation: Dict[str, Any], now: Optional[float] = None) -> np.ndarray:
        """Full public observation -> float32[T, 65] absolute radians.

        `now` is the observation time in seconds; it defaults to wall-clock.
        An offline replay passes the frame timestamp so the tactile-history
        resampling and the self-anchor index see replay time, not CPU time.
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

        slow_imgs = [self._pil(obs[OBS_HEAD_LEFT])]
        fast_imgs = [self._pil(obs[OBS_WRIST_RIGHT]), self._pil(obs[OBS_WRIST_LEFT])]
        tactile = self._tactile_tensors(f6, obs.get(OBS_DEFORM), now)
        timing["prep_ms"] = 1000 * (time.time() - t0)

        t1 = time.time()
        slow, pos, fast, mask, state_emb = self._embed(slow_imgs, fast_imgs, state)
        self._sync()
        timing["embed_ms"] = 1000 * (time.time() - t1)

        t2 = time.time()
        norm = self._flow(slow, pos, fast, mask, state_emb, tactile, timing)
        timing["flow_ms"] = 1000 * (time.time() - t2)

        t3 = time.time()
        actions = self._reconstruct(norm.mean(axis=0), state, now)
        timing["post_ms"] = 1000 * (time.time() - t3)
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

    def _flow(self, slow, pos, fast, mask, state_emb, tactile, timing) -> np.ndarray:
        """K draws from one prefix -> normalised [K, T, D] (float64 numpy)."""
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
            if self.cfg.mode == "blind":
                outs.append(model.forward_flow_action_full(
                    inputs_embeds=slow_r[sl], position_ids=pos_r[:, sl],
                    attention_mask=mask_r[sl], noise=noise[sl],
                    state_embeds=None if state_r is None else state_r[sl],
                    fast_embeds=fast_r[sl], num_steps=self.cfg.total_steps))
                continue
            ts = time.time()
            x_split, kv, n_act, tau = model.forward_flow_action_partial(
                inputs_embeds=slow_r[sl], position_ids=pos_r[:, sl],
                attention_mask=mask_r[sl], noise=noise[sl],
                state_embeds=None if state_r is None else state_r[sl],
                fast_embeds=fast_r[sl], num_steps_total=self.cfg.total_steps,
                split_step=self.cfg.split_step, refresh_clean_kv=True)
            self._sync()
            timing["slow_ms"] = timing.get("slow_ms", 0.0) + 1000 * (time.time() - ts)
            tf = time.time()
            outs.append(model.tactile_flow_continue(
                cached_kv=kv, latent_position_ids=pos_r[:, sl], n_action_in_cache=n_act,
                x_split=x_split, tau_split=tau, attention_mask=mask_r[sl],
                num_steps_total=self.cfg.total_steps, split_step=self.cfg.split_step,
                **{k: (None if v is None else v[sl]) for k, v in tac_r.items()}))
            self._sync()
            timing["fast_ms"] = timing.get("fast_ms", 0.0) + 1000 * (time.time() - tf)
        return torch.cat(outs, dim=0).float().cpu().numpy().astype(np.float64)

    def _prev_command(self, state: np.ndarray, now: float) -> np.ndarray:
        """The command the robot was on one frame before this observation."""
        if self.cfg.anchor_source == "state":
            return state
        fallback = state + self.te_mean          # expected previous command
        if (self.cfg.anchor_source == "state_offset" or self._last_chunk is None
                or self._last_obs_time is None):
            return fallback
        elapsed = now - self._last_obs_time
        if elapsed < 0 or elapsed > self.cfg.stale_anchor_s:
            return fallback
        # The Gateway aligns chunk row 0 with the observation it came from, so
        # row (frames since that observation) - 1 is the previous frame's command.
        k = int(round(elapsed * CONTROL_HZ)) - 1
        k = max(0, min(k, self._last_chunk.shape[0] - 1))
        return self._last_chunk[k]

    def _reconstruct(self, norm_mean: np.ndarray, state: np.ndarray, now: float) -> np.ndarray:
        rel = denormalize(norm_mean, self.action_mask, self.action_min, self.action_max)
        anchor = build_anchor(state, self._prev_command(state, now), self.anchor_spec)
        absolute = clamp_frozen_absolute(rel + anchor[None, :], self.action_mask, state)
        if self.projector is not None:
            absolute = self.projector.project(state, absolute)
        self._last_chunk = absolute
        self._last_obs_time = now
        actions = np.ascontiguousarray(absolute, dtype=np.float32)
        if actions.shape != (self.action_chunk, self.action_dim) or not np.isfinite(actions).all():
            raise RuntimeError("policy produced an invalid action chunk")
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
    parser.add_argument("--anchor_source", choices=["state", "state_offset", "self"],
                        default=env("TREX_ANCHOR_SOURCE", "state_offset"))
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


def config_from_args(args) -> PolicyConfig:
    return PolicyConfig(
        checkpoint_path=args.checkpoint_path, mode=args.mode, total_steps=args.total_steps,
        split_step=args.split_step, n_draws=args.n_draws, max_flow_batch=args.max_flow_batch,
        anchor_source=args.anchor_source, safety=args.safety, rate_margin=args.rate_margin,
        limits_path=args.limits_path, instruction=args.instruction,
        tactile_history=args.tactile_history, device=args.device, seed=args.seed,
        warmup=not args.no_warmup, compile=bool(args.compile), compile_mode=args.compile_mode)
