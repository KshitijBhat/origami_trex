"""Deployment-configuration eval for the T-Rex origami policy.

`eval_smoothed.py` settled two things: the cascaded path beats the blind one,
and averaging K flow draws removes most of the sampling variance.  This script
answers the questions that remain before the policy goes behind the
competition's `origami-zenoh-v1` boundary, all on held-out val seasons:

  flow steps      the cascaded schedule is 10 Euler steps split 6/4.  Request
                  latency scales with the step count, and the organizer's async
                  Gateway aligns each chunk to the *observation* it was computed
                  from -- a 25-step chunk that arrives 30 frames late is already
                  entirely in the past.  `5:3` keeps tau_split = 0.4 (what the
                  tactile expert trained on) at half the transformer passes.

  prompt          the checkpoint records no `instruction`, so it trained on the
                  dataset's own "north ces task".  Scored against the current
                  trainer default as well, to make sure serving pins the right
                  string.

  anchor source   the 14 arm dims are deltas from the *previous command*.  The
                  offline evals hand the model the teleoperator's previous
                  command (an oracle).  On the robot the policy only has (a) the
                  measured state or (b) its own last emitted chunk, and under
                  async execution the Gateway blends several chunks so even (b)
                  is not what the robot was told.  All three are scored.  The
                  anchor enters additively at the output, so every variant is a
                  reconstruction of the *same* draws -- no extra GPU cost, and
                  the comparison is exactly paired.

  safety          the sequential URDF/step-jump/velocity projection from
                  eval_smoothed, on and off.

  gateway replay  a receding-horizon rollout on contiguous rows of held-out
                  episodes (`val_stride5`, one row per 5 frames) that replays
                  the kit's own `TemporalEnsembler` (agg_n=4, exp_k=0.01,
                  hold_last) exactly as `openpi_origami_async.py` runs it: a
                  chunk computed from frame f_obs becomes available L frames
                  later, is aligned to start at f_obs, and the next inference
                  starts only after the previous one returned.  Swept over L, so
                  the number that comes out is "executed-stream error at this
                  request latency" -- the quantity a step-count / K choice
                  actually trades against.  The `latest` aggregation (agg_n=1)
                  approximates the sync Gateway path.

Everything is open-loop against teleop data: it ranks inference configurations,
it is not a folding success rate.  The `self` anchor in particular is
pessimistic here -- on the robot the state follows our commands, in a replay it
follows the teleoperator's.

Usage:
    python scripts/eval_deploy.py \
        --checkpoint_path /workspace/outputs/checkpoint-2-7000 \
        --origami_root    /workspace/data/origami_trex/origami_flat/full/val \
        --rollout_root    /workspace/data/origami_trex/origami_flat/full/val_stride5 \
        --out_dir         /workspace/eval/full/checkpoint-2-7000/deploy \
        --urdf            /workspace/urdf/north_poc2_2_v3_1.urdf
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import os
import sys
import time
from collections import defaultdict
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from qwen_vla.origami_dataset import (JOINT_GROUPS, OrigamiDataset,
                                      clamp_frozen_absolute, denormalize,
                                      frozen_action_dims)
from trex_origami.anchoring import build_anchor, describe as describe_anchor, to_absolute

RAD2DEG = 180.0 / math.pi
KIT_ASYNC = os.environ.get(
    "ORIGAMI_KIT_ASYNC",
    "/workspace/origami_trex/origami-inference-kit-participant/"
    "sharpa_north_ces_lite_sdk-main/examples/openpi_origami_async.py")


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module          # dataclasses in the module need this
    spec.loader.exec_module(module)
    return module


EV = ES = None      # eval_origami / eval_smoothed, loaded in main()


# ── gateway replay ────────────────────────────────────────────────────────────
class GatewayReplay:
    """Replay the kit's async temporal aggregation over one episode.

    `chunk_rel` [R, T, D] are the *target-space* chunks (arm deltas, absolute
    hands) predicted from rows 0..R-1, each row `gap` frames apart.  With
    request latency `latency` (frames) the inference thread replans every
    `m = max(1, ceil(latency / gap))` rows, and each chunk is pushed into the
    ensembler `latency` frames after its observation with
    `offset_steps = latency`, i.e. start_step = f_obs, which is the kit's
    "auto" compensation.
    """

    def __init__(self, ensembler_cls, anchor_spec, action_mask, projector,
                 gap: int, te_mean: np.ndarray):
        self.ensembler_cls = ensembler_cls
        self.spec = anchor_spec
        self.mask = action_mask
        self.projector = projector
        self.gap = gap
        self.te_mean = te_mean

    def run(self, chunk_rel: np.ndarray, state_rows: np.ndarray,
            prev_cmd_rows: np.ndarray, latency: int, anchor_mode: str,
            agg_n: int, exp_k: float, safe: bool) -> Tuple[np.ndarray, np.ndarray]:
        R, T, D = chunk_rel.shape
        gap = self.gap
        n_frames = R * gap
        m = max(1, math.ceil(latency / gap)) if latency > 0 else 1
        pending: Dict[int, int] = {gap * r + latency: r for r in range(0, R, m)}
        ens = self.ensembler_cls(max_chunks=16, agg_n=agg_n, exp_k=exp_k,
                                 hold_last=True)
        executed = np.empty((n_frames, D), dtype=np.float64)
        covered = np.zeros(n_frames, dtype=bool)
        last_chunk = None
        last_start = 0
        for f in range(n_frames):
            ens.set_current_step(f)
            r = pending.get(f)
            if r is not None:
                f_obs = gap * r
                state = state_rows[r]
                if anchor_mode == "oracle":
                    prev = prev_cmd_rows[r]
                elif anchor_mode == "state":
                    prev = state
                elif anchor_mode == "state_offset":
                    prev = state + self.te_mean
                elif anchor_mode == "self":
                    # Row of our own last chunk that the robot was on one frame
                    # before this observation (kit alignment: row 0 <-> f_obs).
                    if last_chunk is None:
                        prev = state
                    else:
                        idx = int(np.clip(f_obs - last_start - 1, 0, T - 1))
                        prev = last_chunk[idx]
                else:
                    raise ValueError(anchor_mode)
                absolute = chunk_rel[r] + build_anchor(state, prev, self.spec)[None, :]
                absolute = clamp_frozen_absolute(absolute, self.mask, state)
                if safe and self.projector is not None:
                    absolute = self.projector.project(state, absolute)
                last_chunk, last_start = absolute, f_obs
                ens.push_chunk({"a": absolute.astype(np.float32)},
                               offset_steps=latency)
            # A frame is covered only if some chunk really spans it; pop_step's
            # hold_last otherwise repeats the previous action, which is what the
            # Gateway does too but must be counted as a stale frame here.
            covered[f] = any(c.start_step <= f < c.end_step_exclusive for c in ens._chunks)
            step = ens.pop_step(f)
            executed[f] = state_rows[0] if step is None else step["a"]
        return executed, covered


# ── main ──────────────────────────────────────────────────────────────────────
def main(argv: Optional[Sequence[str]] = None) -> int:
    global EV, ES
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--origami_root", required=True,
                        help="strided val split for the chunk pass")
    parser.add_argument("--rollout_root", default="",
                        help="dense (stride 5) val split for the gateway replay; "
                             "empty = skip the rollout pass")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--urdf", default="/workspace/urdf/north_poc2_2_v3_1.urdf")
    parser.add_argument("--prompts", nargs="*", default=[""],
                        help="prompts to score; '' = what the checkpoint/dataset "
                             "declares.  Only the first cascaded schedule is run "
                             "for the non-default prompts")
    parser.add_argument("--steps", nargs="+", default=["10:6", "5:3"],
                        help="cascaded schedules total:split")
    parser.add_argument("--blind_steps", type=int, default=10,
                        help="steps for the action-expert-only reference; 0 = skip")
    parser.add_argument("--draws", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--anchors", nargs="+", default=["oracle", "state", "state_offset"],
                        choices=["oracle", "state", "state_offset"],
                        help="anchor sources for the chunk pass (self needs a rollout). "
                             "state_offset = state + the prep's mean command-minus-state "
                             "tracking offset, i.e. the expected previous command")
    parser.add_argument("--num_eval_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_flow_batch", type=int, default=64)
    parser.add_argument("--rollout_episodes", type=int, default=6)
    parser.add_argument("--rollout_rows", type=int, default=150)
    parser.add_argument("--rollout_draws", type=int, default=8)
    parser.add_argument("--latencies", type=int, nargs="+",
                        default=[0, 5, 10, 15, 20, 25, 30],
                        help="request latencies in 30 Hz frames for the replay")
    parser.add_argument("--agg_n", type=int, default=4)
    parser.add_argument("--exp_k", type=float, default=0.01)
    parser.add_argument("--clamp_positions", choices=["tol", "urdf", "off"], default="tol")
    parser.add_argument("--rate_margin", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip_chunk", action="store_true")
    args = parser.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    draws = sorted(set(args.draws))

    EV = _load("trex_eval_origami", os.path.join(_SCRIPT_DIR, "eval_origami.py"))
    ES = _load("trex_eval_smoothed", os.path.join(_SCRIPT_DIR, "eval_smoothed.py"))
    ES.EV = EV
    kit = _load("openpi_origami_async", KIT_ASYNC)
    test_mod = _load("trex_test_server", os.path.join(_SCRIPT_DIR, "test.py"))

    schedules = []
    for item in args.steps:
        total, split = (int(x) for x in item.split(":"))
        schedules.append(("cascaded", total, split))
    configs = list(schedules)
    if args.blind_steps > 0:
        configs.append(("blind", args.blind_steps, 0))

    def tag(config) -> str:
        mode, total, split = config
        return f"{mode[:3]}{total}" if mode == "cascaded" else f"blind{total}"

    # ── model ─────────────────────────────────────────────────────────────────
    with open(os.path.join(args.checkpoint_path, "training_args.json")) as handle:
        saved = json.load(handle)
    load_args = SimpleNamespace(
        checkpoint_path=args.checkpoint_path, base_model_path="", stats_path="",
        dataset_name="", action_dim=int(saved.get("action_dim", 65)),
        action_chunk=int(saved.get("action_chunk", 25)),
        use_robot_state=int(saved.get("use_robot_state", 1)),
        use_tactile_vec=int(saved.get("use_tactile_vec", 1)),
        use_tactile_deform=int(saved.get("use_tactile_deform", 1)),
        tactile_intermediate_size=0, n_flare_tokens_per_frame=0, n_flare_steps=0,
        use_tactile_code=0, vqvae_codebook_size=64, use_tactile_vqvae=0,
        vqvae_config=None, cascaded_total_steps=10, cascaded_split_step=6)
    model, processor, statistic = test_mod.model_load(load_args)
    model = model.to(device).eval()
    with open(os.path.join(args.checkpoint_path, "stats_data.json")) as handle:
        ckpt_stats = json.load(handle)
    te_block = ckpt_stats[next(iter(ckpt_stats))].get("tracking_error", {})
    te_mean = np.asarray(te_block.get("mean", np.zeros(65)), dtype=np.float64)
    print(f"mean command-minus-state tracking offset: arms "
          f"{RAD2DEG * np.abs(te_mean[list(range(7)) + list(range(29, 36))]).mean():.3f} deg")

    checker = projector = None
    if os.path.exists(args.urdf):
        checker = EV.SafetyChecker(args.urdf)
        projector = ES.SafetyProjector(checker, args.clamp_positions, args.rate_margin)
    else:
        print(f"WARNING: no URDF at {args.urdf}; safety checks/projection disabled")

    def make_dataset(root: str, prompt: str) -> OrigamiDataset:
        overrides = {"instruction": prompt} if prompt else {}
        config = EV.dataset_config_from_checkpoint(args.checkpoint_path, overrides)
        # Normalise with the checkpoint's own stats, as serving does, rather
        # than with whatever the split's meta/norm_stats.json says.
        ds = OrigamiDataset(config, processor, EV._Printer(), root=root,
                            _stats=ckpt_stats)
        EV.check_anchor_consistency(args.checkpoint_path, ds)
        return ds

    generator = torch.Generator(device=device).manual_seed(args.seed)
    payload: dict = {
        "checkpoint": args.checkpoint_path,
        "configs": [f"{tag(c)}={c}" for c in configs],
        "draws": draws,
        "safety_projection": {"position_mode": args.clamp_positions,
                              "rate_margin": args.rate_margin,
                              "enabled": projector is not None},
    }

    # ═══════════════════ pass 1: chunk-level, strided ═════════════════════════
    chunk_reports: Dict[str, dict] = {}
    if not args.skip_chunk:
        accs: Dict[str, "EV.ErrorAccumulator"] = {}
        smooths: Dict[str, ES.Smoothness] = {}

        def acc(name: str, horizon: int, dim: int):
            if name not in accs:
                accs[name] = EV.ErrorAccumulator(horizon, dim)
                smooths[name] = ES.Smoothness()
            return accs[name]

        for p_i, prompt in enumerate(args.prompts):
            dataset = make_dataset(args.origami_root, prompt)
            horizon, dim = dataset.action_chunk, dataset.action_dim
            mask = dataset.action_mask
            amin, amax = dataset.action_min, dataset.action_max
            spec = dataset.anchor_spec
            frozen = frozen_action_dims(mask)
            suffix = "" if p_i == 0 else f"_p{p_i}"
            run_configs = configs if p_i == 0 else configs[:1]
            print(f"\nchunk pass prompt[{p_i}] = {dataset.effective_instruction!r}; "
                  f"configs {[tag(c) for c in run_configs]}; anchoring "
                  f"{describe_anchor(spec)}; frozen dims {frozen.tolist()}")

            if args.num_eval_samples and args.num_eval_samples < len(dataset):
                idx = np.unique(np.linspace(0, len(dataset) - 1,
                                            args.num_eval_samples).astype(int))
                eval_set = Subset(dataset, idx.tolist())
            else:
                eval_set = dataset
            loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers,
                                collate_fn=dataset.collate_fn, pin_memory=True)
            started = time.time()
            for step, batch in enumerate(loader):
                state = batch["eval_state"].numpy().astype(np.float64)
                prev_cmd = batch["eval_prev_command"].numpy().astype(np.float64)
                contact = batch.get("eval_contact")
                contact = None if contact is None else contact.numpy()
                anchor_gt = build_anchor(state, prev_cmd, spec)
                gt_abs = to_absolute(batch["eval_action_raw"].numpy().astype(np.float64),
                                     anchor_gt)
                motion = lambda c: c - c[:, :1, :]
                gt_motion = motion(gt_abs)
                anchors = {"oracle": anchor_gt,
                           "state": build_anchor(state, state, spec),
                           "state_offset": build_anchor(state, state + te_mean, spec)}

                for config in run_configs:
                    mode, total, split = config
                    generator.manual_seed(args.seed * 1_000_003 + step)
                    norm = ES.predict_draws(model, batch, device, mode, total,
                                            split, max(draws), generator=generator,
                                            max_flow_batch=args.max_flow_batch)
                    rel = denormalize(norm.float().cpu().numpy().astype(np.float64),
                                      mask, amin, amax)            # [B, K, T, D]
                    for a_name in args.anchors:
                        absolute = rel + anchors[a_name][:, None, None, :]
                        absolute = clamp_frozen_absolute(
                            absolute, mask, state[:, None, :])
                        for k in draws:
                            pred = absolute[:, :k].mean(axis=1)
                            name = f"{tag(config)}_k{k}_{a_name}{suffix}"
                            acc(name, horizon, dim).add(
                                pred - gt_abs, contact, motion(pred) - gt_motion)
                            smooths[name].add(pred)
                            if checker is not None:
                                for b in range(pred.shape[0]):
                                    accs[name].add_safety(checker.check(state[b], pred[b]))
                            if projector is not None:
                                safe = projector.project(state, pred)
                                sname = name + "_safe"
                                acc(sname, horizon, dim).add(
                                    safe - gt_abs, contact, motion(safe) - gt_motion)
                                smooths[sname].add(safe)
                                for b in range(safe.shape[0]):
                                    accs[sname].add_safety(checker.check(state[b], safe[b]))

                if p_i == 0:
                    const = lambda pose: np.repeat(np.asarray(pose)[:, None, :],
                                                   horizon, axis=1)
                    for name, chunk in (("hold_state", const(state)),
                                        ("repeat_command", const(gt_abs[:, 0])),
                                        ("oracle_prev_command", const(prev_cmd))):
                        acc(name, horizon, dim).add(chunk - gt_abs, contact,
                                                    motion(chunk) - gt_motion)
                    acc("teleop_gt", horizon, dim).add(np.zeros_like(gt_abs))
                    smooths["teleop_gt"].add(gt_abs)
                    if checker is not None:
                        for b in range(gt_abs.shape[0]):
                            accs["teleop_gt"].add_safety(checker.check(state[b], gt_abs[b]))
                if step % 10 == 0:
                    print(f"  {(step + 1) * args.batch_size}/{len(eval_set)} | "
                          f"{time.time() - started:.0f}s")
            payload.setdefault("prompts", []).append(dataset.effective_instruction)

        for name, a in accs.items():
            chunk_reports[name] = a.report()
            chunk_reports[name]["smoothness"] = smooths[name].report()
        payload["chunk_pass"] = chunk_reports
        payload["n_eval_samples"] = len(eval_set)
        with open(os.path.join(args.out_dir, "metrics_deploy.json"), "w") as handle:
            json.dump(payload, handle, indent=2)

    # ═══════════════════ pass 2: gateway replay ═══════════════════════════════
    rollout: Optional[dict] = None
    if args.rollout_root and args.rollout_episodes > 0:
        dataset = make_dataset(args.rollout_root, args.prompts[0])
        horizon, dim = dataset.action_chunk, dataset.action_dim
        mask, amin, amax, spec = (dataset.action_mask, dataset.action_min,
                                  dataset.action_max, dataset.anchor_spec)
        gap = dataset.sample_stride // max(1, dataset.chunk_stride)
        if gap < 1 or gap > horizon:
            raise SystemExit(f"rollout root stride {dataset.sample_stride} does not "
                             f"fit the {horizon}-step chunk")
        replay = GatewayReplay(kit.TemporalEnsembler, spec, mask, projector, gap, te_mean)
        starts, off = [], 0
        for n in dataset.ep_rows:
            starts.append(off)
            off += n
        n_eps = min(args.rollout_episodes, len(dataset.ep_rows))
        picked = sorted(set(np.linspace(0, len(dataset.ep_rows) - 1, n_eps)
                            .astype(int).tolist()))
        print(f"\nrollout pass: {len(picked)} episodes x {args.rollout_rows} rows, "
              f"gap {gap} frames, schedules {[tag(c) for c in schedules]}, "
              f"K={args.rollout_draws}, latencies {args.latencies}")

        aggs = {"gw": (args.agg_n, args.exp_k), "latest": (1, 0.0)}
        variants = list(itertools.product(
            [tag(c) for c in schedules], ["k%d" % args.rollout_draws, "k1"],
            ["oracle", "state", "state_offset", "self"], ["raw", "safe"], args.latencies, aggs))
        streams: Dict[tuple, ES.StreamStats] = defaultdict(lambda: ES.StreamStats(dim))
        uncovered: Dict[tuple, List[float]] = defaultdict(list)
        base_streams = {"hold_state": ES.StreamStats(dim), "teleop_gt": ES.StreamStats(dim)}
        traces: List[dict] = []

        for ep_i in picked:
            n_rows = min(args.rollout_rows, dataset.ep_rows[ep_i])
            ep_loader = DataLoader(
                Subset(dataset, list(range(starts[ep_i], starts[ep_i] + n_rows))),
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, collate_fn=dataset.collate_fn)
            rel_by_sched: Dict[str, List[np.ndarray]] = defaultdict(list)
            st, pc, gt = [], [], []
            for bstep, batch in enumerate(ep_loader):
                b_state = batch["eval_state"].numpy().astype(np.float64)
                b_prev = batch["eval_prev_command"].numpy().astype(np.float64)
                st.append(b_state)
                pc.append(b_prev)
                gt.append(to_absolute(batch["eval_action_raw"].numpy().astype(np.float64),
                                      build_anchor(b_state, b_prev, spec)))
                for config in schedules:
                    mode, total, split = config
                    generator.manual_seed(args.seed * 1_000_003 + ep_i * 10_007 + bstep * 101)
                    norm = ES.predict_draws(model, batch, device, mode, total, split,
                                            args.rollout_draws, generator=generator,
                                            max_flow_batch=args.max_flow_batch)
                    rel_by_sched[tag(config)].append(denormalize(
                        norm.float().cpu().numpy().astype(np.float64), mask, amin, amax))
            state_rows = np.concatenate(st)
            prev_rows = np.concatenate(pc)
            gt_abs = np.concatenate(gt)                                # [R, T, D]
            R = gt_abs.shape[0]
            gt_stream = np.concatenate([gt_abs[r, :gap] for r in range(R)])
            state_stream = np.repeat(state_rows, gap, axis=0)
            base_streams["hold_state"].add(state_stream, gt_stream, state_rows[0], checker)
            base_streams["teleop_gt"].add(gt_stream, gt_stream, state_rows[0], checker)

            for (sched, kname, anchor, safe, latency, agg) in variants:
                rel_all = np.concatenate(rel_by_sched[sched])        # [R, K, T, D]
                rel = rel_all.mean(axis=1) if kname != "k1" else rel_all[:, 0]
                executed, covered = replay.run(
                    rel, state_rows, prev_rows, latency, anchor,
                    aggs[agg][0], aggs[agg][1], safe == "safe")
                key = (sched, kname, anchor, safe, latency, agg)
                streams[key].add(executed, gt_stream, state_rows[0], checker)
                uncovered[key].append(1.0 - covered.mean())
                if (len(traces) < 3 and sched == tag(schedules[0])
                        and kname != "k1" and anchor == "state_offset" and safe == "safe"
                        and latency == 15 and agg == "gw"):
                    traces.append({"episode": os.path.basename(dataset.episodes[ep_i]["file"]),
                                   "gt": gt_stream, "executed": executed,
                                   "state": state_stream, "latency": latency})
            print(f"  ep {ep_i:3d} {os.path.basename(dataset.episodes[ep_i]['file'])}: "
                  f"{R} rows -> {gt_stream.shape[0]} frames")

        def vname(key) -> str:
            return "_".join(str(k) if not isinstance(k, int) else f"L{k}" for k in key)

        rollout = {
            "gap_frames": gap, "episodes": len(picked), "rows": args.rollout_rows,
            "draws": args.rollout_draws, "latencies": args.latencies,
            "aggregations": {"gw": {"agg_n": args.agg_n, "exp_k": args.exp_k},
                             "latest": {"agg_n": 1, "exp_k": 0.0}},
            "baselines": {n: s.report() for n, s in base_streams.items()},
            "variants": {},
        }
        for key, s in streams.items():
            rep = s.report()
            rep["uncovered_fraction"] = float(np.mean(uncovered[key]))
            rep["key"] = {"schedule": key[0], "draws": key[1], "anchor": key[2],
                          "safety": key[3], "latency_frames": key[4], "agg": key[5]}
            rollout["variants"][vname(key)] = rep
        payload["rollout"] = rollout
        with open(os.path.join(args.out_dir, "metrics_deploy.json"), "w") as handle:
            json.dump(payload, handle, indent=2)
        write_rollout_plots(args.out_dir, rollout, traces, schedules, tag, args)

    # ── console summary ───────────────────────────────────────────────────────
    def unsafe(report) -> str:
        rates = report.get("safety", {}).get("violation_rate_per_value")
        return "" if rates is None else f"{100 * sum(rates.values()):8.3f}%"

    if chunk_reports:
        print(f"\nchunk pass ({payload['n_eval_samples']} samples)")
        print(f"{'config':<30} {'MAE':>7} {'motion':>7} {'RMSE':>7} {'arms':>7} "
              f"{'hands':>7} {'jerk':>7} {'unsafe%':>9}")
        print("-" * 90)
        for name in sorted(chunk_reports, key=lambda n: (n.count("_") == 0, n)):
            r = chunk_reports[name]
            o = r.get("overall", {})
            g = o.get("per_group", {})
            arms = np.mean([g.get(x, {}).get("mae_deg", np.nan) for x in ("left_arm", "right_arm")])
            hands = np.mean([g.get(x, {}).get("mae_deg", np.nan) for x in ("left_hand", "right_hand")])
            print(f"{name:<30} {o.get('mae_deg', np.nan):7.3f} "
                  f"{r.get('motion', {}).get('mae_deg', np.nan):7.3f} "
                  f"{o.get('rmse_deg', np.nan):7.3f} {arms:7.3f} {hands:7.3f} "
                  f"{r['smoothness'].get('mean_abs_2nd_diff_deg', np.nan):7.3f} {unsafe(r)}")

    if rollout:
        print(f"\ngateway replay ({rollout['episodes']} episodes, gap {rollout['gap_frames']})")
        print(f"{'variant':<40} {'MAE':>7} {'arms':>7} {'hands':>7} {'jerk':>7} "
              f"{'unsafe%':>9} {'uncov':>6}")
        print("-" * 90)
        for name, r in rollout["baselines"].items():
            print(f"{name:<40} {r['mae_deg']:7.3f} {'':>7} {'':>7} "
                  f"{r['smoothness']['mean_abs_2nd_diff_deg']:7.3f} {unsafe(r)}")
        for name in sorted(rollout["variants"]):
            r = rollout["variants"][name]
            g = r["per_group_mae_deg"]
            arms = (g["left_arm"] + g["right_arm"]) / 2
            hands = (g["left_hand"] + g["right_hand"]) / 2
            print(f"{name:<40} {r['mae_deg']:7.3f} {arms:7.3f} {hands:7.3f} "
                  f"{r['smoothness']['mean_abs_2nd_diff_deg']:7.3f} {unsafe(r)} "
                  f"{r['uncovered_fraction']:6.3f}")
    print(f"\nwrote {args.out_dir}/metrics_deploy.json")
    return 0


def write_rollout_plots(out_dir, rollout, traces, schedules, tag, args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from trex_origami.seasons import JOINT_NAMES

    V = rollout["variants"]
    lat = rollout["latencies"]
    kmax = f"k{args.rollout_draws}"

    def get(sched, k, anchor, safe, L, agg, field="mae_deg"):
        key = f"{sched}_{k}_{anchor}_{safe}_L{L}_{agg}"
        return V.get(key, {}).get(field, np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    ax = axes[0]
    for i, c in enumerate(schedules):
        s = tag(c)
        ax.plot(lat, [get(s, kmax, "state_offset", "safe", L, "gw") for L in lat],
                marker="o", color=f"C{i}", label=f"{s} K={args.rollout_draws} state_offset safe")
        ax.plot(lat, [get(s, "k1", "state_offset", "safe", L, "gw") for L in lat],
                marker="x", ls="--", color=f"C{i}", label=f"{s} K=1 state_offset safe")
    ax.axhline(rollout["baselines"]["hold_state"]["mae_deg"], color="gray", ls=":",
               label="hold_state")
    ax.set_xlabel("request latency (frames @30 Hz)")
    ax.set_ylabel("executed-stream MAE (deg)")
    ax.set_title("Gateway async replay: error vs latency")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1]
    s = tag(schedules[0])
    for j, anchor in enumerate(("oracle", "state", "state_offset", "self")):
        ax.plot(lat, [get(s, kmax, anchor, "safe", L, "gw") for L in lat],
                marker="o", color=f"C{j}", label=f"anchor={anchor}")
    ax.plot(lat, [get(s, kmax, "state_offset", "raw", L, "gw") for L in lat],
            marker="s", ls="--", color="C2", alpha=0.6, label="state_offset, no safety")
    ax.plot(lat, [get(s, kmax, "state_offset", "safe", L, "latest") for L in lat],
            marker="^", ls=":", color="C2", alpha=0.8, label="state_offset, latest-chunk (sync-like)")
    ax.set_xlabel("request latency (frames @30 Hz)")
    ax.set_title(f"{s}: anchor source / safety / aggregation")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[2]
    for i, c in enumerate(schedules):
        s = tag(c)
        ax.plot(lat, [get(s, kmax, "state_offset", "safe", L, "gw",
                          field="uncovered_fraction") for L in lat],
                marker="o", color=f"C{i}", label=s)
    ax.set_xlabel("request latency (frames @30 Hz)")
    ax.set_ylabel("fraction of frames with no chunk to execute")
    ax.set_title("Coverage (hold_last engages when 0 < fraction)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "replay_vs_latency.png"), dpi=140)
    plt.close(fig)

    picks = [3, 7, 14, 36, 43]
    for i, tr in enumerate(traces):
        fig, axes = plt.subplots(len(picks), 1, figsize=(13, 2.1 * len(picks)), sharex=True)
        for ax, d in zip(axes, picks):
            ax.plot(tr["gt"][:, d], label="teleop command", lw=1.3)
            ax.plot(tr["state"][:, d], label="measured state", lw=0.8, alpha=0.6)
            ax.plot(tr["executed"][:, d], label=f"executed (L={tr['latency']}, gw agg)", lw=1.0)
            ax.set_ylabel(JOINT_NAMES[d], fontsize=7)
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=8, loc="upper right")
        axes[0].set_title(f"{tr['episode']} -- gateway replay", fontsize=9)
        axes[-1].set_xlabel("frame (30 Hz)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"replay_trace_{i:02d}.png"), dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
