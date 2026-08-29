# Confirmed working torch/CUDA versions for Blackwell (RTX 50-series, sm_100/sm_120)

Tested 2026-08-30 against a real RTX 5090 (sm_120) host. The Dockerfile's pinned
`torch==2.6.0+cu124`/`torchvision==0.21.0+cu124` only ship kernels/PTX up to sm_90
(Hopper) -- confirmed failing with `RuntimeError: CUDA error: no kernel image is
available for execution on the device` on the very first embedding lookup
(`modeling_vla.py:428`, `prepare_inputs_embeds`).

## Confirmed-working combo

```
--index-url https://download.pytorch.org/whl/cu128
torch==2.11.0+cu128
torchvision==0.26.0+cu128
```

Verified: `torch.cuda.get_arch_list()` includes `sm_100` and `sm_120`, and the
actual T-Rex model (construction + a real `infer()` forward pass through
`checkpoint-1-5500`) ran without any CUDA kernel error on the RTX 5090 --
progressed past the point that failed under 2.6.0+cu124, all the way into
`_run_slow_and_fast`'s state-normalization step (where it then hit an
unrelated, real bug in `trex_policy_server.py` -- see below, not a GPU issue).

## Not yet done / not verified

- Have NOT changed the actual `Dockerfile` pins -- this was tested in a
  standalone venv on the host, not the submission image itself.
- Have NOT re-checked `requirements-inference.lock`'s other pins
  (transformers==4.57.3 etc.) for compatibility with torch 2.11 beyond
  "pip installed without conflict" -- no deeper compatibility audit done.
- Have NOT re-run this against an older-generation GPU (3090/4090) to confirm
  no regression there.
- Unknown whether the real competition eval GPU is Blackwell-class at all --
  the org's own docs never disclose the eval hardware. This upgrade is a
  "supports newer hardware too" improvement, not a confirmed requirement.

## The separate bug found while testing this

`_model_load()` in `trex_policy_server.py` builds `statistic` from
`stats_data.json` with only `action_*` and `tacf6_*` keys:

```python
statistic = {
    "action_mask": ..., "action_min": ..., "action_max": ...,
    "tacf6_mask": ..., "tacf6_min": ..., "tacf6_max": ...,
}
```

But `_run_slow_and_fast` (line 343) calls
`_normalize(state, statistic["state_mask"], statistic["state_min"], statistic["state_max"])`
-- `state_mask`/`state_min`/`state_max` are never populated. This raises
`KeyError: 'state_mask'` on every real `infer()` call, on **any** GPU --
not specific to Blackwell or this test. We only surfaced it now because the
CUDA error on older torch was happening *earlier* in the pipeline, before
this code path was ever reached.
