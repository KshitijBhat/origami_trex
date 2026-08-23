# Participant Docker build files

Run all build commands from `sharpa_north_ces_lite_sdk-main/`.

## Framework-neutral policy template

```bash
docker build \
  -f scripts/docker/policy-template.Dockerfile \
  -t origami-policy-template:dev \
  .
```

The template returns a hold-position action only to verify protocol wiring.
Replace `TeamPolicy` and build the team's real self-contained image before
submission.

## Public Zenoh validator

```bash
docker build \
  -f scripts/docker/validator.Dockerfile \
  -t origami-policy-validator:dev \
  .

docker run --rm --network host \
  origami-policy-validator:dev \
  --endpoint tcp/127.0.0.1:17447 \
  --session-id local-contract-test \
  --expected-horizon 25
```

## Public read-only observation client

```bash
docker build \
  -f scripts/docker/remote-client.Dockerfile \
  -t origami-remote-observation-client:dev \
  .
```

Run it with the endpoint, session, token, and TLS CA issued by the organizer.
Mount certificates read-only and pass credentials at runtime; never bake them
into the image.

## OpenPI submission images

OpenPI-specific Dockerfiles and inference code are in:

```text
../openpi-base-main/scripts/docker/
../openpi-base-main/scripts/serve_policy_zenoh.py
```

Teams must include their own checkpoint, normalization assets, tokenizer, and
all runtime dependencies in the final submission image.

Example from the repository root:

```bash
docker build \
  --build-arg RUNTIME_IMAGE=<compatible-openpi-runtime-image> \
  --build-context checkpoint=/absolute/path/to/checkpoint \
  --build-context python_packages=/absolute/path/to/python/site-packages \
  -f openpi-base-main/scripts/docker/submission-zenoh-bundled.Dockerfile \
  -t team-name/origami-openpi:submission \
  openpi-base-main
```

The `python_packages` context must contain `zenoh/` and
`eclipse_zenoh-1.9.0.dist-info/`. The checkpoint context must contain `params/`
or `model.safetensors` plus `assets/**/norm_stats.json`.
