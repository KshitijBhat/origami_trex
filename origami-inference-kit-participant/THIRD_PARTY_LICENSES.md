# Third-Party Licenses

This public competition package uses or distributes the following third-party components. Each component is governed by its official license.

## Source code bundled under `openpi-base-main/`

- Physical Intelligence OpenPI and `openpi-client` — Apache License 2.0
  https://github.com/Physical-Intelligence/openpi
  Full license text: `openpi-base-main/LICENSE`
- Big Vision model components — Apache License 2.0
- Hugging Face Transformers-derived model components — Apache License 2.0
- Gemma model usage is additionally subject to the Google Gemma Terms of Use.
  Full terms: `openpi-base-main/LICENSE_GEMMA.txt`

Teams that redistribute an OpenPI-based image must preserve the applicable
license and Gemma notices and must separately verify the license for their model
weights.

## Python runtime dependencies

- Eclipse Zenoh / `eclipse-zenoh` — Eclipse Public License 2.0 /
  Apache License 2.0
  https://github.com/eclipse-zenoh/zenoh-python
- MessagePack / `msgpack` — Apache License 2.0
  https://github.com/msgpack/msgpack-python
- NumPy — BSD-3-Clause
  https://github.com/numpy/numpy
- OpenCV / `opencv-python` — Apache License 2.0
  https://github.com/opencv/opencv-python

These dependencies are installed through the Python package manager. Their source code is not distributed with this repository.

## Frontend files bundled with the local Shadow evaluator

- Three.js — MIT License
  https://github.com/mrdoob/three.js
  Full license text:
  `sharpa_north_ces_lite_sdk-main/participant_local_evaluator/static/vendor/LICENSE.three.txt`
- `urdf-loaders` / URDFLoader — Apache License 2.0
  https://github.com/gkjohnson/urdf-loaders
  Full license text:
  `sharpa_north_ces_lite_sdk-main/participant_local_evaluator/static/vendor/LICENSE.urdf-loader.txt`

## Team responsibilities

Model code, frameworks, checkpoints, tokenizers, weights, and other assets submitted by competition
teams are not part of this public package. Each team must independently verify that it has the
necessary rights to use and redistribute those materials and must include all required LICENSE files,
NOTICE files, and model terms in its final image.
