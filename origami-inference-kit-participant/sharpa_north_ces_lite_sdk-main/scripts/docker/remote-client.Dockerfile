ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

ARG INSTALL_DEPENDENCIES=1
ARG PIP_INSTALL_ARGS=
ENV PATH="/.venv/bin:${PATH}"
RUN if [ "${INSTALL_DEPENDENCIES}" = "1" ]; then \
      python -m pip install --no-cache-dir ${PIP_INSTALL_ARGS} \
        "eclipse-zenoh>=1.9.0" \
        "msgpack>=1.0.5" \
        "numpy>=1.22.4,<3"; \
    fi

WORKDIR /kit
COPY examples/remote_observation_client.py examples/remote_observation_client.py

USER 65532:65532
ENTRYPOINT ["python", "examples/remote_observation_client.py"]
