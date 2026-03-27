# syntax=docker/dockerfile:1.4

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /opt/voicevox_engine

RUN <<EOF
    set -eux
    apt-get update
    apt-get install -y git curl cmake libsndfile1 ca-certificates build-essential unzip
    apt-get clean
    rm -rf /var/lib/apt/lists/*
EOF

# Install uv (system-wide)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install Python 3.8 and dependencies (including coeiroink_core)
COPY requirements.txt ./
RUN uv venv --python 3.8 .venv
RUN uv pip install --python .venv/bin/python setuptools "numpy==1.20.0" "cython==0.29.24"
RUN uv pip install --python .venv/bin/python --no-build-isolation -r requirements.txt git+https://github.com/0kqnet/coeiroink_core.git
RUN uv pip install --python .venv/bin/python "torch==1.13.1+cu116" "torchvision==0.14.1+cu116" "torchaudio==0.13.1" --extra-index-url https://download.pytorch.org/whl/cu116

# Copy app files
COPY voicevox_engine/ ./voicevox_engine/
COPY run.py generate_licenses.py presets.yaml default.csv default_setting.yml engine_manifest.json ./
COPY ui_template/ ./ui_template/
COPY engine_manifest_assets/ ./engine_manifest_assets/
COPY docs/ ./docs/

# Download pyopenjtalk dictionary at build time
RUN .venv/bin/python -c "import pyopenjtalk; pyopenjtalk._lazy_init()"

COPY entrypoint.sh ./
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
CMD ["uv", "run", "--python", ".venv/bin/python", "run.py", "--use_gpu", "--host", "0.0.0.0"]