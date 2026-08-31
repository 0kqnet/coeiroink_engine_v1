# syntax=docker/dockerfile:1.4

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive

ARG PYTHON_VERSION=3.9
ARG PYTORCH_VERSION=2.7.1
ARG TORCHVISION_VERSION=0.22.1
ARG TORCHAUDIO_VERSION=2.7.1

WORKDIR /opt/voicevox_engine

RUN <<EOF
    set -eux
    apt-get update
    apt-get install -y git curl cmake libsndfile1 ca-certificates build-essential unzip
    apt-get clean
    rm -rf /var/lib/apt/lists/*
EOF

# Install a pinned uv release without depending on GHCR authentication.
ADD https://astral.sh/uv/0.11.1/install.sh /tmp/uv-installer.sh
RUN env UV_UNMANAGED_INSTALL=/usr/local/bin sh /tmp/uv-installer.sh \
    && rm /tmp/uv-installer.sh

# PyTorch 2.7 + CUDA 12.8 is the first stable combination with native
# Blackwell (RTX 50 series / sm_120) support.  Keep Python 3.9 because the
# COEIROINK v1 / ESPnet dependency set was built for the Python 3.9 era.
COPY constraints-gpu.txt ./
RUN uv venv --python "${PYTHON_VERSION}" .venv
RUN uv pip install --python .venv/bin/python \
        "torch==${PYTORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
        "torchaudio==${TORCHAUDIO_VERSION}" \
        --index-url https://download.pytorch.org/whl/cu128
COPY requirements.txt ./
RUN uv pip install --python .venv/bin/python setuptools "numpy==1.24.4" "cython==0.29.24"
RUN uv pip install --python .venv/bin/python \
        --no-build-isolation \
        --constraint constraints-gpu.txt \
        -r requirements.txt \
        git+https://github.com/0kqnet/coeiroink_core.git
RUN uv pip check --python .venv/bin/python \
    && .venv/bin/python -c "import torch; flags = torch._C._cuda_getArchFlags().split(); assert torch.__version__ == '${PYTORCH_VERSION}+cu128', torch.__version__; assert 'sm_120' in flags, flags; from espnet2.bin.tts_inference import Text2Speech"

# Copy app files
COPY voicevox_engine/ ./voicevox_engine/
COPY run.py verify_cuda.py generate_licenses.py presets.yaml default.csv default_setting.yml engine_manifest.json ./
COPY ui_template/ ./ui_template/
COPY engine_manifest_assets/ ./engine_manifest_assets/
COPY docs/ ./docs/

# Download pyopenjtalk dictionary at build time
RUN .venv/bin/python -c "import pyopenjtalk; pyopenjtalk._lazy_init()"

COPY entrypoint.sh ./
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
CMD ["uv", "run", "--python", ".venv/bin/python", "run.py", "--use_gpu", "--host", "0.0.0.0"]
