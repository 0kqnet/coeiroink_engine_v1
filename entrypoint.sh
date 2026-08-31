#!/bin/bash
set -e

SPEAKER_INFO_DIR="/opt/voicevox_engine/speaker_info"
DOWNLOAD_URL="https://www.dropbox.com/s/lvxit2svbi1dvdf/3c37646f-3881-5374-2a83-149267990abc.zip?dl=1"

if [ -z "$(ls -A "$SPEAKER_INFO_DIR" 2>/dev/null)" ]; then
    echo "speaker_info is empty, downloading..."
    mkdir -p "$SPEAKER_INFO_DIR"
    TMP_ZIP=$(mktemp /tmp/speaker_info_XXXXXX.zip)
    curl -fSL "$DOWNLOAD_URL" -o "$TMP_ZIP"
    unzip -o "$TMP_ZIP" -d "$SPEAKER_INFO_DIR"
    rm "$TMP_ZIP"
    echo "speaker_info download complete."
fi

if [[ " $* " == *" --use_gpu "* ]]; then
    .venv/bin/python verify_cuda.py
fi

exec "$@"
