import base64
import io
from typing import List, Tuple

import numpy as np
import wave as wave_module

import struct

from scipy.signal import resample


def _read_wav(file):
    """soundfile.readの代替。(ndarray, samplerate)を返す"""
    with wave_module.open(file, "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        samplerate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    if sampwidth == 2:
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
    elif sampwidth == 4:
        # 32-bit float WAV
        data = np.frombuffer(frames, dtype=np.float32)
    elif sampwidth == 1:
        data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128) / 128.0
    else:
        raise ConnectBase64WavesException("wavファイルを読み込めませんでした")

    if channels > 1:
        data = data.reshape(-1, channels)

    return data, samplerate


class ConnectBase64WavesException(Exception):
    def __init__(self, message: str):
        self.message = message


def decode_base64_waves(waves: List[str]) -> List[Tuple[np.ndarray, int]]:
    """
    base64エンコードされた複数のwavデータをデコードする
    Parameters
    ----------
    waves: list[str]
        base64エンコードされたwavデータのリスト
    Returns
    -------
    waves_nparray_sr: List[Tuple[np.ndarray, int]]
        (NumPy配列の音声波形データ, サンプリングレート) 形式のタプルのリスト
    """
    if len(waves) == 0:
        raise ConnectBase64WavesException("wavファイルが含まれていません")

    waves_nparray_sr = []
    for wave in waves:
        try:
            wav_bin = base64.standard_b64decode(wave)
        except ValueError:
            raise ConnectBase64WavesException("base64デコードに失敗しました")
        try:
            _data = _read_wav(io.BytesIO(wav_bin))
        except ConnectBase64WavesException:
            raise
        except Exception:
            raise ConnectBase64WavesException("wavファイルを読み込めませんでした")
        waves_nparray_sr.append(_data)

    return waves_nparray_sr


def connect_base64_waves(waves: List[str]) -> Tuple[np.ndarray, int]:
    waves_nparray_sr = decode_base64_waves(waves)

    max_sampling_rate = max([sr for _, sr in waves_nparray_sr])
    max_channels = max([x.ndim for x, _ in waves_nparray_sr])
    assert 0 < max_channels <= 2

    waves_nparray_list = []
    for nparray, sr in waves_nparray_sr:
        if sr != max_sampling_rate:
            nparray = resample(nparray, max_sampling_rate * len(nparray) // sr)
        if nparray.ndim < max_channels:
            nparray = np.array([nparray, nparray]).T
        waves_nparray_list.append(nparray)

    return np.concatenate(waves_nparray_list), max_sampling_rate
