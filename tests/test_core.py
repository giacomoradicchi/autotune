import numpy as np
from py_autotune.core import autotune_audio
import soundfile as sf
import os

def test_autotune_audio(tmp_path):
    sr = 16000
    t = np.linspace(0, 1, sr)
    x = 0.5 * np.sin(2 * np.pi * 220 * t)
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"
    sf.write(input_path, x, sr)
    autotune_audio(str(input_path), str(output_path), key="C major")
    assert os.path.exists(output_path)
    y, sr2 = sf.read(output_path)
    assert y.shape[0] > 0
    assert sr2 == sr
