"""
Modulo core: pipeline autotune_audio
"""
import soundfile as sf
from .detection import find_pitch, find_closest_note
from .shifting import pitch_shifting
from .utils import dict_scale, ma_filter
import numpy as np


def autotune_audio(input_path, output_path, key="C major"):
    """
    Applica autotune a un file audio.
    Args:
        input_path (str): percorso file input
        output_path (str): percorso file output
        key (str): scala musicale
    """
    x, fs = sf.read(input_path)
    if x.ndim > 1:
        x = x[:, 0]
    x_tuned = tuning(x, fs, key)
    sf.write(output_path, x_tuned, fs)


def tuning(data, sr, scale="chromatic"):
    sample_length = len(data)
    frame_length = 2048
    hop_length = frame_length // 4
    x_out = np.zeros_like(data)
    pitch_factors = []
    window = np.hanning(frame_length)
    for i in range(0, sample_length, hop_length):
        if i + frame_length >= sample_length:
            frame = np.zeros(shape=(frame_length,))
            frame[:sample_length - i] = data[i:sample_length]
        else:
            frame = data[i:i + frame_length]
        frame = frame * window
        pitch = find_pitch(frame, sr)
        goal_pitch = find_closest_note(pitch, dict_scale[scale])
        pitch_factors.append(goal_pitch / pitch)
        smooth_pitch_factor = ma_filter(pitch_factors)
        fract = abs(smooth_pitch_factor - goal_pitch) / (abs(smooth_pitch_factor) + abs(goal_pitch))
        smooth_pitch_factor = smooth_pitch_factor * fract + 1 * (1 - fract)
        frame = pitch_shifting(frame, sr, smooth_pitch_factor)
        tau = 0
        if i != 0:
            from .utils import find_best_lag
            tau = find_best_lag(x_out, frame, i, frame_length - hop_length)
        if i + tau + frame_length >= sample_length:
            x_out[i + tau:sample_length] = x_out[i + tau:sample_length] + frame[:sample_length - i - tau]
        else:
            x_out[i + tau:i + tau + frame_length] = x_out[i + tau:i + tau + frame_length] + frame
    max_value = np.max(x_out)
    clip = 1
    if max_value > clip:
        x_out /= (max_value / clip)
    return x_out
