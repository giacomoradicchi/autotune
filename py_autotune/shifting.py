"""
Modulo shifting: pitch shifting e time stretch
"""
import numpy as np
import scipy as sp


def time_stretch(x, fs, stretch_factor, frame_length=2048, hop_length=512):
    _, t, S = sp.signal.stft(
        x, fs=fs, nperseg=frame_length, noverlap=frame_length - hop_length, window=np.hanning(frame_length)
    )
    magnitude = np.abs(S)
    phase = np.angle(S)
    time_steps = np.arange(0, len(t), stretch_factor)
    new_S = np.zeros(
        shape=(np.shape(S)[0], len(time_steps)), dtype=np.complex64)
    phase_accumulator = phase[:, 0]
    for i, step in enumerate(time_steps):
        index = int(np.floor(step))
        fraction = step - index
        mag1 = magnitude[:, index]
        mag2 = magnitude[:, min(index + 1, np.shape(magnitude)[1] - 1)]
        interpolated_magnitude = (1 - fraction) * mag1 + fraction * mag2
        phase1 = phase[:, index]
        phase2 = phase[:, min(index + 1, np.shape(magnitude)[1] - 1)]
        instant_freq = phase2 - phase1
        instant_freq = np.mod(instant_freq + np.pi, 2 * np.pi) - np.pi
        phase_accumulator += instant_freq
        new_S[:, i] = interpolated_magnitude * np.exp(1j * phase_accumulator)
    _, x_stretched = sp.signal.istft(
        new_S, fs=fs, nperseg=frame_length, noverlap=frame_length - hop_length
    )
    return x_stretched


def pitch_shifting(x, fs, pitch_factor, frame_length=2048, hop_length=512):
    x_stretched = time_stretch(
        x, fs, 1 / pitch_factor, frame_length=frame_length, hop_length=hop_length
    )
    stretched_length = len(x_stretched)
    t_old = np.arange(0, stretched_length)
    t_new = np.arange(0, stretched_length, pitch_factor)
    x_pitched = np.interp(t_new, t_old, x_stretched)
    t_old = np.arange(0, len(x_pitched))
    t_new = np.arange(0, len(x))
    x_pitched = np.interp(t_new, t_old, x_pitched)
    return x_pitched
