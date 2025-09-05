"""
Modulo detection: pitch detection
"""
import numpy as np

def find_pitch(sample, sr):
    voice_range = [80, 200] # Hz
    min_lag = sr // voice_range[1] 
    max_lag = sr // voice_range[0] 
    diff_areas = [get_difference_area(sample, sample[tau:]) for tau in range(min_lag, max_lag)] 
    delta_lag = np.argmin(diff_areas) 
    return sr / (delta_lag + min_lag) 

def get_difference_area(x1, x2):
    length = min(len(x1), len(x2)) 
    return np.sum((x1[:length] - x2[:length]) ** 2) / length 

def find_closest_note(pitch, selected_scale):
    from .utils import get_note_freq
    c0 = get_note_freq("C", 0)
    octave = np.floor(np.log2(pitch / c0))
    note_frequencies = np.array([get_note_freq(note, octave) for note in selected_scale])
    index_closest_note = np.argmin(np.abs(note_frequencies - pitch))
    return get_note_freq(selected_scale[index_closest_note], octave)
