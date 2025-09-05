"""
Modulo utils: scale, note, lag, filtro MA
"""
import numpy as np
A = 440 #hz

dict_scale = {
    "chromatic": ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"],
    "C major": ["C", "D", "E", "F", "G", "A", "B"],
    "C# major": ["C#", "D#", "F", "F#", "G#", "A#", "C"],
    "D major": ["D", "E", "F#", "G", "A", "B", "C#"],
    "D# major": ["D#", "F", "G", "G#", "A#", "C", "D"],
    "E major": ["E", "F#", "G#", "A", "B", "C#", "D#"],
    "F major": ["F", "G", "A", "A#", "C", "D", "E"],
    "F# major": ["F#", "G#", "A#", "B", "C#", "D#", "F"],
    "G major": ["G", "A", "B", "C", "D", "E", "F#"],
    "G# major": ["G#", "A#", "C", "C#", "D#", "F", "G"],
    "A major": ["A", "B", "C#", "D", "E", "F#", "G#"],
    "A# major": ["A#", "C", "D", "D#", "F", "G", "A"],
    "B major": ["B", "C#", "D#", "E", "F#", "G#", "A#"],
    "C minor": ["C", "D", "D#", "F", "G", "G#", "A#"],
    "C# minor": ["C#", "D#", "E", "F#", "G#", "A", "B"],
    "D minor": ["D", "E", "F", "G", "A", "A#", "C"],
    "D# minor": ["D#", "F", "F#", "G#", "A#", "B", "C#"],
    "E minor": ["E", "F#", "G", "A", "B", "C", "D"],
    "F minor": ["F", "G", "G#", "A#", "C", "C#", "D#"],
    "F# minor": ["F#", "G#", "A", "B", "C#", "D", "E"],
    "G minor": ["G", "A", "A#", "C", "D", "D#", "F"],
    "G# minor": ["G#", "A#", "B", "C#", "D#", "E", "F#"],
    "A minor": ["A", "B", "C", "D", "E", "F", "G"],
    "A# minor": ["A#", "C", "C#", "D#", "F", "F#", "G#"],
    "B minor": ["B", "C#", "D", "E", "F#", "G", "A"],
}

def get_note_freq(note, octave):
    index = np.where(np.array(dict_scale["chromatic"]) == note)[0][0]
    return A * 2**((index - 9)/12) * 2**(octave - 4)

def find_best_lag(current_sample, frame, start_index, n_overlap):
    max_lag = n_overlap // 8
    search_start = max(0, start_index - max_lag)
    search_end = min(len(current_sample), start_index + len(frame) + max_lag)
    search_region = current_sample[search_start:search_end]
    if len(search_region) < len(frame):
        return 0
    correlation = np.correlate(search_region, frame, mode='valid')
    best_offset = np.argmax(correlation)
    best_lag = search_start + best_offset - start_index
    return best_lag

def ma_filter(pf_list, ma_length = 5):
    list_length = len(pf_list)
    window = np.arange(ma_length, dtype=float)
    window = window * window * window
    window *= 1/np.sum(window)
    if list_length < ma_length:
        window = window[ma_length - list_length:]
        return np.sum(pf_list[:list_length] * window)
    return np.sum(pf_list[list_length - ma_length:list_length] * window)
