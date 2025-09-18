import numpy as np
import scipy as sp
import soundfile as sf

# Settings
input_file = "my_audio.wav"
output_file = "my_audio_tuned.wav"
selected_key = "C minor"
A = 440 #hz


dict_scale = {
    "chromatic": ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"],

    # major scales
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

    # minor scales
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


def time_stretch(x, fs, stretch_factor, frame_length=2048, hop_length=512):
    # Short time Fourier Transform
    f, t, S = sp.signal.stft(x, fs=fs, nperseg=frame_length, noverlap=frame_length-hop_length, window=np.hanning(frame_length))

    # Getting magnitude and phase:
    magnitude = np.abs(S)
    phase = np.angle(S)

    # time step:
    # if stretch_factor > 1, the audio will be compressed
    # if stretch_factor < 1, the audio will be stretched
    time_steps = np.arange(0, len(t), stretch_factor)

    # Initializing new stft
    new_S = np.zeros(shape=(np.shape(S)[0], len(time_steps)), dtype=np.complex64)

    # the accumulator stands for integral
    phase_accumulator = phase[:, 0]

    for i, step in enumerate(time_steps):
        index = int(np.floor(step)) # round to the nearest integer value that is below step
        fraction = step - index

        # interpolate between magnitude:
        mag1 = magnitude[:, index]
        mag2 = magnitude[:, min(index + 1, np.shape(magnitude)[1] - 1)]
        interpolated_magnitude = (1 - fraction) * mag1 + fraction * mag2

        # calculate instant frequency:
        phase1 = phase[:, index]
        phase2 = phase[:, min(index + 1, np.shape(magnitude)[1] - 1)]
        instant_freq = phase2 - phase1

        # wrapping frequency between [-pi, pi]
        instant_freq = np.mod(instant_freq + np.pi, 2 * np.pi) - np.pi

        # updating accumulator with new instant freq
        phase_accumulator += instant_freq #stretch_factor

        # populating new stft
        new_S[:, i] = interpolated_magnitude * np.exp(1j * phase_accumulator)

    _, x_stretched = sp.signal.istft(new_S, fs=fs, nperseg=frame_length, noverlap=frame_length-hop_length)
    return x_stretched

def pitch_shifting(x, fs, pitch_factor, frame_length=2048, hop_length=512):
    x_stretched = time_stretch(x, fs, 1/pitch_factor, frame_length=2048, hop_length=512)
    stretched_length = len(x_stretched)

    t_old = np.arange(0, stretched_length)
    t_new = np.arange(0, stretched_length, pitch_factor)
    x_pitched = np.interp(t_new, t_old, x_stretched)

    t_old = np.arange(0, len(x_pitched))
    t_new = np.arange(0, len(x))
    x_pitched = np.interp(t_new, t_old, x_pitched)

    return x_pitched

def find_best_lag(current_sample, frame, start_index, n_overlap):
    max_lag = n_overlap // 8

    # Estrai la regione di interesse dal segnale corrente
    search_start = max(0, start_index - max_lag)
    search_end = min(len(current_sample), start_index + len(frame) + max_lag)
    search_region = current_sample[search_start:search_end]

    if len(search_region) < len(frame):
        return 0

    # Usa correlazione crociata
    correlation = np.correlate(search_region, frame, mode='valid')
    best_offset = np.argmax(correlation)

    # Converti in lag relativo alla posizione originale
    best_lag = search_start + best_offset - start_index

    return best_lag

def find_pitch(sample, sr):
    voice_range = [80, 200] # Hz
    min_lag = sr // voice_range[1]
    max_lag = sr // voice_range[0]

    diff_areas = [get_difference_area(sample, sample[tau:]) for tau in range(min_lag, max_lag)]
    mean = np.sum

    # first relative_min found will be the delta lag
    delta_lag = 0
    for tau in range(1, len(diff_areas) - 1):
        if diff_areas[tau] < diff_areas[tau-1] and diff_areas[tau] < diff_areas[tau+1]:
            delta_lag = tau

    delta_lag = np.argmin(diff_areas)


    return sr / (delta_lag + min_lag)


def get_difference_area(x1, x2):
    length = min(len(x1), len(x2))
    return np.sum((x1[:length] - x2[:length]) ** 2) / length

def get_note_freq(note, octave):
    index = np.where(np.array(dict_scale["chromatic"]) == note)[0][0]
    return A * 2**((index - 9)/12) * 2**(octave - 4)


def find_closest_note(pitch, selected_scale):
    # finding the right octave:
    c0 = get_note_freq("C", 0)
    octave = np.floor(np.log2(pitch / c0))

    # finding the closest note:
    note_frequencies = np.array([get_note_freq(note, octave) for note in selected_scale])
    index_closest_note = np.argmin(np.abs(note_frequencies - pitch))

    return get_note_freq(selected_scale[index_closest_note], octave)


def tuning(data, sr, scale="chromatic"):
    sample_length = len(data)

    frame_length = int(2048)
    hop_length = frame_length // 4

    x_out = np.zeros_like(data)

    # attributes for MA filter (moving average)
    pitch_factors = []


    # Overlap-add
    window = np.hanning(frame_length)
    for i in range(0, sample_length, hop_length):
        if i + frame_length >= sample_length:
            frame = np.zeros(shape=(frame_length,))
            frame[: sample_length - i] = data[i : sample_length]
        else:
            frame = data[i : i + frame_length]

        # multiplying frame by window function
        frame = frame * window

        # finding frame pitch
        pitch = find_pitch(frame, sr)

        goal_pitch = find_closest_note(pitch, dict_scale[scale])
        pitch_factors.append(goal_pitch/pitch)
        smooth_pitch_factor = ma_filter(pitch_factors)
        #final smoothing:
        fract = abs(smooth_pitch_factor - goal_pitch) / (abs(smooth_pitch_factor) + abs(goal_pitch))
        smooth_pitch_factor = smooth_pitch_factor*fract + 1*(1-fract)

        # shifting to the aim pitch
        frame = pitch_shifting(frame, sr, smooth_pitch_factor)

        tau = 0
        if i != 0:
            tau = find_best_lag(x_out, frame, i, frame_length - hop_length)


        # Add
        if i + tau + frame_length >= sample_length:
            x_out[i + tau : sample_length] = x_out[i + tau : sample_length] + frame[: sample_length - i - tau]
        else:
            x_out[i + tau : i + tau + frame_length] = x_out[i + tau: i + tau + frame_length] + frame

    # Normalize output
    max_value = np.max(x_out)
    clip = 1
    if max_value > clip:
        x_out /= (max_value/clip)

    return x_out

def ma_filter(pf_list, ma_length = 5):
    list_length = len(pf_list)

    # custom window
    window = np.arange(ma_length, dtype=float)
    window = window * window * window
    # normalizing area
    window *= 1/np.sum(window)

    # moving average filter:
    # it's not necessary to divide to np.sum(window) since
    # window function was normalized
    if list_length < ma_length:
        window = window[ma_length - list_length:]
        return np.sum(pf_list[:list_length] * window)

    return np.sum(pf_list[list_length - ma_length:list_length] * window)


print("Reading audio...")
x, fs = sf.read(input_file)

# Currently working with audio mono
if x.ndim > 1 and np.shape(x)[1] > 1:
   x = x[:, 0]

print("Starting tuning...")
x_tuned = tuning(x, fs, selected_key)

print("Saving...")
sf.write(output_file, x_tuned, fs)

print("Completed.")




