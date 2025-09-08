function autotune(file_name) 
    % Parameters
    A = 440; %Hz
    scale = "D# minor";
    
    [data, sr] = audioread(file_name);

    % lavoriamo con audio mono per semplicità
    if size(data, 2) > 1
        data = data(:, 1);
    end
    
    note_dict = dictionary( ...
        "chromatic",...
        {["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]} ...
    );

    % SCALE MAGGIORI
    note_dict("C major")  = {["C", "D", "E", "F", "G", "A", "B"]};
    note_dict("C# major") = {["C#", "D#", "F", "F#", "G#", "A#", "C"]};
    note_dict("D major")  = {["D", "E", "F#", "G", "A", "B", "C#"]};
    note_dict("D# major") = {["D#", "F", "G", "G#", "A#", "C", "D"]};
    note_dict("E major")  = {["E", "F#", "G#", "A", "B", "C#", "D#"]};
    note_dict("F major")  = {["F", "G", "A", "A#", "C", "D", "E"]};
    note_dict("F# major") = {["F#", "G#", "A#", "B", "C#", "D#", "F"]};
    note_dict("G major")  = {["G", "A", "B", "C", "D", "E", "F#"]};
    note_dict("G# major") = {["G#", "A#", "C", "C#", "D#", "F", "G"]};
    note_dict("A major")  = {["A", "B", "C#", "D", "E", "F#", "G#"]};
    note_dict("A# major") = {["A#", "C", "D", "D#", "F", "G", "A"]};
    note_dict("B major")  = {["B", "C#", "D#", "E", "F#", "G#", "A#"]};
    
    % SCALE MINORI (naturali)
    note_dict("C minor")  = {["C", "D", "D#", "F", "G", "G#", "A#"]};
    note_dict("C# minor") = {["C#", "D#", "E", "F#", "G#", "A", "B"]};
    note_dict("D minor")  = {["D", "E", "F", "G", "A", "A#", "C"]};
    note_dict("D# minor") = {["D#", "F", "F#", "G#", "A#", "B", "C#"]};
    note_dict("E minor")  = {["E", "F#", "G", "A", "B", "C", "D"]};
    note_dict("F minor")  = {["F", "G", "G#", "A#", "C", "C#", "D#"]};
    note_dict("F# minor") = {["F#", "G#", "A", "B", "C#", "D", "E"]};
    note_dict("G minor")  = {["G", "A", "A#", "C", "D", "D#", "F"]};
    note_dict("G# minor") = {["G#", "A#", "B", "C#", "D#", "E", "F#"]};
    note_dict("A minor")  = {["A", "B", "C", "D", "E", "F", "G"]};
    note_dict("A# minor") = {["A#", "C", "C#", "D#", "F", "F#", "G#"]};
    note_dict("B minor")  = {["B", "C#", "D", "E", "F#", "G", "A"]};
    
    
    x_out = tuning(data, sr, scale, note_dict, A);

    % salvataggio audio:
    audiowrite("tuned_travis.wav", x_out, sr);
end

function x_tuned = tuning(data, sr, scale, note_dict, A) 
    sample_length = numel(data);
    frame_length = 2048;
    hop_length = frame_length / 4;
    
    x_tuned = zeros(size(data));

    % overlap add
    window = hamming(frame_length);

    for ii = (1: hop_length: sample_length) 
        if ii + frame_length >= sample_length
            frame = zeros(frame_length, 1);
            frame(1:sample_length - (ii-1)) = data(ii:sample_length);
        else 
            frame = data(ii : ii + frame_length-1);
        end
        
        
        % multiplying frame by window function
        frame = frame.* window;
        

        % finding frame pitch
        pitch = find_pitch(frame, sr);

        goal_pitch = find_closest_note(pitch, note_dict(scale), note_dict, A);
        
        %shift_factor = goal_pitch/pitch;
        shift_factor = 0.9;
        frame = pitch_shifting(frame, sr, shift_factor);
        tau = 0;
        if ii ~= 1 
            %tau = find_best_lag_2(x_tuned, frame, ii, frame_length - hop_length);
        end

        % add

        if ii + tau + frame_length >= sample_length 
            x_tuned(ii + tau : end) = x_tuned(ii + tau : end) + frame(1: sample_length - ii - tau + 1);
        else 
            temp = x_tuned(ii + tau : ii + tau + frame_length-1) + frame;
            x_tuned(ii + tau : ii + tau + frame_length-1) = temp;
        end
    end

    % Normalize output
    max_value = max(x_tuned);
    clip = 1;
    if max_value > clip
        x_tuned = x_tuned / (max_value/clip);
    end
end

function f = get_note_freq(note, octave, note_dict, A) 
    notes = note_dict("chromatic");
    notes = notes{1};
    index = find(notes == note);
    f = A * 2^((index - 10)/12) * 2^(octave - 4);
end

function f = find_closest_note(pitch, selected_scale, note_dict, A) 
    % calculating octave
    C0 = get_note_freq("C", 0, note_dict, A);
    octave = floor(log2(pitch/C0));
    scale = selected_scale{1};
    
    % finding index note
    note_frequencies = zeros(1, numel(scale));
    for ii = 1:numel(scale)
        note_frequencies(ii) = get_note_freq(scale(ii), octave, note_dict, A);
    end
    
    [~, index_closest_note] = min(abs(note_frequencies - pitch));
    f = get_note_freq(scale(index_closest_note), octave, note_dict, A);
end

function x_stretched = time_stretch(x, sr, stretch_factor)
    % Short-Time Fourier Transform
    
    hop_length = 512;
    nfft = 2048;
    window = hann(nfft, "periodic");

    [S, ~, T] = stft(x, sr, 'Window', window, ...
        'OverlapLength', nfft - hop_length, ...
        'FFTLength', nfft);

    % Calcolo del modulo e della fase
    magnitude = abs(S);
    phase = angle(S);

    % Generazione time steps
    % se stretch_factor > 1, l'audio sarà più breve
    % se stretch_factor < 1, l'audio sarà più lungo
    time_steps = 1 : stretch_factor : size(T, 1);

    % Inizializzazione nuova STFT
    new_S = zeros(size(S,1), length(time_steps));

    % l'accumulatore svolge il ruolo dell'integrale
    phase_accumulator = phase(:, 1);
    og_stft_length = size(S, 2);
    i = 1;
    for step = time_steps
        index = floor(step);
        fraction = step - index;

        % interpolazione tra i moduli
        mag1 = magnitude(:, uint64(index));
        mag2 = magnitude(:, min(uint64(index) + 1, og_stft_length));
        interpolated_magnitude = (1 - fraction) * mag1 + fraction * mag2;

        % calcolo della frequenza istantanea (derivata)
        phase1 = phase(:, index);
        phase2 = phase(:, min(index + 1, og_stft_length));
        instant_freq = phase2 - phase1;

        % wrapping della frequenza istantanea
        instant_freq = mod(instant_freq + pi, 2 * pi) - pi;

        % aggiornamento dell'accumulatore (integrale)
        phase_accumulator = phase_accumulator + instant_freq;
        
        new_S(:, i) = interpolated_magnitude .* exp(1j * phase_accumulator);

        i = i + 1;
    end
    x_stretched = real(istft(new_S, sr, 'Window', window, ...
        'OverlapLength', nfft - hop_length, 'FFTLength', nfft));
    

    % dissolvenza di volume all'inizio e alla fine:
    L = length(x_stretched(1:end));
    smooth_window = zeros(L, 1);
    dt = 0.001; %sec
    t1 = dt * sr;
    t2 = L - t1;
    smooth_window(1:uint32(t1)) = (0:t1-1)/t1;
    smooth_window(uint32(t1):L-uint32(t1)-1) = 1;
    smooth_window(L-uint32(t1):L) = (L:-1:t2)/t1 - L/t1 +1;

    x_stretched = smooth_window.*x_stretched;
    

    % normalizzo per evitare clipping
    max_value = max(abs(x_stretched));
    if max_value > 0.99 
        x_stretched = x_stretched/max_value;
    end
end

function x_pitched = pitch_shifting(x, sr, pitch_factor)
    
    x_stretched = time_stretch(x, sr, 1/pitch_factor);
    

    %x_pitched = custom_resample(x_stretched, sr, pitch_factor);
    x_pitched = resample(x_stretched, uint32(pitch_factor*1000), 1000);
    
    
    % la lunghezza del segnale in ingresso x deve essere uguale a quella
    % del segnale in uscita x_pitched:
    x_pitched = resample(x_pitched, length(x), length(x_pitched));
    %old_t = 1:length(x_pitched);
    %new_t = 1:length(x);
    %x_pitched = interp1(old_t, x_pitched, new_t, 'pchip');
    %x_pitched = x_pitched(:);
end

function f0 = find_pitch(sample, fs)
    % Intervallo di frequenze vocali (Hz)
    voice_range = [80, 200];
    min_lag = floor(fs / voice_range(2));
    max_lag = floor(fs / voice_range(1));

    % Calcolo delle differenze quadrate normalizzate
    diff_areas = zeros(1, max_lag - min_lag + 1);
    for tau = min_lag:max_lag
        diff_areas(tau - min_lag + 1) = get_difference_area(sample, sample(tau+1:end));
    end

    % Ricerca del primo minimo relativo
    delta_lag = 0;
    for tau = 2:length(diff_areas)-1
        if diff_areas(tau) < diff_areas(tau-1) ...
                && diff_areas(tau) < diff_areas(tau+1)
            delta_lag = tau;
            break;
        end
    end

    % Se non trovato, prendo il minimo globale
    %if delta_lag == 0
    %    [~, delta_lag] = min(diff_areas);
    %end
    [~, delta_lag] = min(diff_areas);

    % Conversione in frequenza fondamentale
    f0 = fs / (delta_lag + min_lag - 1);
end

function diff = get_difference_area(x1, x2)
    L = min(length(x1), length(x2));
    x1 = x1(1:L);
    x2 = x2(1:L);
    diff = sum((x1(:) - x2(:)).^2) / L;
end

function best_lag = find_best_lag_2(current_sample, frame, start_index, n_overlap)
    % Calcola il lag massimo consentito
    max_lag = floor(n_overlap / 8);

    % Definisci la regione di ricerca
    search_start = max(1, start_index - max_lag); 
    search_end = min(length(current_sample), start_index + length(frame) + max_lag);

    search_region = current_sample(search_start:search_end);

    % Se la regione è troppo corta, ritorna 0
    if length(search_region) < length(frame)
        best_lag = 0;
        return;
    end

    % Calcola la correlazione incrociata (equivalente a np.correlate con 'valid')
    correlation = xcorr(search_region, frame, 'none');

    % Trova l’offset migliore
    [~, best_offset] = max(correlation);

    % Calcola il lag relativo alla posizione originale
    best_lag = (search_start + best_offset - 1) - start_index;
end

function best_lag = find_best_lag(current_sample, frame, start_index, n_overlap)
    best_lag = 0;
    min_area = realmax;
    max_lag = min(n_overlap / 16, length(frame) / 16);
    for tau = [-uint32(max_lag), uint32(max_lag)]
        chopped_sample = current_sample(start_index + tau : min(start_index + n_overlap, length(current_sample)));
        area = get_difference_area(chopped_sample, frame);
        if area < min_area
            min_area = area;
            best_lag = tau;
        end
    end
end