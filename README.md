# python-autotune

Libreria Python per correzione automatica dell’intonazione (autotune) su file audio.

## Installazione

```bash
pip install python-autotune
```

## Utilizzo rapido

```python
from autotune import autotune_audio

autotune_audio("input.wav", "output.wav", key="C major")
```

## Funzionalità
- Pitch detection (stima della frequenza fondamentale)
- Pitch shifting (modifica dell’intonazione)
- Correzione automatica verso la nota più vicina in una scala musicale
- API Python semplice e modulare

## API

### autotune_audio
```python
autotune_audio(input_path, output_path, key="C major")
```
Applica autotune a un file audio WAV.

### tuning
```python
tuning(data, sr, scale="chromatic")
```
Applica autotune a un array numpy.

## Esempio avanzato

```python
import soundfile as sf
from py_autotune.core import tuning

x, sr = sf.read("input.wav")
x_tuned = tuning(x, sr, scale="A minor")
sf.write("output.wav", x_tuned, sr)
```

## Scale supportate
- Maggiori, minori, cromatica

## Dipendenze
- numpy
- scipy
- soundfile

## Test
Esegui i test automatici:
```bash
pytest tests/
```

## Licenza
MIT
