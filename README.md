# Autotune from Scratch 🎶

Take a look at the [new version](https://github.com/giacomoradicchi/pitch_correction)!
Questo progetto contiene un’implementazione prototipale di un algoritmo di **autotuning** sviluppato interamente da zero, sia in **MATLAB** che in **Python**.

> [Provalo Qui!](https://huggingface.co/spaces/giacomoradicchi/autotune)
## Descrizione
L’approccio seguito si basa su:
- Segmentazione del segnale in frame con finestra di Hanning
- Stima della frequenza fondamentale (pitch detection)
- Correzione verso la nota più vicina in una scala musicale selezionata
- Ricostruzione del segnale mediante tecnica di **overlap-add**

## Finalità
Il progetto ha scopi **didattici e dimostrativi**: non si tratta di una versione ottimizzata o pronta per l’uso in ambito produttivo.

## Sviluppi futuri
Alcuni miglioramenti possibili includono:
- Resampling sinc-based per una migliore qualità spettrale
- Preservazione delle formanti vocali
- Ottimizzazione delle prestazioni computazionali (attualmente penalizzate dall’uso di cicli `for` in linguaggi interpretati)

## Linguaggi
- MATLAB
- Python

---

⚠️ Nota: Questo progetto è sperimentale e destinato principalmente allo studio dei principi di base del pitch shifting automatico. La versione MATLAB è ancora in fase di miglioramento.
