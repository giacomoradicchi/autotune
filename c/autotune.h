//implementazione by SimoSbara
#include <fftw3.h>
#include <stdint.h>

#define SCALES              25
#define NOTES               7
#define SAMPLE_LENGTH       4096
#define FRAME_LENGTH        2048
#define N_BINS              (FRAME_LENGTH / 2) + 1
#define HOP_LENGTH          512
#define MA_LENGTH           5
#define NK                  1 //+ (SAMPLE_LENGTH -  FRAME_LENGTH) / HOP_LENGTH //numero trasformate
#define NP                  SAMPLE_LENGTH / HOP_LENGTH //numero dei pitch factor

#define VOICE_MIN_F         80 //hz
#define VOICE_MAX_F         200 //hz
#define A_FREQUENCY         440.0 //hz, sarebbe La

typedef struct
{
    double* window;
    double* filterWindow; //ma filter
    double* frame;

    double* resampledFrame;
    uint16_t curResampleSize;

    double* stretchedFrame;
    uint16_t curStretchedSize;

    double* realFFT;
    fftw_complex* complexFFT[NK];
    fftw_complex* stretchedFFT;
    fftw_plan frameFFTPlan;
    fftw_plan frameIFFTPlan;

    double pitchFactors[NP];
} AutotuneParams;

static const char* chromaticScale[12] = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};

static const char* scales[SCALES][NOTES] =
{
    /* major scales */
    {"C", "D", "E", "F", "G", "A", "B"},                /* C major */
    {"C#", "D#", "F", "F#", "G#", "A#", "C"},           /* C# major */
    {"D", "E", "F#", "G", "A", "B", "C#"},              /* D major */
    {"D#", "F", "G", "G#", "A#", "C", "D"},             /* D# major */
    {"E", "F#", "G#", "A", "B", "C#", "D#"},            /* E major */
    {"F", "G", "A", "A#", "C", "D", "E"},               /* F major */
    {"F#", "G#", "A#", "B", "C#", "D#", "F"},           /* F# major */
    {"G", "A", "B", "C", "D", "E", "F#"},               /* G major */
    {"G#", "A#", "C", "C#", "D#", "F", "G"},            /* G# major */
    {"A", "B", "C#", "D", "E", "F#", "G#"},             /* A major */
    {"A#", "C", "D", "D#", "F", "G", "A"},              /* A# major */
    {"B", "C#", "D#", "E", "F#", "G#", "A#"},           /* B major */

    /* minor scales */
    {"C", "D", "D#", "F", "G", "G#", "A#"},             /* C minor */
    {"C#", "D#", "E", "F#", "G#", "A", "B"},            /* C# minor */
    {"D", "E", "F", "G", "A", "A#", "C"},               /* D minor */
    {"D#", "F", "F#", "G#", "A#", "B", "C#"},           /* D# minor */
    {"E", "F#", "G", "A", "B", "C", "D"},               /* E minor */
    {"F", "G", "G#", "A#", "C", "C#", "D#"},            /* F minor */
    {"F#", "G#", "A", "B", "C#", "D", "E"},             /* F# minor */
    {"G", "A", "A#", "C", "D", "D#", "F"},              /* G minor */
    {"G#", "A#", "B", "C#", "D#", "E", "F#"},           /* G# minor */
    {"A", "B", "C", "D", "E", "F", "G"},                /* A minor */
    {"A#", "C", "C#", "D#", "F", "F#", "G#"},           /* A# minor */
    {"B", "C#", "D", "E", "F#", "G", "A"},              /* B minor */
};

void AllocateAutotuneParams(AutotuneParams* params);
void DestroyAutotuneParams(AutotuneParams* params);
double* Autotune(AutotuneParams* params, double* audio, double* tuned, int n, uint16_t sampleRate, const char* note, uint8_t scale);
