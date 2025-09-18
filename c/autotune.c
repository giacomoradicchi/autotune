//implementazione by SimoSbara

#include "autotune.h"
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <math.h>
#include <float.h>
#include <limits.h>

void PrintMinMax(double* xt, int n)
{
    double min = xt[0];
    double max = xt[0];

   // printf("begin min %f\n", min);
   // printf("begin min %f\n", max);

    for(int i = 1; i < n; i++)
    {
        double x = xt[i];

        if(x < min)
        {
            //printf("min %f\n", x);
            min = x;
        }
        else if(x > max)
        {
            //printf("max %f\n", x);
            max = x;
        }
    }
}

void Normalize(double* xt, int n)
{
    double min = xt[0];
    double max = xt[0];
    double range;

  //  printf("begin norm min %f\n", min);
   // printf("begin norm max %f\n", max);

    for(int i = 1; i < n; i++)
    {
        double x = xt[i];

        if(x < min)
        {
                
            min = x;
        }
        else if(x > max)
        {

            max = x;
        }
    }


    //printf("end norm min %f\n", min);
    //printf("end norm max %f\n", max);

    range = max - min;

    for(int i = 0; i < n; i++)
    {
        xt[i] = ((xt[i] - min) / range) * 2.0 - 1;
    }

    //printf("normalized\n");
    //PrintMinMax(xt, n);
}

double* HanningWindow(uint32_t m)
{
    double* w = malloc(m * sizeof(double));

    for(uint32_t i = 0; i < m; i++)
        w[i] = 0.5 - 0.5 * cos(2.0 * M_PI * i / (m - 1));

    return w;
}

//filtraggio con media mobile
double MAFilter(double* pitchFactors, uint8_t n, double* filter, uint8_t m)
{   
    double sum = 0;

    //filtro più grande (per i primi MA_LENGTH valori)
    if(m > n)
    {
        int start = m - n;

        for(int i = 0, j = start; i < n; i++, j++)
            sum += pitchFactors[i] * filter[j];
    }
    else
    {
        int start = n - m;

        for(int i = start, j = 0; i < n; i++, j++)
            sum += pitchFactors[i] * filter[j];
    }

    return sum;
}

double GetDiffArea(double* x1, uint32_t n1, double* x2, uint32_t n2)
{
    int len = (n1 < n2) ? n1 : n2;
    double sum = 0;

    for(int i = 0; i < len; i++)
    {
        sum += (x1[i] - x2[i]) * (x1[i] - x2[i]);
    }

    return sum / len;
}

double GetNoteFrequency(const char* note, uint8_t octave)
{
    int8_t index = -1;

    //cerco nella scala cromatica
    for(int i = 0; i < 12; i++)
    {
        if(!strcmp(chromaticScale[i], note))
        {
            index = i;
            break;
        }
    }

    if(index < -1)
        return 0;

    double comp1 = ((double)index - 9.0) / 12.0;
    double comp2 = octave - 4;

    return A_FREQUENCY * pow(2, comp1) * pow(2, comp2);
}

double FindPitch(double* frame, uint32_t n, uint32_t sampleRate)
{
    double minLag = sampleRate / VOICE_MAX_F;
    double maxLag = sampleRate / VOICE_MIN_F;
    double deltaLag = DBL_MAX;

    for(uint32_t tau = minLag; tau <= maxLag; tau++)
    {
        double diff = GetDiffArea(frame, n, frame + tau, n - tau);

        if(diff < deltaLag)
            deltaLag = diff;
    }

    //tolgo, dal codice originale ha cambiato metodo...
    /*
    double deltaLag = 0;
    double* diffs = malloc((maxLag - minLag) * sizeof(double));

    for(uint32_t tau = minLag; tau <= maxLag; tau++)
        diffs[tau] = GetDiffArea(frame, n, frame + tau, n - tau);

    for(uint32_t tau = minLag + 1; tau <= maxLag - 1; tau++)
    {
        //trovo il primo minimo, un pò greedy... ma voglio rispettare chi lo ha scritto
        if(diffs[tau - 1] > diffs[tau] && diffs[tau] < diffs[tau + 1])
        {
            deltaLag = diffs[tau];
            break;
        }
    }
    */

    return (double)sampleRate / (deltaLag + minLag);
}

//migliorato rispetto alla versione originale
double FindClosestNote(double pitch, uint8_t scaleIndex, const char* note)
{
    double f = GetNoteFrequency(note, 0);
    uint8_t octave = log2(pitch / f);

    int8_t closest = 0;
    double closestF = GetNoteFrequency(scales[scaleIndex][0], octave);
    double closestDelta = fabs(closestF - pitch);

    for(int i = 1; i < NOTES; i++)
    {
        double fnote = GetNoteFrequency(scales[scaleIndex][i], octave);
        double delta = fabs(fnote - pitch);

        if(delta < closestDelta)
        {
            closest = i;
            closestF = fnote;
            closestDelta = delta;
        }
    }

    return closestF;
}

//cross correlation
uint16_t FindBestLag(double* xt, uint16_t n, double* yt, uint16_t m, uint16_t start, uint16_t overlap)
{
    uint16_t maxLag = overlap >> 3;
    uint16_t searchStart = fmax(0, start - maxLag);
    uint16_t searchEnd = fmin(n, start + m + maxLag);
    uint16_t searchSize = searchEnd - searchStart;

    if(searchSize < m)
        return 0;

    double best = DBL_MAX;
    uint16_t bestOffset = -1;

    //SDF come indicato da Giacomo
    for(uint16_t lag = 0; lag < searchSize - m - 1; lag++)
    {
        double diff = GetDiffArea(xt + lag, m, yt, m);

        if(diff < best)
        {
            best = diff;
            bestOffset = lag;
        }
    }

    return fmax(0, searchStart + bestOffset - start);
}

//il cuore
uint16_t ApplyTimeStretch(AutotuneParams* params, double* xt, uint16_t n, double stretchFactor)
{
    //distribuzione di fase e ampiezze
    //double phases[NK][FRAME_LENGTH];
    //double amplitudes[NK][FRAME_LENGTH];
    double phaseAcc[N_BINS];

    memset(phaseAcc, 0, sizeof(phaseAcc));

    //stft sarebbero più dft con una finestra mobile (hanning in questo caso)
    //una spiegazione più accurata è su matlab 
    //https://it.mathworks.com/help/signal/ref/stft.html#mw_d9607346-5bbf-4fa5-8792-c1021fc5bf3e
    for(int i = 0, offset = 0; i < NK; i++, offset += HOP_LENGTH)
    {
        double* input = params->realFFT;
        fftw_complex* output = params->complexFFT[i];

        //si può vettorizzare con SIMD
        for(int j = 0; j < FRAME_LENGTH; j++)
            input[j] = xt[offset + j] * params->window[j];

        //fft grazie alla libreria FFTW... la stessa usata internamente da Matlab
        //sempre in input un segnale reale
        fftw_execute_dft_r2c(params->frameFFTPlan, input, output);

        //si può vettorizzare anche questo con SIMD
        // for(int j = 0; j < N_BINS; j++)
        // {
        //     double a = output[j][0];
        //     double b = output[j][1];

        //     phases[i][j] = atan2(b, a);
        //     amplitudes[i][j] = sqrt(a * a + b * b);
        // }
    }

    uint16_t stretchedCount = (uint16_t)((double)(NK / stretchFactor));

    if(stretchedCount < 1)
        stretchedCount = 1;

    uint16_t outputSize = (stretchedCount - 1) * HOP_LENGTH + FRAME_LENGTH;
    double* timeSteps = malloc(stretchedCount * sizeof(double));

    if(params->curStretchedSize < outputSize)
    {
        if(params->curStretchedSize > 0)
        {
            fftw_free(params->stretchedFrame);
            params->stretchedFrame = fftw_malloc(outputSize * sizeof(double));
        }
        else
            params->stretchedFrame = fftw_malloc(outputSize * sizeof(double));

        params->curStretchedSize = outputSize;
    }

    for(int i = 0; i < outputSize; i++)
        params->stretchedFrame[i] = 0;

    for (int i = 0; i < stretchedCount; i++) 
        timeSteps[i] = (double)i * stretchFactor;

    double step = 0;
    fftw_complex z;

    //interpoliamo!
    for (int i = 0, offset = 0; i < stretchedCount; i++, step += stretchFactor, offset += HOP_LENGTH) 
    {
        int index = floor(step);
        double frac = step - index;

        fftw_complex* S1 = params->complexFFT[index];
        fftw_complex* S2 = params->complexFFT[(int)fmin(index + 1, NK - 1)];

        for (int k = 0; k < N_BINS; k++) 
        {
            //modulo
            double mag1 = sqrt(S1[k][0] * S1[k][0] + S1[k][1] * S1[k][1]);
            double mag2 = sqrt(S2[k][0] * S2[k][0] + S2[k][1] * S2[k][1]);
            double m = (1-frac)*mag1 + frac*mag2;

            //fase
            double phase1 = atan2(S1[k][1], S1[k][0]);
            double phase2 = atan2(S2[k][1], S2[k][0]);
            double f = phase2 - phase1;

            //wrapping nel range [-pi, pi]
            f = fmod(f + M_PI, 2*M_PI) - M_PI;

            //accumulo fase
            phaseAcc[k] += f;

            //avvaliamoci di eulero per poter comporre e^j*theta
            z[0] = m * cos(phaseAcc[k]);
            z[1] = m * sin(phaseAcc[k]);

            //STFT stretchata
            params->stretchedFFT[k][0] = z[0];
            params->stretchedFFT[k][1] = z[1];

            if(isnan(z[0]) || isnan(z[1]))
            {
                printf("nan %d\n", k);
            }
        }

        for(int k = 0; k < FRAME_LENGTH; k++)
            params->realFFT[k] = 0;

        //faccio subito la trasformata inversa del frame attuale
        //da complesso a reale
        fftw_execute_dft_c2r(params->frameIFFTPlan, params->stretchedFFT, params->realFFT);

        for (int j = 0; j < FRAME_LENGTH; j++)
        {
            params->stretchedFrame[j + offset] += params->realFFT[j] * params->window[j];
        }

        //PrintMinMax(params->realFFT, FRAME_LENGTH);
        //PrintMinMax(params->window, FRAME_LENGTH);
    }

    return outputSize;
}

//con l'uso dell'interpolazione lineare
void Resample(double* input, uint16_t n, double* output, uint16_t m, double step)
{
    double x1, x2, y1, y2;
    double s = 0;
    double val;

    for(int i = 0; i < m - 2; i++, s += step)
    {
        x1 = (int)s;
        x2 = (int)s + 1;
        y1 = input[(int)x1];
        y2 = input[(int)x2];
        
        val = ((x2 - s) * y1 + (s - x1) * y2) / (x2 - x1);

        output[i] = val;
    }
}

//il secondo cuore
void PitchShifting(AutotuneParams* params, double* xt, int n, uint32_t sampleRate, double pitchFactor)
{
    uint16_t stretchSize = ApplyTimeStretch(params, xt, n, 1.0 / pitchFactor);
    uint16_t samplingSize = stretchSize / pitchFactor;

    if(params->curResampleSize < samplingSize)
    {
        if(params->curResampleSize > 0)
            params->resampledFrame = realloc(params->resampledFrame, samplingSize * sizeof(double));
        else
            params->resampledFrame = malloc(samplingSize * sizeof(double));

        params->curResampleSize = samplingSize;
    }

    //printf("stretched\n");
    //PrintMinMax(params->stretchedFrame, stretchSize);

    //upsampling
    Resample(params->stretchedFrame, stretchSize, params->resampledFrame, samplingSize, pitchFactor);
    
    //printf("resampled\n");
    //PrintMinMax(params->resampledFrame, samplingSize);

    //sampling alla lunghezza dell'input
    Resample(params->resampledFrame, samplingSize, xt, FRAME_LENGTH, (double)samplingSize / (double)FRAME_LENGTH);

    //printf("resampled 2\n");
    //PrintMinMax(xt, FRAME_LENGTH);
}

//alloco tutti i parametri
void AllocateAutotuneParams(AutotuneParams* params)
{
    params->window = HanningWindow(FRAME_LENGTH);

    double sum = 0;
    params->filterWindow = malloc(MA_LENGTH * sizeof(double));
    for(int i = 0; i < MA_LENGTH; i++)
    {
        double value = i * i * i;
        params->filterWindow[i] = value;
        sum += value;
    }

    //normalizzazione
    for(int i = 0; i < MA_LENGTH; i++)
    {
        params->filterWindow[i] /= sum; 
        printf("filter %f\n", params->filterWindow[i]);
    }

    params->frame = malloc(FRAME_LENGTH * sizeof(double));
    params->realFFT = fftw_malloc(FRAME_LENGTH * sizeof(double));
    params->stretchedFFT = fftw_malloc(FRAME_LENGTH * sizeof(fftw_complex));

    for(int i = 0; i < NK; i++)
        params->complexFFT[i] = fftw_malloc(FRAME_LENGTH * sizeof(fftw_complex));

    params->frameFFTPlan = fftw_plan_dft_r2c_1d(FRAME_LENGTH, params->realFFT, params->complexFFT[0], FFTW_ESTIMATE);
    params->frameIFFTPlan = fftw_plan_dft_c2r_1d(FRAME_LENGTH, params->complexFFT[0], params->realFFT, FFTW_ESTIMATE);

    params->curStretchedSize = 0;
    params->curResampleSize = 0;

    memset(params->pitchFactors, 0, NP * sizeof(double));
}

void DestroyAutotuneParams(AutotuneParams* params)
{
    free(params->window);
    free(params->filterWindow);
    free(params->frame);
    
    fftw_destroy_plan(params->frameFFTPlan);
    fftw_destroy_plan(params->frameIFFTPlan);

    fftw_free(params->realFFT);
    fftw_free(params->stretchedFFT);

    for(int i = 0; i < NK; i++)
        fftw_free(params->complexFFT[i]);
}

double* Autotune(AutotuneParams* params, double* audio, double* tuned, int n, uint16_t sampleRate, const char* note, uint8_t scale)
{
    double frame[FRAME_LENGTH];

    memset(tuned, 0, n * sizeof(double));

    for(int i = 0; i < n; i += HOP_LENGTH)
    {
        if(i + FRAME_LENGTH >= n)
        {
            memset(frame, 0, FRAME_LENGTH * sizeof(double));
            memcpy(frame, audio + i, fmin(n - i, FRAME_LENGTH) * sizeof(double));
        }
        else
            memcpy(frame, audio + i, FRAME_LENGTH * sizeof(double));

        for(int j = 0; j < FRAME_LENGTH; j++)
            frame[j] *= params->window[j];

        double pitch = FindPitch(frame, FRAME_LENGTH, sampleRate);
        double goal = FindClosestNote(pitch, scale, note);

        params->pitchFactors[i] = goal / pitch; //il famoso rapporto

        double smoothingFactor = MAFilter(params->pitchFactors, i + 1, params->filterWindow, MA_LENGTH);
        double fract = fabs(smoothingFactor - goal) / fabs(smoothingFactor + goal);
        smoothingFactor = smoothingFactor * fract + 1 - fract;

        PitchShifting(params, frame, FRAME_LENGTH, sampleRate, smoothingFactor);

        uint16_t tau = 0;

        if(i > 0)
            tau = FindBestLag(tuned, n, frame, FRAME_LENGTH, i, FRAME_LENGTH - HOP_LENGTH);

        //printf("tau %d\n", tau);

        uint16_t end = fmin(n - 1, i + tau + FRAME_LENGTH);
        uint16_t start = fmin(end, fmax(0, i + tau));

        for(uint16_t j = start, k = 0; k < FRAME_LENGTH && j <= end; j++, k++)
            tuned[j] += tuned[j] + frame[k];
    }

    //normalizzo tra -1 e 1
    Normalize(tuned, n);

    return tuned;
}