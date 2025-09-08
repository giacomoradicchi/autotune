//#include <raylib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <limits.h>
#include <complex.h>

#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"
#include "autotune.h"

#define USE_NOTE            "C" //nota usata per tuning
#define USE_SCALE           12 //C minore
#define SAMPLE_RATE         48000 //Hz
#define USE_SURROUND        1 //indica se utilizzare 5 canali o 2

#if USE_SURROUND
#define CHANNELS            6 //6 canali per il surround 5.1
#else
#define CHANNELS            2 //2 canali per stereo
#endif

//#define SAMPLE_LENGTH       4096 * CHANNELS

//static int autotune = false;
ma_device device;
ma_channel_converter converter;
AutotuneParams params;

double input[SAMPLE_LENGTH];
double result[SAMPLE_LENGTH];
float outputAudio[SAMPLE_LENGTH];

void AudioCallback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount)
{
    const float* microphone = (const float*)pInput;
    float* speaker = (float*)pOutput;

    for (ma_uint32 i = 0; i < frameCount; i++) 
       input[i] = microphone[i];

    //test
    //memset(input, 0, SAMPLE_LENGTH * sizeof(double));

    double x = 0;
    double dx = (double)frameCount / (M_PI * 8.0);

    //for (ma_uint32 i = 0; i < frameCount; i++, x += dx)
    //    input[i] = cos(x);

    memset(result, 0, sizeof(result));

    Autotune(&params, input, result, SAMPLE_LENGTH, SAMPLE_RATE, USE_NOTE, USE_SCALE);

    //for (ma_uint32 i = 0; i < frameCount; i++) 
       //speaker[i] = input[i];

    for (ma_uint32 i = 0; i < frameCount; i++)
    {
        float sample = result[i]; // il tuo campione mono

        // Copia lo stesso sample su tutti i canali
        for (int ch = 0; ch < 1; ch++)
        {
            speaker[i * 1 + ch] = sample;
        }
    }

/*


    ma_result result = ma_channel_converter_process_pcm_frames(&converter, speaker, outputAudio, frameCount);
    if (result != MA_SUCCESS) 
    {
        printf("Errore conversione: %d\n", result);
        return;
    }*/

    /* In this example the format and channel count are the same for both input and output which means we can just memcpy(). */
    //MA_COPY_MEMORY(pOutput, pInput, frameCount * ma_get_bytes_per_frame(pDevice->capture.format, pDevice->capture.channels));

    printf("%d\n", frameCount);
}

int StartMicrophone()
{
    ma_result result;
    ma_device_config config;

    config = ma_device_config_init(ma_device_type_duplex);
    config.capture.pDeviceID  = NULL;
    config.capture.format     = ma_format_f32;
    config.capture.channels   = 1;
    config.capture.shareMode  = ma_share_mode_shared;
    config.playback.pDeviceID = NULL;
    config.playback.format    = ma_format_f32;
    config.playback.channels  = 1;
    config.sampleRate         = 48000;
    config.dataCallback       = AudioCallback;
    config.periodSizeInFrames = SAMPLE_LENGTH;

    // Inizializza il device
    result = ma_device_init(NULL, &config, &device);
    if (result != MA_SUCCESS) {
        printf("Errore inizializzazione dispositivo: %d\n", result);
        return 0;
    }

    // Avvia la cattura
    result = ma_device_start(&device);
    if (result != MA_SUCCESS) {
        printf("Errore avvio dispositivo: %d\n", result);
        ma_device_uninit(&device);
        return 0;
    }

    ma_channel_converter_config convConfig = ma_channel_converter_config_init(
        ma_format_f32,                      // Sample format
        1,                              // Input channels
        NULL,                           // Input channel map
        CHANNELS,                       // Output channels
        NULL,                           // Output channel map
        ma_channel_mix_mode_default);   // The mixing algorithm to use when combining channels.

    result = ma_channel_converter_init(&convConfig, NULL, &converter);
    if (result != MA_SUCCESS) 
    {
        printf("Errore converter: %d\n", result);
        return 0;
    }

    return 1;
}

void StopMicrophone()
{
    ma_device_uninit(&device);
}

int main(void) 
{
    if(!StartMicrophone())
        return 1;

    AllocateAutotuneParams(&params);

    // InitWindow(800, 450, "Autotune");
    // SetTargetFPS(60);

    // while (!WindowShouldClose()) 
    // {
    //     BeginDrawing();
    //     ClearBackground(RAYWHITE);

    //     //DrawLine()
    //     EndDrawing();
    // }

    // CloseWindow();

    _getch();

    DestroyAutotuneParams(&params);

    return 0;
}