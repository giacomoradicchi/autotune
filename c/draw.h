//disegno su schermo con raylib
#include <stdint.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_mutex.h>

typedef struct
{
    uint16_t* voice;
    uint16_t* tuned;
    uint16_t width, height;
    
    SDL_mutex* mutex;
} DrawParams;

void AllocateDrawParams(DrawParams* params, uint16_t width, uint16_t height);
void DestroyDrawParams(DrawParams* params);

int GraphicsLoop(DrawParams* params);
void SetVoice(DrawParams* params, double* buffer, uint16_t n);
void SetTunedVoice(DrawParams* params, double* buffer, uint16_t n);