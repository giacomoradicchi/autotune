#include "draw.h"

void AllocateDrawParams(DrawParams* params, uint16_t width, uint16_t height)
{
    params->width = width;
    params->height = height;

    params->voice = malloc(params->width * sizeof(uint16_t));
    params->tuned = malloc(params->width * sizeof(uint16_t));
    params->mutex = SDL_CreateMutex();

    memset(params->voice, 0, params->width * sizeof(uint16_t));
    memset(params->tuned, 0, params->width * sizeof(uint16_t));
}

void DestroyDrawParams(DrawParams* params)
{
    free(params->voice);
    free(params->tuned);
    SDL_DestroyMutex(params->mutex);
}

int GraphicsLoop(DrawParams* params)
{
    if (SDL_Init(SDL_INIT_VIDEO) != 0) 
    {
        printf("SDL_Init Error: %s\n", SDL_GetError());
        return 0;
    }

    SDL_Window* win = SDL_CreateWindow("Autotune",
                                       SDL_WINDOWPOS_CENTERED,
                                       SDL_WINDOWPOS_CENTERED,
                                       params->width, params->height,
                                       SDL_WINDOW_SHOWN);

    if (!win) 
    {
        printf("SDL_CreateWindow Error: %s\n", SDL_GetError());
        SDL_Quit();
        return 0;
    }

    SDL_Renderer* ren = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED);
    if (!ren) 
    {
        SDL_DestroyWindow(win);
        SDL_Quit();
        return 1;
    }

    int running = 1;
    SDL_Event e;
    while (running) 
    {
        while (SDL_PollEvent(&e)) 
        {
            if (e.type == SDL_QUIT)
                running = 0;
        }

        SDL_SetRenderDrawColor(ren, 0, 0, 0, 255);
        SDL_RenderClear(ren);

        SDL_SetRenderDrawColor(ren, 0, 255, 0, 255);

        SDL_LockMutex(params->mutex);
        for(int x = 0; x < params->width - 1; x++)
        {
            SDL_RenderDrawLine(ren, x, params->voice[x], x + 1, params->voice[x + 1]);
            SDL_RenderDrawLine(ren, x, params->tuned[x], x + 1, params->tuned[x + 1]);
        }
        SDL_UnlockMutex(params->mutex);     

        SDL_RenderPresent(ren);
    }

    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(win);
    SDL_Quit();

    return 1;
}

void SetVoice(DrawParams* params, double* buffer, uint16_t n)
{
    SDL_LockMutex(params->mutex);

    double x = 0;
    double dx = (double)n / (double)params->width;
    double h2 = params->height >> 1;

    for(uint16_t i = 0; i < params->width; i++, x += dx)
        params->voice[i] = ((buffer[(int)x] + 1.0) * 0.5) * h2;

    SDL_UnlockMutex(params->mutex);
}

void SetTunedVoice(DrawParams* params, double* buffer, uint16_t n)
{
    SDL_LockMutex(params->mutex);

    double x = 0;
    double dx = (double)n / (double)params->width;
    double h2 = params->height >> 1;

    for(uint16_t i = 0; i < params->width; i++, x += dx)
        params->tuned[i] = ((buffer[(int)x] + 1.0) * 0.5) * h2;

    SDL_UnlockMutex(params->mutex);
}