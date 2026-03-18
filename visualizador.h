#ifndef VISUALIZADOR_H
#define VISUALIZADOR_H

#include <SDL2/SDL.h>
#include <vector>
#include <iostream>
#include "color.h"

class Visualizador {
private:
    SDL_Window* window = nullptr;
    SDL_Renderer* renderer = nullptr;
    SDL_Texture* texture = nullptr;
    std::vector<uint32_t> pixel_buffer;
    int width, height;

public:
    Visualizador(int w, int h, const char* titulo) : width(w), height(h) {
        if (SDL_Init(SDL_INIT_VIDEO) < 0) {
            std::cerr << "Error SDL: " << SDL_GetError() << std::endl;
            return;
        }

        window = SDL_CreateWindow(titulo, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                  width, height, SDL_WINDOW_SHOWN);
        
        renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
        
        texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, 
                                    SDL_TEXTUREACCESS_STREAMING, width, height);
        
        pixel_buffer.resize(width * height);
    }

    ~Visualizador() {
        SDL_DestroyTexture(texture);
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
    }

    bool procesarEventos() {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) return false;
            if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_ESCAPE) return false;
        }
        return true;
    }

    const Uint8* getEstadoTeclado() {
        return SDL_GetKeyboardState(NULL);
    }

    float acesFilmic(float x) const {
        const float a = 2.51f;
        const float b = 0.03f;
        const float c = 2.43f;
        const float d = 0.59f;
        const float e = 0.14f;
        
        x = x * 0.6f;  
        
        return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0f, 1.0f);
    }
    
    void actualizar(const std::vector<Color>& acumulador, int muestras_actuales) {
        float inv_muestras = 1.0f / muestras_actuales;

        #pragma omp parallel for
        for (int i = 0; i < width * height; ++i) {
            Color c = acumulador[i] * inv_muestras;

            float r = acesFilmic(c.r);
            float g = acesFilmic(c.g);
            float b = acesFilmic(c.b);

            uint8_t ir = static_cast<uint8_t>(r * 255.0f);
            uint8_t ig = static_cast<uint8_t>(g * 255.0f);
            uint8_t ib = static_cast<uint8_t>(b * 255.0f);

            pixel_buffer[i] = (255 << 24) | (ir << 16) | (ig << 8) | ib;
        }

        SDL_UpdateTexture(texture, NULL, pixel_buffer.data(), width * sizeof(uint32_t));
        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, NULL, NULL);
        SDL_RenderPresent(renderer);
    }
};

#endif