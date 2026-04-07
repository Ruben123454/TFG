// ################
// rng.h
// Generador de números aleatorios PCG32
// ################

#ifndef RNG_H
#define RNG_H

#include <cuda_runtime.h>

// Generador de 32 bits sin estado global.
__device__ inline uint32_t pcg32(uint32_t& state) {
    state = state * 747796405U + 2891336453U;
    
    uint32_t word = ((state >> ((state >> 28U) + 4U)) ^ state) * 277803737U;
    return (word >> 22U) ^ word;
}

// Genera un número uniforme entre 0.0 y 1.0
__device__ inline float pcg32_float(uint32_t& state) {
    return pcg32(state) * (1.0f / 4294967296.0f);
}

// Inicializa semilla
__device__ inline uint32_t inicializarSemilla(int px, int py, int frame_number) {
    uint32_t state = (uint32_t)(py << 16) | (uint32_t)(px & 0xFFFF);
    state ^= (uint32_t)frame_number * 719393U; 
    
    // Hash de Wang para mezclar
    state = (state ^ 61U) ^ (state >> 16U);
    state *= 0x27d4eb2dU;
    state = (state ^ (state >> 15U));
    
    return (state == 0U) ? 1U : state;
}

#endif // RNG_H