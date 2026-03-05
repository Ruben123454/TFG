// ################
// Autores: 
// Mir Ramos, Rubén 869039
//
// imagen_gpu.h
// ################

#ifndef IMAGEN_GPU
#define IMAGEN_GPU

#include <cuda_runtime.h>
#include "color.h"

class ImagenGPU {
public:
    int anchura, altura;
    Color* datos;

    // Constructor para GPU que recibe memoria ya reservada
    __host__ ImagenGPU(int anch, int alt, Color* data_ptr) 
        : anchura(anch), altura(alt), datos(data_ptr) {}

    // Método para asignar un color a un píxel
    __device__ void setPixel(int x, int y, const Color& color) {
        if (x >= 0 && x < anchura && y >= 0 && y < altura) {
            datos[y * anchura + x] = color;
        }
    }

    __device__ Color getPixel(int x, int y) const {
        if (x >= 0 && x < anchura && y >= 0 && y < altura) {
            return datos[y * anchura + x];
        }
        return Color(0, 0, 0); // Devolver  negro si está fuera de los límites 
    }
    
};

#endif