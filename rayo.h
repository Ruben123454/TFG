// ################
// Autores: 
// Mir Ramos, Rubén 869039
//
// rayo.h
// ################

#ifndef RAYO
#define RAYO

#include <cuda_runtime.h>
#include "vector3d.h"

class Rayo {
private:
    Vector3d origen_;
    Vector3d direccion_;

public:
    // Constructor solo para GPU
    __device__ Rayo() : origen_(Vector3d()), direccion_(Vector3d()) {}
    
    __device__ Rayo(const Vector3d& origen, const Vector3d& direccion) 
        : origen_(origen), direccion_(direccion.normalized()) {}

    // Getters
    __device__ Vector3d origen() const { return origen_; }
    __device__ Vector3d direccion() const { return direccion_; }
};

#endif