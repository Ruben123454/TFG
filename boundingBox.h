// ################
// Autores: 
// Mir Ramos, Rubén 869039
//
// boundingBox.h
// ################

#ifndef BOUNDING_BOX
#define BOUNDING_BOX

#include <limits>
#include <algorithm>
#include "vector3d.h"
#include "rayo.h"

class BoundingBox {
public:
    Vector3d min; // Punto mínimo de la caja
    Vector3d max; // Punto máximo de la caja

    // Constructores
    __host__ __device__ BoundingBox() {
        min = Vector3d( 1e30, 1e30, 1e30 );
        max = Vector3d( -1e30, -1e30, -1e30 );
    }

    __host__ __device__ BoundingBox(const Vector3d& min, const Vector3d& max) : min(min), max(max) {}

    // Expandir la caja para incluir otra caja
    __host__ __device__ void expandir(const BoundingBox& otra) {
        min = min.cwiseMin(otra.min);
        max = max.cwiseMax(otra.max);
    }

    // Obtener el eje más largo: 0 = X, 1 = Y, 2 = Z
    __host__ __device__ int ejeMasLargo() const {
        Vector3d tam = max - min; // Tamaño en cada eje
        if (tam.x > tam.y && tam.x > tam.z) return 0; // X
        if (tam.y > tam.z) return 1; // Y
        return 2; // Z
    }

    // Calcular el área superficial de la caja
    __host__ __device__ float area() const {
        Vector3d d = max - min;
        return 2.0f * (d.x * d.y + d.x * d.z + d.y * d.z);
    }

    // Intersección rayo-caja delimitadora usando el método de los slabs (método de los planos paralelos)
    __device__ bool intersecta(const Rayo& r, float& tMin, float& tMax) const {
        tMin = -1e30f;
        tMax =  1e30f;

        // Recorremos cada eje (X, Y, Z)
        for (int i = 0; i < 3; ++i) {
            float invD = 1.0f / r.direccion()[i];  // inversa de la dirección
            float t0 = (min[i] - r.origen()[i]) * invD;
            float t1 = (max[i] - r.origen()[i]) * invD;

            // Si la dirección es negativa, intercambiamos t0 y t1
            float t_temp;
            if (invD < 0.0f) {
                t_temp = t0;
                t0 = t1;
                t1 = t_temp;
            }

            // Actualizamos los límites del intervalo
            if (t0 > tMin) {
                tMin = t0;
            }
            if (t1 < tMax) {
                tMax = t1;
            }

            // Si el intervalo es inválido, el rayo no toca la caja
            if (tMax <= tMin)
                return false;
        }

        return true; // hay solapamiento en los tres ejes
    }

};

#endif