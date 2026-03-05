// ################
// Autores:
// Mir Ramos, Rubén 869039
//
// primitiva.cu
// ################

#ifndef PRIMITIVA
#define PRIMITIVA

#include <cuda_runtime.h>
#include "color.h"
#include "vector3d.h"
#include "rayo.h"
#include "boundingBox.h"

enum TipoPrimitiva { ESFERA, PLANO, TRIANGULO };

struct Primitiva {
    TipoPrimitiva tipo;
    Color emision;
    Color difuso;
    Color especular;
    Color transmision;
    float indice_refraccion = 1.0f;

    // Union para almacenar cualquier tipo de primitiva
    union {
        struct { 
            Vector3d centro; 
            float radio; 
        } esfera;
        struct { 
            Vector3d normal; 
            float distancia; 
        } plano;
        struct { 
            Vector3d v0, v1, v2; 
            Vector3d v0_normal, v1_normal, v2_normal;
        } triangulo;
    };

    __device__ bool intersecta(const Rayo& r, float& t) const {
        switch(tipo) {
            case ESFERA:
                return intersectaEsfera(r, t);
            case PLANO:
                return intersectaPlano(r, t);
            case TRIANGULO:
                return intersectaTriangulo(r, t);
            default:
                return false;
        }
    }

    // Función extendida que también devuelve coordenadas baricéntricas para triángulos
    __device__ bool intersecta(const Rayo& r, float& t, float& u, float& v) const {
        switch(tipo) {
            case TRIANGULO:
                return intersectaTriangulo(r, t, u, v);
            default:
                // Para otras primitivas, usar la función normal y poner u,v a 0
                u = 0.0f;
                v = 0.0f;
                return intersecta(r, t);
        }
    }

    __device__ Vector3d calcularNormal(const Vector3d& punto) const {
        switch(tipo) {
            case ESFERA:
                return calcularNormalEsfera(punto);
            case PLANO:
                return calcularNormalPlano();
            case TRIANGULO: {
                return calcularNormalTriangulo(punto);
            }
            default:
                return Vector3d(0, 0, 0);
        }
    }

    // Función para calcular normal suavizada (solo para triángulos)
    __device__ Vector3d calcularNormalSuavizada(const Vector3d& punto, float u, float v) const {
        // n = (b0*n0 + b1*n1 + b2*n2) / ||b0*n0 + b1*n1 + b2*n2||
        float b0 = 1.0f - u - v;  // b0
        float b1 = u;             // b1  
        float b2 = v;             // b2
        
        Vector3d normal_suavizada = b0 * triangulo.v0_normal + 
                                   b1 * triangulo.v1_normal + 
                                   b2 * triangulo.v2_normal;
        
        // Normalizar
        float length = sqrt(normal_suavizada.dot(normal_suavizada));
        if (length > 1e-6f) {
            normal_suavizada = normal_suavizada * (1.0f / length);
            return normal_suavizada;
        }
        return Vector3d(0, 0, 0);
    }

    // Obtener la caja delimitadora de la primitiva (para BVH en malla de triángulos)
    BoundingBox obtenerCaja() const {
        if (tipo == TRIANGULO) {
            Vector3d min = triangulo.v0.cwiseMin(triangulo.v1).cwiseMin(triangulo.v2);
            Vector3d max = triangulo.v0.cwiseMax(triangulo.v1).cwiseMax(triangulo.v2);
            return BoundingBox(min, max);
        }
        return BoundingBox(); // Caja vacía para otras primitivas
    }

    // Calcular centroide de la primitiva (para BVH en malla de triángulos)
    Vector3d centroide() const{
        BoundingBox caja = obtenerCaja();
        return (caja.min + caja.max) * 0.5;
    }

private:

    __device__ bool intersectaEsfera(const Rayo& r, float& t) const {
        Vector3d oc = r.origen() - esfera.centro;
        float a = r.direccion().dot(r.direccion());
        float b = 2.0f * oc.dot(r.direccion());
        float c = oc.dot(oc) - esfera.radio * esfera.radio;
        float discriminante = b * b - 4.0f * a * c;
        
        if (discriminante < 0) return false;
        
        float sqrt_disc = sqrtf(discriminante);
        float t1 = (-b - sqrt_disc) / (2.0f * a);
        float t2 = (-b + sqrt_disc) / (2.0f * a);
        
        // Encontrar la intersección más cercana positiva
        if (t1 > 1e-6f && t1 < t2) { 
            t = t1; 
            return true; 
        }
        if (t2 > 1e-6f) { 
            t = t2; 
            return true; 
        }
        return false;
    }
    
    __device__ bool intersectaPlano(const Rayo& r, float& t) const {
        float denom = plano.normal.dot(r.direccion());
        if (fabsf(denom) > 1e-6f) {
            Vector3d p0 = plano.normal * plano.distancia;
            t = (p0 - r.origen()).dot(plano.normal) / denom;
            return (t > 1e-6f);
        }
        return false;
    }
    
    // Método de intersección con triángulo usando el algoritmo Möller–Trumbore
    __device__ bool intersectaTriangulo(const Rayo& r, float& t) const {
        Vector3d v0v1 = triangulo.v1 - triangulo.v0;
        Vector3d v0v2 = triangulo.v2 - triangulo.v0;
        Vector3d pvec = r.direccion().cross(v0v2);
        float det = v0v1.dot(pvec);
        
        // Rayo paralelo al triángulo
        if (fabsf(det) < 1e-6f) return false;
        
        float invDet = 1.0f / det;
        Vector3d tvec = r.origen() - triangulo.v0;
        float u = tvec.dot(pvec) * invDet;
        if (u < 0.0f || u > 1.0f) return false;
        
        Vector3d qvec = tvec.cross(v0v1);
        float v = r.direccion().dot(qvec) * invDet;
        if (v < 0.0f || u + v > 1.0f) return false;
        
        t = v0v2.dot(qvec) * invDet;
        return (t > 1e-6f);
    }

    // Versión extendida que devuelve coordenadas baricéntricas
    __device__ bool intersectaTriangulo(const Rayo& r, float& t, float& u, float& v) const {
        Vector3d v0v1 = triangulo.v1 - triangulo.v0;
        Vector3d v0v2 = triangulo.v2 - triangulo.v0;
        Vector3d pvec = r.direccion().cross(v0v2);
        float det = v0v1.dot(pvec);
        
        // Rayo paralelo al triángulo
        if (fabsf(det) < 1e-6f) return false;
        
        float invDet = 1.0f / det;
        Vector3d tvec = r.origen() - triangulo.v0;
        u = tvec.dot(pvec) * invDet;
        if (u < 0.0f || u > 1.0f) return false;
        
        Vector3d qvec = tvec.cross(v0v1);
        v = r.direccion().dot(qvec) * invDet;
        if (v < 0.0f || u + v > 1.0f) return false;
        
        t = v0v2.dot(qvec) * invDet;
        return (t > 1e-6f);
    }

    __device__ Vector3d calcularNormalEsfera(const Vector3d& punto) const {
        return (punto - esfera.centro).normalized();
    }

    __device__ Vector3d calcularNormalPlano() const {
        return plano.normal;
    }

    __device__ Vector3d calcularNormalTriangulo(const Vector3d& punto) const {
        Vector3d normal = (triangulo.v1 - triangulo.v0).cross(triangulo.v2 - triangulo.v0);
    
        Vector3d haciaCamara = -punto;
        
        if (normal.dot(haciaCamara) < 0) {
            normal = normal * -1.0f; // Invertir si apunta en dirección contraria
        }
        
        return normal.normalized();
    }
};


#endif