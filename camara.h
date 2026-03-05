// ################
// Autores: 
// Mir Ramos, Rubén 869039
//
// camara.h
// ################

#ifndef CAMERA
#define CAMERA

#include <cuda_runtime.h>
#include "rayo.h"
#include "vector3d.h"

struct Camara {
    Vector3d posicion;
    Vector3d frente;
    Vector3d arriba;
    Vector3d izquierda;
    float ancho_plano, alto_plano;
    float distancia_focal;
    
    __device__ Camara() = default;
    
    __device__ Camara(Vector3d pos, Vector3d fr, Vector3d arr, Vector3d izq, 
                        float ancho_p, float alto_p, float dist_focal) 
        : posicion(pos), frente(fr), arriba(arr), izquierda(izq),
          ancho_plano(ancho_p), alto_plano(alto_p), distancia_focal(dist_focal) {}

    __device__ Camara(const Vector3d& pos = Vector3d(0, 0, 0), 
           const Vector3d& f = Vector3d(0, 0, -1), 
           const Vector3d& a = Vector3d(0, 1, 0), 
           double ancho = 2.0, 
           double alto = 2.0, 
           double dist = 1.0)
        : posicion(pos), frente(f.normalized()), arriba(a.normalized()), ancho_plano(ancho), alto_plano(alto), distancia_focal(dist) {

        izquierda = frente.cross(arriba).normalized();
        arriba = izquierda.cross(frente).normalized(); // Recalcular arriba para asegurar ortogonalidad
    }
    
    __device__ Rayo generarRayo(int x, int y, int ancho_img, int alto_img, 
                               float offset_x, float offset_y) const {
        // Convertir coordenadas de píxel a coordenadas del plano
        float u = (x + offset_x) / ancho_img;
        float v = (y + offset_y) / alto_img;
        
        float px = (2.0f * u - 1.0f) * (ancho_plano * 0.5f);
        float py = (1.0f - 2.0f * v) * (alto_plano * 0.5f);
        
        Vector3d punto_plano = posicion + 
                               frente * distancia_focal + 
                               izquierda * px + 
                               arriba * py;

        Vector3d direccion = (punto_plano - posicion).normalized();

        return Rayo{posicion, direccion};
    }
};

#endif