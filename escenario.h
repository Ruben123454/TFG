// ################
// Autores: 
// Mir Ramos, Rubén 869039
//
// escenario.h
// ################

#ifndef ESCENARIO
#define ESCENARIO

#include <cuda_runtime.h>
#include "primitiva.h"
#include "luzPuntual.h"
#include "bvh.h"
#include "tinybvh_wrapper.h"
#include "traverse_cwbvh.cuh"

class Escenario {
public:
    // PRIMITIVAS MANUALES
    Primitiva* primitivas;  // Array de estructuras manuales
    int num_primitivas;
    LuzPuntual* luces; // Array de luces puntuales
    int num_luces;

    // PRIMITIVAS DE MALLA (triángulos del modelo cargado)
    Primitiva* primitivas_malla;
    int num_primitivas_malla;

    // BVH propio
    const NodoBVH* nodos_bvh;        // Puntero a nodos BVH en GPU
    const Primitiva* primitivas_bvh; // Array de primitivas de la malla de triángulos
    int num_nodos_bvh;

    // BVH de tiny_bvh
    TinyBVHD_GPU tiny_bvh; // Descriptor de buffers tiny_bvh en GPU (por valor)
    const Primitiva* primitivas_tiny_bvh; // Array de primitivas de la malla de triángulos para tiny_bvh
    int num_primitivas_tiny_bvh;

    const float EPSILON = 1e-3f;

    // Constructor para GPU
    __device__ Escenario() : primitivas(nullptr), num_primitivas(0), luces(nullptr), num_luces(0) {}

    // Constructor para GPU con memoria pre-asignada
    __device__ Escenario(Primitiva* prims, int num_prims, LuzPuntual* luces, int num_luces) 
        : primitivas(prims), num_primitivas(num_prims), luces(luces), num_luces(num_luces) {}

    // Constructor para GPU con BVH propio
    __device__ Escenario(Primitiva* prims, int num_prims, LuzPuntual* luces, int num_luces, 
                           Primitiva* prims_malla, int num_prims_malla,
                           const NodoBVH* nodos_bvh_, const Primitiva* primitivas_bvh_, int num_nodos_bvh_) 
        : primitivas(prims), num_primitivas(num_prims), luces(luces), num_luces(num_luces),
          primitivas_malla(prims_malla), num_primitivas_malla(num_prims_malla),
          nodos_bvh(nodos_bvh_), primitivas_bvh(primitivas_bvh_), num_nodos_bvh(num_nodos_bvh_),
          tiny_bvh{nullptr, nullptr, 0u, 0u},
          primitivas_tiny_bvh(nullptr), num_primitivas_tiny_bvh(0) {}

    // Constructor para GPU con BVH de tiny_bvh
    __device__ Escenario(Primitiva* prims, int num_prims, LuzPuntual* luces, int num_luces, 
                                                 Primitiva* prims_malla, int num_prims_malla,
                                                 TinyBVHD_GPU tiny_bvh_, const Primitiva* primitivas_tiny_bvh_, int num_prims_tiny_bvh_) 
    : primitivas(prims), num_primitivas(num_prims), luces(luces), num_luces(num_luces),
      primitivas_malla(prims_malla), num_primitivas_malla(num_prims_malla),
      nodos_bvh(nullptr), primitivas_bvh(nullptr), num_nodos_bvh(0),
      tiny_bvh(tiny_bvh_), primitivas_tiny_bvh(primitivas_tiny_bvh_), num_primitivas_tiny_bvh(num_prims_tiny_bvh_) {}


    // Método de intersección sin coordenadas baricéntricas
    __device__ bool intersecta(const Rayo& r, float& t, const Primitiva** objeto) const {
        bool hay_interseccion = false;
        float t_min = 1e9f;
        const Primitiva* objeto_mas_cercano = nullptr;
        
        for (int i = 0; i < num_primitivas; i++) {
            float t_temp;
            if (primitivas[i].intersecta(r, t_temp) && t_temp < t_min && t_temp > 1e-6f) {
                t_min = t_temp;
                objeto_mas_cercano = &primitivas[i];
                hay_interseccion = true;
            }
        }
        
        if (hay_interseccion) {
            t = t_min;
            *objeto = objeto_mas_cercano;
            return true;
        }
        return false;
    }

    // Método de intersección que devuelve coordenadas baricéntricas
    __device__ bool intersecta(const Rayo& r, float& t, const Primitiva** objeto, float& u, float& v) const {
        bool hay_interseccion = false;
        float t_min = 1e9f;
        const Primitiva* objeto_mas_cercano = nullptr;
        float u_temp = 0.0f, v_temp = 0.0f;
        
        for (int i = 0; i < num_primitivas; i++) {
            float t_temp;
            float u_local = 0.0f, v_local = 0.0f;
        
            if (primitivas[i].intersecta(r, t_temp, u_local, v_local) && t_temp < t_min && t_temp > 1e-6f) {
                t_min = t_temp;
                objeto_mas_cercano = &primitivas[i];
                u_temp = u_local;
                v_temp = v_local;
                hay_interseccion = true;
            }
        }
        
        if (hay_interseccion) {
            t = t_min;
            *objeto = objeto_mas_cercano;
            u = u_temp;
            v = v_temp;
            return true;
        }
        return false;
    }

    // Método auxiliar para calcular la normal correcta (suavizada o no)
    __device__ Vector3d calcularNormal(const Primitiva* primitiva, const Vector3d& punto, float u, float v) const {
        if (primitiva->tipo == TRIANGULO) {
            // Verificar si es un triángulo con normales suavizadas (de modelo cargado)
            // Los triángulos de modelos cargados tienen normales por vértice diferentes
            bool tiene_normales_suavizadas = 
                (primitiva->triangulo.v0_normal != primitiva->triangulo.v1_normal) ||
                (primitiva->triangulo.v0_normal != primitiva->triangulo.v2_normal);
            
            if (tiene_normales_suavizadas) {
                // Usar suavizado
                return primitiva->calcularNormalSuavizada(punto, u, v);
            } else {
                // Usar normal de cara
                return primitiva->calcularNormal(punto);
            }
        } else {
            // Otras primitivas: usar normal normal
            return primitiva->calcularNormal(punto);
        }
    }

    __device__ Color calcularLuzDirecta(const Vector3d& punto, const Vector3d& normal,
                                       const Primitiva* objeto_intersectado,
                                       int px, int py, double tiempo_acumulado, const Color& camino, double current_ior) const {
        Color L_directa(0, 0, 0);
        
        for (int i = 0; i < num_luces; i++) {
            const LuzPuntual* luz = &luces[i];
            Vector3d direccion_hacia_luz = luz->posicion - punto;
            float distancia = direccion_hacia_luz.norm();

            direccion_hacia_luz = direccion_hacia_luz.normalized();

            // Verificar que la luz está en el hemisferio correcto
            float cos_theta = normal.dot(direccion_hacia_luz);
            if (cos_theta <= 0.0f) continue;
            
            // Crear rayo sombra
            Rayo rayo_sombra(punto + direccion_hacia_luz * EPSILON, direccion_hacia_luz);
            float t_sombra;
            const Primitiva* objeto_sombra;
            
            // Verificar si el rayo sombra está bloqueado
            float u, v;
            //if (intersecta(rayo_sombra, t_sombra, &objeto_sombra)) { // Sin BVH
            if (intersectaPrimitivasBVH(rayo_sombra, t_sombra, &objeto_sombra, u, v)) {
                if (t_sombra < distancia) {
                    continue; // La luz está bloqueada
                }
            }
            
            Color brdf = objeto_intersectado->difuso / M_PI;
            
            distancia = max(distancia, 1.0f); // Evitar división por cero (solución luciérnagas)
            Color Li = luz->intensidad / (distancia * distancia);
            
            Color contrib = Li * brdf * cos_theta;
            L_directa = L_directa + contrib;
        }
        
        return L_directa;
    }

    // Método de intersección usando BVH
    __device__ bool intersectaNodoBVH(const Rayo& r, float& t, const Primitiva** objeto, float& u, float& v) const {
        float tMin, tMax;
        bool hit = false;
        float t_closest = 1e9f;
        const Primitiva* closest_obj = nullptr;
        float u_closest = 0.0f, v_closest = 0.0f;

        // Pila para recorrido iterativo
        int stack[64];
        int stack_ptr = 0;
        stack[stack_ptr++] = 0; // Empezar con el nodo raíz

        while (stack_ptr > 0) {
            int node_index = stack[--stack_ptr];
            const NodoBVH& node = nodos_bvh[node_index];

            if (node.caja.intersecta(r, tMin, tMax)) {
                if (node.esHoja) {
                    // Revisar todas las primitivas en la hoja
                    for (int i = node.inicio; i < node.fin; i++) {
                        float t_temp;
                        if (primitivas_bvh[i].intersecta(r, t_temp, u ,v) && t_temp < t_closest && t_temp > 1e-6f) {
                            t_closest = t_temp;
                            u_closest = u;
                            v_closest = v;
                            closest_obj = &primitivas_bvh[i];
                            hit = true;
                        }
                    }
                } else {
                    // Añadir hijos a la pila
                    if (node.izquierda != -1) {
                        stack[stack_ptr++] = node.izquierda;
                    }
                    if (node.derecha != -1) {
                        stack[stack_ptr++] = node.derecha;
                    }
                }
            }
        }

        // Si hubo intersección, actualizar resultados
        if (hit) {
            t = t_closest;
            *objeto = closest_obj;
            u = u_closest;
            v = v_closest;
            return true;
        }
        return false;
    }

    // Método principal de intersección con BVH
    __device__ bool intersectaBVH(const Rayo& r, float& t, const Primitiva** objeto, float& u, float& v) const {
        if(nodos_bvh == nullptr || primitivas_bvh == nullptr || num_nodos_bvh == 0) {
            return false;
        }
        return intersectaNodoBVH(r, t, objeto, u, v);
    }

    // Método que utiliza intersección normal para primitivas manuales y BVH para modelo (malla de triángulos)
    __device__ bool intersectaPrimitivasBVH(const Rayo& r, float& t, const Primitiva** objeto, float& u, float& v) const {
        bool hay_interseccion = false;
        float t_min = 1e9f;
        const Primitiva* objeto_mas_cercano = nullptr;
        float u_temp = 0.0f, v_temp = 0.0f;
        
        // Primero, revisar primitivas manuales
        for (int i = 0; i < num_primitivas; i++) {
            float t_temp;
            float u_local = 0.0f, v_local = 0.0f;
        
            if (primitivas[i].intersecta(r, t_temp, u_local, v_local) && t_temp < t_min && t_temp > 1e-6f) {
                t_min = t_temp;
                objeto_mas_cercano = &primitivas[i];
                u_temp = u_local;
                v_temp = v_local;
                hay_interseccion = true;
            }
        }

        // Luego, usar BVH para modelo
        float t_bvh;
        float u_bvh = 0.0f, v_bvh = 0.0f;
        const Primitiva* objeto_bvh;
        if (intersectaBVH(r, t_bvh, &objeto_bvh, u_bvh, v_bvh)) {
            if (t_bvh < t_min && t_bvh > 1e-6f) {
                t_min = t_bvh;
                u_temp = u_bvh;
                v_temp = v_bvh;
                objeto_mas_cercano = objeto_bvh;
                hay_interseccion = true;
            }
        }
        
        if (hay_interseccion) {
            t = t_min;
            *objeto = objeto_mas_cercano;
            u = u_temp;
            v = v_temp;
            return true;
        }
        return false;
    }

    //------------------------------
    // INTERSECCIÓN CON BVH DE tiny_bvh
    //------------------------------

    __device__ float intersectaAABB(const Rayo& rayo, const float3& bmin, const float3& bmax) const {
        float tx1 = (bmin.x - rayo.origen().x) * (1.0f / rayo.direccion().x);
        float tx2 = (bmax.x - rayo.origen().x) * (1.0f / rayo.direccion().x);
        float tmin = min(tx1, tx2);
        float tmax = max(tx1, tx2);

        float ty1 = (bmin.y - rayo.origen().y) * (1.0f / rayo.direccion().y);
        float ty2 = (bmax.y - rayo.origen().y) * (1.0f / rayo.direccion().y);
        tmin = max(tmin, min(ty1, ty2));
        tmax = min(tmax, max(ty1, ty2));

        float tz1 = (bmin.z - rayo.origen().z) * (1.0f / rayo.direccion().z);
        float tz2 = (bmax.z - rayo.origen().z) * (1.0f / rayo.direccion().z);
        tmin = max(tmin, min(tz1, tz2));
        tmax = min(tmax, max(tz1, tz2));

        if (tmax >= tmin && tmax > 0) return tmin;
        return 1e9f; // No hay intersección
    }

    __device__ bool intersectaPrimitivasBVH_tiny(const Rayo& r, float& t, const Primitiva** objeto, float& u, float& v) const {
        bool hay_interseccion = false;
        float t_min = 1e9f;
        const Primitiva* objeto_mas_cercano = nullptr;
        float u_temp = 0.0f, v_temp = 0.0f;
        
        // Primero, revisar primitivas manuales
        for (int i = 0; i < num_primitivas; i++) {
            float t_temp;
            float u_local = 0.0f, v_local = 0.0f;
        
            if (primitivas[i].intersecta(r, t_temp, u_local, v_local) && t_temp < t_min && t_temp > 1e-6f) {
                t_min = t_temp;
                objeto_mas_cercano = &primitivas[i];
                u_temp = u_local;
                v_temp = v_local;
                hay_interseccion = true;
            }
        }

        // BVH
        if (tiny_bvh.d_nodos != nullptr && tiny_bvh.d_tris != nullptr && num_primitivas_tiny_bvh > 0) {
            // Obtener origen y dirección
            float3 O = make_float3((float)r.origen().x, (float)r.origen().y, (float)r.origen().z);
            float3 D = make_float3((float)r.direccion().x, (float)r.direccion().y, (float)r.direccion().z);
            float3 rD = make_float3(1.0f / D.x, 1.0f / D.y, 1.0f / D.z);

            // Recorrido CWBVH
            float4 hit = traverse_cwbvh(tiny_bvh.d_nodos, tiny_bvh.d_tris, O, D, rD, t_min, nullptr);

            // Comprobar si hubo colisión leyendo el ID del triángulo primitivo
            uint32_t hitAddr = __float_as_uint(hit.w);
            // El ID devuelto es (uint32_t)-1 si no chocó con nada o si el t devuelto es mayor o igual a t_min.
            if (hitAddr != (uint32_t)-1 && hit.x < t_min && hit.x > EPSILON) {
                t_min = hit.x;       // Nuevo tiempo más cercano
                u_temp = hit.y;      // Coordenada baricéntrica U
                v_temp = hit.z;      // Coordenada baricéntrica V
                objeto_mas_cercano = &primitivas_tiny_bvh[hitAddr];
                hay_interseccion = true;
            }
        }

        if (hay_interseccion) {
            t = t_min;
            *objeto = objeto_mas_cercano;
            u = u_temp;
            v = v_temp;
            return true;
        }
        return false;
    }

};

#endif