// ################
// Autores: 
// Mir Ramos, Rubén 869039
// Lopez Torralba, Alejandro 845154
//
// main.cu
// ################

#ifdef __CUDACC__
#ifndef _AMX_TILE_INCLUDED
#define _AMX_TILE_INCLUDED
#endif
#ifndef __AMXTILEINTRIN_H_INCLUDED
#define __AMXTILEINTRIN_H_INCLUDED
#endif
#ifndef _AMXINT8INTRIN_H_INCLUDED
#define _AMXINT8INTRIN_H_INCLUDED
#endif
#ifndef _AMXBF16INTRIN_H_INCLUDED
#define _AMXBF16INTRIN_H_INCLUDED
#endif
#endif

#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <thread>
#include <cstring>
#include <iomanip>
#include <atomic>
#include <memory>
#include <cuda_runtime.h>
#include "imagen.h"
#include "imagen_utils.h"
#include "imagen_gpu.h"
#include "camara.h"
#include "primitiva.h"
#include "render.h"
#include "escenario.h"
#include "color.h"
#include "vector3d.h"
#include "cargarModelo.h"
#include "bvh.h"
#include "mlp.h"
#include "mlp_types.h"
#include "gridsearch.h"
#include "visualizador.h"
#include "tinybvh_wrapper.h"

using namespace std;

// Función kernel para renderizar la escena en la GPU
__global__ void kernelRender(const Camara* camara, const Primitiva* primitivas, int num_primitivas, const LuzPuntual* luces, int num_luces,
                            const Primitiva* primitivas_malla, int num_primitivas_malla, int ancho_img, int alto_img, int samples_per_pixel, 
                            const NodoBVH* nodos_bvh, const Primitiva* primitivas_bvh, int num_nodos_bvh, 
                            ImagenGPU imagen_directa,
                            curandState* rand_states, 
                            // Datos para ENTRENAMIENTO
                            unsigned int* dev_counter, RegistroEntrenamiento* buffer_registros, unsigned int* counter_train, int max_cap_train,
                            // Datos para INFERENCIA
                            DatosMLP* buffer_inference, Color* buffer_throughput, bool usar_red_inferencia, bool entrenar_red) { 

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x < ancho_img && y < alto_img) {
        Escenario escena(const_cast<Primitiva*>(primitivas), num_primitivas, const_cast<LuzPuntual*>(luces), num_luces,
                        const_cast<Primitiva*>(primitivas_malla), num_primitivas_malla, nodos_bvh, primitivas_bvh, num_nodos_bvh);
        Render renderer(0.9, samples_per_pixel);
                    
        renderer.renderizar(*camara, escena, ancho_img, alto_img, x, y, 
                            imagen_directa, rand_states, 
                            dev_counter, buffer_registros, counter_train, max_cap_train,
                            buffer_inference, buffer_throughput, usar_red_inferencia, entrenar_red);
    }
}

__global__ void kernelRender_tiny(const Camara* camara, const Primitiva* primitivas, int num_primitivas, const LuzPuntual* luces, int num_luces,
                            const Primitiva* primitivas_malla, int num_primitivas_malla, int ancho_img, int alto_img, int samples_per_pixel, 
                            TinyBVHD_GPU nodos_bvh, const Primitiva* primitivas_bvh, int num_nodos_bvh, 
                            ImagenGPU imagen_directa,
                            curandState* rand_states, 
                            // Datos para ENTRENAMIENTO
                            unsigned int* dev_counter, RegistroEntrenamiento* buffer_registros, unsigned int* counter_train, int max_cap_train,
                            // Datos para INFERENCIA
                            DatosMLP* buffer_inference, Color* buffer_throughput, bool usar_red_inferencia, bool entrenar_red) { 

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x < ancho_img && y < alto_img) {
        Escenario escena(const_cast<Primitiva*>(primitivas), num_primitivas, const_cast<LuzPuntual*>(luces), num_luces,
                        const_cast<Primitiva*>(primitivas_malla), num_primitivas_malla, nodos_bvh, primitivas_bvh, num_nodos_bvh);
        Render renderer(0.9, samples_per_pixel);
                    
        renderer.renderizar(*camara, escena, ancho_img, alto_img, x, y, 
                            imagen_directa, rand_states, 
                            dev_counter, buffer_registros, counter_train, max_cap_train,
                            buffer_inference, buffer_throughput, usar_red_inferencia, entrenar_red);
    }
}

// Combina la imagen física (PT) con la predicción neuronal
// Pixel = LuzPT + (Throughput * Prediccion_Red)
__global__ void kernelComposite(Color* img_pt, Color* img_prediccion, Color* throughput_map, 
                                int ancho, int alto) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * ancho + x;

    if(x < ancho && y < alto) {
        Color final = img_pt[idx]; // L_directa + Emisión (Path Tracing limpio)
        Color th = throughput_map[idx];
        Color indirecta_neuronal = img_prediccion[idx];

        // Solo sumamos la red si el rayo chocó con una superficie y consultó el MLP
        if (th.r > 0 || th.g > 0 || th.b > 0) {
            final = final + (th * indirecta_neuronal);
        } 
        
        // Guardamos el resultado combinado
        img_pt[idx] = final;
    }
}

__global__ void setupRandomStates(curandState* states, unsigned long seed, int total_threads) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < total_threads) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// Combinar escena manual con modelo cargado
void combinarEscenas(Primitiva** d_primitivas, int* num_primitivas, LuzPuntual** d_luces, int* num_luces) {
    cout << "\n=================================================" << endl;
    cout << "====             CREACIÓN ESCENA             ====" << endl;
    cout << "=================================================" << endl;


    cout << "Combinando escena base con modelo cargado..." << endl;

    vector<Primitiva> host_primitivas;
    vector<LuzPuntual> host_luces;

    
    LuzPuntual luz_techo = {};
    luz_techo.posicion = Vector3d(0, -0.23, -2.0);
    luz_techo.intensidad = Color(70, 70, 70);
    host_luces.push_back(luz_techo);

    // Pared derecha (verde) - MÁS SEPARADA
    Primitiva pared_der = {};
    pared_der.tipo = PLANO;
    pared_der.plano.normal = Vector3d(-1, 0, 0);
    pared_der.plano.distancia = -4.0f;  // Más separada
    pared_der.emision = Color(0,0,0);
    pared_der.difuso = Color(0.8,0.4,0.1);
    pared_der.especular = Color(0,0,0);
    pared_der.transmision = Color(0,0,0);
    pared_der.indice_refraccion = 1.0f;
    host_primitivas.push_back(pared_der);

    // Techo (gris claro)
    Primitiva techo = {};
    techo.tipo = PLANO;
    techo.plano.normal = Vector3d(0, -0.25, 0);
    techo.plano.distancia = -3.0f;
    //techo.emision = Color(5.0f, 5.0f, 5.0f);
    techo.emision = Color(0,0,0);
    techo.difuso = Color(0.9,0.9,0.9);
    techo.especular = Color(0,0,0);
    techo.transmision = Color(0,0,0);
    techo.indice_refraccion = 1.0f;
    host_primitivas.push_back(techo);

    // Pared fondo (gris) - MÁS ATRÁS
    Primitiva pared_fondo = {};
    pared_fondo.tipo = PLANO;
    pared_fondo.plano.normal = Vector3d(0, 0, 1);
    pared_fondo.plano.distancia = -10.0f;
    pared_fondo.emision = Color(0,0,0);
    pared_fondo.difuso = Color(0.9,0.9,0.9);
    pared_fondo.especular = Color(0,0,0);
    pared_fondo.transmision = Color(0,0,0);
    pared_fondo.indice_refraccion = 1.0f;
    host_primitivas.push_back(pared_fondo);

    // Esfera golpeando pared izquierda
    Primitiva esfera_izq = {};
    esfera_izq.tipo = ESFERA;
    esfera_izq.esfera.centro = Vector3d(-4.0f, -1.1f, -5.2f);
    esfera_izq.esfera.radio = 0.7f;
    esfera_izq.emision = Color(0,0,0);
    esfera_izq.difuso = Color(0.1,0.1,0.1);
    esfera_izq.especular = Color(0.3f, 0.3f, 0.3f);
    esfera_izq.transmision = Color(0,0,0);
    esfera_izq.indice_refraccion = 1.0f;
    host_primitivas.push_back(esfera_izq);

    // Esfera golpeando esfera rota
    Primitiva esfera_rota = {};
    esfera_rota.tipo = ESFERA;
    esfera_rota.esfera.centro = Vector3d(0.8f, -0.5f, -1.8f);
    esfera_rota.esfera.radio = 0.4f;
    esfera_rota.emision = Color(0,0,0);
    esfera_rota.difuso = Color(0.0f, 0.0f, 0.0f);
    esfera_rota.especular = Color(0.04f, 0.04f, 0.04f);
    esfera_rota.transmision = Color(0.9f, 0.1f, 0.1f); 
    esfera_rota.indice_refraccion = 3.0f; 
    host_primitivas.push_back(esfera_rota);

    // Esfera golpeando busto
    Primitiva esfera_busto = {};
    esfera_busto.tipo = ESFERA;
    esfera_busto.esfera.centro = Vector3d(0.1f, -1.1f, -2.3f);
    esfera_busto.esfera.radio = 0.15f;
    esfera_busto.emision = Color(0,0,0);
    esfera_busto.difuso = Color(0.0f, 0.0f, 0.0f); 
    esfera_busto.especular = Color(0.04f, 0.04f, 0.04f); 
    esfera_busto.transmision = Color(0.1f, 0.8f, 0.4f); 
    esfera_busto.indice_refraccion = 1.5f;
    host_primitivas.push_back(esfera_busto);

   
    // Copiar las primitivas manuales a la GPU
    cudaMalloc(d_primitivas, host_primitivas.size() * sizeof(Primitiva));
    cudaMemcpy(*d_primitivas, host_primitivas.data(), 
               host_primitivas.size() * sizeof(Primitiva), cudaMemcpyHostToDevice);
    
    *num_primitivas = host_primitivas.size();
    
    // Copiar las luces a la GPU
    cudaMalloc(d_luces, host_luces.size() * sizeof(LuzPuntual));
    cudaMemcpy(*d_luces, host_luces.data(), 
               host_luces.size() * sizeof(LuzPuntual), cudaMemcpyHostToDevice);
    
    *num_luces = host_luces.size();

    cout << "Escena combinada: \n" << 
            "   -Manual: " <<
            host_primitivas.size() << " primitivas, " << host_luces.size() << " luces puntuales\n";
}

// Crear escena manualmente
void inicializarEscena(Primitiva** d_primitivas, int* num_primitivas, LuzPuntual** d_luces, int* num_luces) {
    cout << "\n=================================================" << endl;
    cout << "====             CREACIÓN ESCENA             ====" << endl;
    cout << "=================================================" << endl;
    
    cout << "Usando escena base..." << endl;
    
    vector<Primitiva> host_primitivas;
    vector<LuzPuntual> host_luces;

    // Luz puntual en el techo
    LuzPuntual luz_techo = {};
    luz_techo.posicion = Vector3d(0, 2.9, -4.0);
    luz_techo.intensidad = Color(100, 100, 100);
    host_luces.push_back(luz_techo);
    
    // Pared izquierda (cian)
    Primitiva pared_izq = {};
    pared_izq.tipo = PLANO;
    pared_izq.plano.normal = Vector3d(1, 0, 0);
    pared_izq.plano.distancia = -3.0f;
    pared_izq.emision = Color(0,0,0);
    pared_izq.difuso = Color(0.1f, 0.8f, 0.9f);
    pared_izq.especular = Color(0,0,0);
    pared_izq.transmision = Color(0,0,0);
    pared_izq.indice_refraccion = 1.0f;
    host_primitivas.push_back(pared_izq);
    
    // Pared derecha (magenta)
    Primitiva pared_der = {};
    pared_der.tipo = PLANO;
    pared_der.plano.normal = Vector3d(-1, 0, 0);
    pared_der.plano.distancia = -3.0f;
    pared_der.emision = Color(0,0,0);
    pared_der.difuso = Color(0.9f, 0.1f, 0.6f);
    pared_der.especular = Color(0,0,0);
    pared_der.transmision = Color(0,0,0);
    pared_der.indice_refraccion = 1.0f;
    host_primitivas.push_back(pared_der);
    
    // Suelo
    Primitiva suelo = {};
    suelo.tipo = PLANO;
    suelo.plano.normal = Vector3d(0, 1, 0);
    suelo.plano.distancia = -3.0f;
    suelo.emision = Color(0,0,0);
    suelo.difuso = Color(0.8f, 0.8f, 0.8f);
    suelo.especular = Color(0,0,0);
    suelo.transmision = Color(0,0,0);
    suelo.indice_refraccion = 1.0f;
    host_primitivas.push_back(suelo);
    
    // Techo
    Primitiva techo = {};
    techo.tipo = PLANO;
    techo.plano.normal = Vector3d(0, -1, 0);
    techo.plano.distancia = -3.0f;
    techo.emision = Color(0, 0, 0);
    techo.difuso = Color(0.8f, 0.8f, 0.8f);
    techo.especular = Color(0,0,0);
    techo.transmision = Color(0,0,0);
    techo.indice_refraccion = 1.0f;
    host_primitivas.push_back(techo);
    
    // Pared fondo
    Primitiva pared_fondo = {};
    pared_fondo.tipo = PLANO;
    pared_fondo.plano.normal = Vector3d(0, 0, 1);
    pared_fondo.plano.distancia = -8.0f;
    pared_fondo.emision = Color(0,0,0);
    pared_fondo.difuso = Color(0.8f, 0.8f, 0.8f);
    pared_fondo.especular = Color(0,0,0);
    pared_fondo.transmision = Color(0,0,0);
    pared_fondo.indice_refraccion = 1.0f;
    host_primitivas.push_back(pared_fondo);

    // Esfera cristal Grande (Izquierda - Muy Cerca)
    Primitiva s1 = {};
    s1.tipo = ESFERA;
    s1.esfera.centro = Vector3d(-1.5f, -1.0f, -3.8f); 
    s1.esfera.radio = 1.1f;
    s1.emision = Color(0,0,0);
    s1.difuso = Color(0,0,0);
    s1.especular = Color(0.04f, 0.04f, 0.04f); 
    s1.transmision = Color(0.95f, 0.95f, 0.95f); 
    s1.indice_refraccion = 1.5f; 
    host_primitivas.push_back(s1);

    
    // 2. Esfera plástico Rojo (Derecha Arriba - Flotando)
    Primitiva s2 = {};
    s2.tipo = ESFERA;
    s2.esfera.centro = Vector3d(1.8f, 1.5f, -4.5f);
    s2.esfera.radio = 0.8f;
    s2.emision = Color(0,0,0);
    s2.difuso = Color(0.9f, 0.1f, 0.1f);
    s2.especular = Color(0.3f, 0.3f, 0.3f);
    s2.transmision = Color(0,0,0);
    s2.indice_refraccion = 1.0f;
    host_primitivas.push_back(s2);

    // Esfera difusa Blanca (Centro Abajo - Cerca)
    Primitiva s3 = {};
    s3.tipo = ESFERA;
    s3.esfera.centro = Vector3d(0.2f, -2.2f, -4.0f);
    s3.esfera.radio = 0.7f;
    s3.emision = Color(0,0,0);
    s3.difuso = Color(0.3f, 0.5f, 0.2f);
    s3.especular = Color(0,0,0);
    s3.transmision = Color(0,0,0);
    s3.indice_refraccion = 1.0f;
    host_primitivas.push_back(s3);

    // Esfera plástico Verde (Izquierda Arriba - Medio)
    Primitiva s4 = {};
    s4.tipo = ESFERA;
    s4.esfera.centro = Vector3d(-1.2f, 1.8f, -5.0f); 
    s4.esfera.radio = 0.6f;
    s4.emision = Color(0,0,0);
    s4.difuso = Color(0.2f, 0.8f, 0.2f);
    s4.especular = Color(0.2f, 0.2f, 0.2f);
    s4.transmision = Color(0,0,0);
    s4.indice_refraccion = 1.0f;
    host_primitivas.push_back(s4);

    // Esfera cristal Pequeña (Derecha Abajo - Medio)
    Primitiva s5 = {};
    s5.tipo = ESFERA;
    s5.esfera.centro = Vector3d(2.0f, -2.0f, -4.2f); 
    s5.esfera.radio = 0.5f;
    s5.emision = Color(0,0,0);
    s5.difuso = Color(0,0,0);
    s5.especular = Color(0.04f, 0.04f, 0.04f); 
    s5.transmision = Color(0.95f, 0.95f, 0.95f); 
    s5.indice_refraccion = 1.5f;
    host_primitivas.push_back(s5);
    

    // Esfera plástico Azul (Centro - Flotando alto)
    Primitiva s6 = {};
    s6.tipo = ESFERA;
    s6.esfera.centro = Vector3d(-0.5f, 0.5f, -4.8f); 
    s6.esfera.radio = 0.6f;
    s6.emision = Color(0,0,0);
    s6.difuso = Color(0.1f, 0.1f, 0.9f);
    s6.especular = Color(0.4f, 0.4f, 0.4f);
    s6.transmision = Color(0,0,0);
    s6.indice_refraccion = 1.0f;
    host_primitivas.push_back(s6);
    
    
    // Esfera difusa Amarilla (Derecha - Fondo medio)
    Primitiva s7 = {};
    s7.tipo = ESFERA;
    s7.esfera.centro = Vector3d(1.0f, -0.5f, -5.5f); 
    s7.esfera.radio = 0.9f;
    s7.emision = Color(0,0,0);
    s7.difuso = Color(0.9f, 0.8f, 0.2f);
    s7.especular = Color(0,0,0);
    s7.transmision = Color(0,0,0);
    s7.indice_refraccion = 1.0f;
    host_primitivas.push_back(s7);
    

    // Copiar las primitivas a la GPU
    cudaMalloc(d_primitivas, host_primitivas.size() * sizeof(Primitiva));
    cudaMemcpy(*d_primitivas, host_primitivas.data(), host_primitivas.size() * sizeof(Primitiva), 
               cudaMemcpyHostToDevice);
    
    *num_primitivas = host_primitivas.size();
    
    // Copiar las luces a la GPU
    cudaMalloc(d_luces, host_luces.size() * sizeof(LuzPuntual));
    cudaMemcpy(*d_luces, host_luces.data(), host_luces.size() * sizeof(LuzPuntual), 
               cudaMemcpyHostToDevice);
    
    *num_luces = host_luces.size();

    cout << "Escena básica: \n" << 
            "   -Manual: " <<
            host_primitivas.size() << " primitivas, " << host_luces.size() << " luces puntuales" << endl;
}

// Inicializar cámara
__global__ void inicializarCamara(Camara* d_camara, int ancho_imagen, int alto_imagen) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float distancia_focal = 1.0f;
        float fov_grados = 45.0f;
        float fov_radianes = fov_grados * M_PI / 180.0f;
        float alto_plano = 2.0f * distancia_focal * tan(fov_radianes / 2.0f);
        float ancho_plano = alto_plano * (static_cast<float>(ancho_imagen) / alto_imagen);
        
        // Usar el constructor de CamaraGPU
        //Vector3d posicion(0, 0, 5);
        //Vector3d frente(0, 0, -1);
        //Vector3d arriba(0, 1, 0);

        //Vector3d posicion(0, -0.75, 4.25);
        //Vector3d frente(0, 0, -1);
        //Vector3d arriba(0, 1, 0);

        Vector3d posicion(2.6f, 0.2f, 4.25f);
        Vector3d frente(-0.6f, -0.25f, -1.0f);
        Vector3d arriba(0, 1, 0);

        *d_camara = Camara(posicion, frente, arriba, ancho_plano, alto_plano, distancia_focal);
    }
}

// Inicializar estados aleatorios para cada thread
curandState* inicializarEstadosAleatorios(int ancho, int alto) {
    // Calcular el grid que usará el kernel de renderizado
    dim3 blockSize(16, 16);
    dim3 gridSize((ancho + blockSize.x - 1) / blockSize.x, 
                  (alto + blockSize.y - 1) / blockSize.y);
    
    int total_threads = gridSize.x * gridSize.y * blockSize.x * blockSize.y;
    curandState* d_states;
    cudaMalloc(&d_states, total_threads * sizeof(curandState));
    
    int block_size = 256;
    int num_blocks = (total_threads + block_size - 1) / block_size;
    
    setupRandomStates<<<num_blocks, block_size>>>(d_states, time(NULL), total_threads);
    cudaDeviceSynchronize();
    
    return d_states;
}

// Liberar memoria GPU
void limpiarGPU(Primitiva* d_primitivas, LuzPuntual* d_luces, curandState* rand_states) {
    if (d_primitivas) {
        cudaFree(d_primitivas);
    }
    if (d_luces) {
        cudaFree(d_luces);
    }
    if (rand_states) {
        cudaFree(rand_states);
    }
}

// Kernel que se ejecuta después de la inferencia del Tail y antes del entrenamiento
// para calcular los targets finales combinando el sufijo real con la predicción del Tail
__global__ void kernelCalcularTargets(
    RegistroEntrenamiento* buffer_registros, 
    Color* buffer_prediccion_tail,
    DatosMLP* buffer_training_final,
    int num_elementos
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elementos) return;

    RegistroEntrenamiento reg = buffer_registros[idx];
    
    if (!reg.valido) {
        buffer_training_final[idx].color = Color(0,0,0);
        return;
    }

    Color iluminacion_tail = buffer_prediccion_tail[idx];
    
    // Radiancia total recolectada por el sufijo
    Color radiance_total = reg.luz_acumulada_sufijo + (reg.throughput_sufijo * iluminacion_tail);
    
    // Denominador: throughput_at_train_vertex * reflectancia_train
    Color denominador = reg.factor_normalizacion;
    
    // Estabilización numérica de la división
    float safe_eps = 1e-8f; 
    
    Color target_final;
    target_final.r = radiance_total.r / fmaxf(denominador.r, safe_eps);
    target_final.g = radiance_total.g / fmaxf(denominador.g, safe_eps);
    target_final.b = radiance_total.b / fmaxf(denominador.b, safe_eps);

    // Clamping del target final
    // Evita que un solo pixel con valor infinito destruya los pesos del HashGrid
    float max_radiance = 100000.0f;
    target_final.r = fminf(target_final.r, max_radiance);
    target_final.g = fminf(target_final.g, max_radiance);
    target_final.b = fminf(target_final.b, max_radiance);

    // Protección extrema contra NaNs (Not a Number) e Infinitos
    if (isnan(target_final.r) || isinf(target_final.r)) target_final.r = 0.0f;
    if (isnan(target_final.g) || isinf(target_final.g)) target_final.g = 0.0f;
    if (isnan(target_final.b) || isinf(target_final.b)) target_final.b = 0.0f;

    DatosMLP d;
    d.posicion = reg.head.posicion;
    d.direccion = reg.head.direccion;
    d.normal = reg.head.normal;
    d.difuso = reg.head.difuso;
    d.especular = reg.head.especular;
    d.color = target_final;

    buffer_training_final[idx] = d;
}

// Kernel para extraer la geometría del Tail y prepararla para la red
__global__ void kernelPrepararInferenciaTail(
    const RegistroEntrenamiento* __restrict__ source_registros,
    DatosMLP* __restrict__ dest_inference_inputs,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    const RegistroEntrenamiento& reg = source_registros[idx];
    
    DatosMLP d;
    d.posicion = Vector3d(0,0,0);
    d.direccion = Vector3d(0,0,0); 
    d.normal = Vector3d(0,0,0);
    d.difuso = Color(0,0,0);
    d.especular = Color(0,0,0);
    d.color = Color(0,0,0);

    if (reg.valido) {
        d.posicion = reg.tail.posicion;
        d.direccion = reg.tail.direccion;
        d.normal = reg.tail.normal;
        d.difuso = reg.tail.difuso;
        d.especular = reg.tail.especular;
    }

    dest_inference_inputs[idx] = d;
}

// Kernel para desordenar los datos de entrenamiento usando curand
__global__ void kernelShuffle(
    const DatosMLP* __restrict__ input_data,
    DatosMLP* __restrict__ shuffled_data,
    unsigned int num_elements,
    curandState* __restrict__ rand_states
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    // Obtener estado aleatorio para este thread
    curandState localState = rand_states[idx];
    
    // Generar índice destino aleatorio
    unsigned int dest_idx = (unsigned int)(curand_uniform(&localState) * num_elements);
    dest_idx = min(dest_idx, num_elements - 1);

    // Guardar estado actualizado
    rand_states[idx] = localState;

    // Escribimos el dato en su nueva posición desordenada
    shuffled_data[dest_idx] = input_data[idx];
}

bool renderizarModoReconstruccion(
    int ancho_imagen,
    int alto_imagen,
    int samples_per_pixel,
    const string& nombre_archivo,
    const string& ruta_modelo,
    Camara* d_camara,
    const Primitiva* d_primitivas,
    int num_primitivas,
    const LuzPuntual* d_luces,
    int num_luces,
    const Primitiva* d_primitivas_malla,
    int num_primitivas_malla,
    const TinyBVH& bvh_modelo_tiny,
    curandState* rand_states,
    const SceneBounds& scene_bounds
) {
    int num_pixels = ancho_imagen * alto_imagen;

    cout << "\n================================================" << endl;
    cout << "====         MODO RECONSTRUCCIÓN           ====" << endl;
    cout << "================================================" << endl;

    tcnn::json config_ganadora;
    uint32_t n_in = 15;
    uint32_t n_out = 3;
    uint32_t batch_size_mlp = 16384;
    auto mlp = std::make_unique<ColorMLP>(n_in, n_out, batch_size_mlp, config_ganadora);
    mlp->setBounds(scene_bounds.min, scene_bounds.max);

    if (!mlp->load_model(ruta_modelo)) {
        cerr << "[MLP] No se pudo cargar el modelo neuronal." << endl;
        return false;
    }

    Color* d_imagen_data = nullptr;
    DatosMLP* d_buffer_inference_inputs = nullptr;
    Color* d_buffer_radiance_predicha = nullptr;
    Color* d_buffer_throughput = nullptr;

    cudaMalloc(&d_imagen_data, num_pixels * sizeof(Color));
    cudaMalloc(&d_buffer_inference_inputs, num_pixels * sizeof(DatosMLP));
    cudaMalloc(&d_buffer_radiance_predicha, num_pixels * sizeof(Color));
    cudaMalloc(&d_buffer_throughput, num_pixels * sizeof(Color));
    cudaMemset(d_imagen_data, 0, num_pixels * sizeof(Color));

    ImagenGPU imagen(ancho_imagen, alto_imagen, d_imagen_data);
    vector<Color> acumulador_cpu(num_pixels, Color(0,0,0));
    vector<Color> frame_cpu(num_pixels);

    dim3 blockSize(16, 16);
    dim3 gridSize((ancho_imagen + blockSize.x - 1) / blockSize.x,
                  (alto_imagen + blockSize.y - 1) / blockSize.y);

    auto inicio = chrono::high_resolution_clock::now();
    for (int s = 0; s < samples_per_pixel; ++s) {
        cudaMemset(d_imagen_data, 0, num_pixels * sizeof(Color));
        cudaMemset(d_buffer_inference_inputs, 0, num_pixels * sizeof(DatosMLP));
        cudaMemset(d_buffer_throughput, 0, num_pixels * sizeof(Color));

        kernelRender_tiny<<<gridSize, blockSize>>>(
            d_camara, d_primitivas, num_primitivas, d_luces, num_luces,
            d_primitivas_malla, num_primitivas_malla, ancho_imagen, alto_imagen, 1,
            bvh_modelo_tiny.obtenerDatosGPU(), bvh_modelo_tiny.getPrimitivasGPU(), bvh_modelo_tiny.getNumPrimitivas(),
            imagen, rand_states,
            nullptr, nullptr, nullptr, 0,
            d_buffer_inference_inputs, d_buffer_throughput, 
            true, false
        );

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            cerr << "Error en kernelRender (reconstrucción): " << cudaGetErrorString(err) << endl;
            cudaFree(d_imagen_data);
            cudaFree(d_buffer_inference_inputs);
            cudaFree(d_buffer_radiance_predicha);
            cudaFree(d_buffer_throughput);
            return false;
        }

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            cerr << "Error ejecutando kernelRender (reconstrucción): " << cudaGetErrorString(err) << endl;
            cudaFree(d_imagen_data);
            cudaFree(d_buffer_inference_inputs);
            cudaFree(d_buffer_radiance_predicha);
            cudaFree(d_buffer_throughput);
            return false;
        }

        mlp->inference(d_buffer_inference_inputs, d_buffer_radiance_predicha, num_pixels);
        kernelComposite<<<gridSize, blockSize>>>(d_imagen_data, d_buffer_radiance_predicha, d_buffer_throughput, ancho_imagen, alto_imagen);
        cudaDeviceSynchronize();

        cudaMemcpy(frame_cpu.data(), d_imagen_data, num_pixels * sizeof(Color), cudaMemcpyDeviceToHost);
        for (int i = 0; i < num_pixels; ++i) {
            acumulador_cpu[i] = acumulador_cpu[i] + frame_cpu[i];
        }

        cout << "\rSample: " << (s + 1) << "/" << samples_per_pixel << flush;
    }
    cout << endl;

    auto fin = chrono::high_resolution_clock::now();
    auto duracion = chrono::duration_cast<chrono::milliseconds>(fin - inicio);

    cout << "\n=================================================" << endl;
    cout << "====       RECONSTRUCCIÓN COMPLETADA       ====" << endl;
    cout << "=================================================" << endl;
    cout << "Tiempo de renderizado GPU: " << duracion.count() << " ms" << endl;

    Imagen imagenCPU(ancho_imagen, alto_imagen);
    float inv_samples = 1.0f / (float)samples_per_pixel;
    for (int i = 0; i < num_pixels; ++i) {
        imagenCPU.datos[i] = acumulador_cpu[i] * inv_samples;
    }

    Imagen imagen_final = imagenCPU.filmic();
    if (guardarPNG(imagen_final, nombre_archivo.c_str())) {
        cout << "Imagen guardada como: " << nombre_archivo << endl;
    } else {
        cout << "Error al guardar la imagen" << endl;
    }

    cudaFree(d_imagen_data);
    cudaFree(d_buffer_inference_inputs);
    cudaFree(d_buffer_radiance_predicha);
    cudaFree(d_buffer_throughput);
    return true;
}

int main() {
    // Habilitar mapeo de memoria host
    cudaSetDeviceFlags(cudaDeviceMapHost);

    try {
        // Selección de modo
        cout << "================================================" << endl;
        cout << "====            SELECCIONAR MODO            ====" << endl;
        cout << "================================================" << endl;
        cout << "1. Modo normal (entrenamiento + uso)" << endl;
        cout << "2. Modo reconstruccion (solo inferencia)" << endl;
        cout << "Selecciona una opcion (1 o 2): ";

        int modo = 1;
        cin >> modo;
        if (modo != 1 && modo != 2) {
            cerr << "Opcion invalida." << endl;
            return 1;
        }

        // Configuración básica
        cout << "================================================" << endl;
        cout << "====          CONFIGURACIÓN IMAGEN          ====" << endl;
        cout << "================================================" << endl;
        
        int ancho_imagen;
        int alto_imagen;
        int samples_per_pixel;
        bool cargar_modelo = false;
        string ruta_modelo_entrenado = "mlp_model.json";

        cout << "Introduce el ancho de la imagen (px): ";
        cin >> ancho_imagen;
        cout << "Introduce el alto de la imagen (px): ";
        cin >> alto_imagen;
        cout << "Introduce el número de muestras por píxel (samples per pixel): ";
        cin >> samples_per_pixel;
        cout << "¿Deseas cargar un modelo 3D externo? (1 = Sí, 0 = No): ";
        cin >> cargar_modelo;

        int num_pixels = ancho_imagen * alto_imagen;

        string nombre_archivo;
        cout << "Introduce el nombre del archivo de salida: ";
        cin >> nombre_archivo;

        // Variable para verificar errores CUDA
        cudaError_t err;
        
        Primitiva* d_primitivas = nullptr;
        int num_primitivas = 0;
        LuzPuntual* d_luces = nullptr;
        int num_luces = 0;

        Primitiva* d_primitivas_malla = nullptr;
        int num_primitivas_malla = 0;

        ArbolBVH bvh_modelo;
        TinyBVH bvh_modelo_tiny;

        if(cargar_modelo) {
            vector<Primitiva> modelo_prims;
            vector<Primitiva> modelo_prims2;
            vector<Primitiva> modelo_prims3;
            vector<Primitiva> modelo_prims4;
            vector<Primitiva> modelo_prims5;
            vector<Primitiva> modelo_prims6;
            vector<Primitiva> modelo_prims7;
            vector<Primitiva> modelo_prims8;
            vector<Primitiva> modelo_prims9;
            
            /*
            cargarModelo("../modelos/mono/scene.gltf", modelo_prims, 
                        1.5f, 0.0f, 0.0f, 0.0f, Vector3d(0.0f, 0.0f, -5.6f),
                        Color(0.8f, 0.4f, 0.15f), Color(0.05f, 0.05f, 0.05f), Color(0.0f, 0.0f, 0.0f), Color(0.0f, 0.0f, 0.0f), 1.5f);
            */
            /*
            cargarModelo("../modelos/conejo/scene.gltf", modelo_prims, 
                        1.0f, 0.0f, -90.0f, 0.0f, Vector3d(0.6f, -3.0f, -4.6f),
                        Color(0.8f, 0.4f, 0.15f), Color(0.0f, 0.0f, 0.0f), Color(0.0f, 0.0f, 0.0f), Color(0.0f, 0.0f, 0.0f), 1.5f);
            */

            // Arriba izquierda
            cargarModelo("../modelos/concurso/columna2.glb", modelo_prims, 
                        0.5f, 0.0f, -90.0f, 0.0f, Vector3d(-1.7f, -3.3f, -6.5f),
                        Color(0.75f, 0.65f, 0.55f), Color(0.01f, 0.01f, 0.01f), Color(0.0f, 0.0f, 0.0f), Color(0.0f, 0.0f, 0.0f), 1.0f);
                        
            // Abajo derecha
            cargarModelo("../modelos/concurso/columna1.glb", modelo_prims2, 
                        0.5f, 0.0f, -90.0f, 0.0f, Vector3d(1.7f, -3.1f, -2.5f),
                        Color(0.75f, 0.65f, 0.55f), Color(0.01f, 0.01f, 0.01f), Color(0.0f, 0.0f, 0.0f), Color(0.0f, 0.0f, 0.0f), 1.0f);

            // Arriba derecha
            cargarModelo("../modelos/concurso/columna3.glb", modelo_prims4,
                        0.5f, 0.0f, 180.0f, 0.0f, Vector3d(1.7f, -3.4f, -6.5f),
                        Color(0.75f, 0.65f, 0.55f), Color(0.01f, 0.01f, 0.01f), Color(0.0f, 0.0f, 0.0f), Color(0.0f, 0.0f, 0.0f), 1.0f);

            // Abajo izquierda
            cargarModelo("../modelos/concurso/columna_suelo.glb", modelo_prims5,
                        0.5f, 0.0f, -90.0f, 0.0f, Vector3d(-1.7f, -3.1f, -2.5f),
                        Color(0.75f, 0.65f, 0.55f), Color(0.01f, 0.01f, 0.01f), Color(0.0f, 0.0f, 0.0f), Color(0.0f, 0.0f, 0.0f), 1.0f);
            
            cargarModelo("../modelos/concurso/suelo2.glb", modelo_prims6,
                        0.5f, 0.0f, 0.0f, 0.0f, Vector3d(-4.0f, -4.8f, -5.2f),
                        Color(0.1,0.2,0.4), Color(0.0f, 0.0f, 0.0f), Color(0.0f, 0.0f, 0.0f), Color(0.0f, 0.0f, 0.0f), 1.0f);

            cargarModelo("../modelos/concurso/suelo_grieta.glb", modelo_prims7,
                        1.0f, 0.0f, 0.0f, 0.0f, Vector3d(0.0f, -3.3f, -5.2f),
                        Color(0.73f, 0.73f, 0.73f), Color(0.0f, 0.0f, 0.0f), Color(0.0f, 0.0f, 0.0f), Color(0.0f, 0.0f, 0.0f), 1.0f);

            cargarModelo("../modelos/concurso/esfera_rota.glb", modelo_prims8,
                        0.5f, 30.0f, 110.0f, 0.0f, Vector3d(1.0f, -2.8f, -2.0f),
                        Color(0.85f, 0.4f, 0.3f), Color(0.0f, 0.0f, 0.0f), Color(0.0f, 0.0f, 0.0f), Color(0.0f, 0.0f, 0.0f), 1.0f);
            
            cargarModelo("../modelos/concurso/cabeza.glb", modelo_prims9,
                        0.15f, 0.0f, 80.0f, 30.0f, Vector3d(0.0f, -1.4f, -2.3f),
                        Color(0.73f, 0.73f, 0.73f), Color(0.0f, 0.0f, 0.0f), Color(0.0f, 0.0f, 0.0f), Color(0.0f, 0.0f, 0.0f), 1.0f);

            modelo_prims.insert(modelo_prims.end(), modelo_prims2.begin(), modelo_prims2.end());
            modelo_prims.insert(modelo_prims.end(), modelo_prims3.begin(), modelo_prims3.end());
            modelo_prims.insert(modelo_prims.end(), modelo_prims4.begin(), modelo_prims4.end());
            modelo_prims.insert(modelo_prims.end(), modelo_prims5.begin(), modelo_prims5.end());
            modelo_prims.insert(modelo_prims.end(), modelo_prims6.begin(), modelo_prims6.end());
            modelo_prims.insert(modelo_prims.end(), modelo_prims7.begin(), modelo_prims7.end());
            modelo_prims.insert(modelo_prims.end(), modelo_prims8.begin(), modelo_prims8.end());
            modelo_prims.insert(modelo_prims.end(), modelo_prims9.begin(), modelo_prims9.end());

            
            // Construir BVH para el modelo cargado
            cout << "\n===============================================" << endl;
            cout << "====               ARBOL BVH               ====" << endl;
            cout << "===============================================" << endl;
            
            //bvh_modelo.construirBVH(modelo_prims);
            //bvh_modelo.obtenerInfo();
                
            bvh_modelo_tiny.construirBVH(modelo_prims);
            bvh_modelo_tiny.obtenerInfo();

            // Combinar escena base con modelo cargado
            combinarEscenas(&d_primitivas, &num_primitivas, &d_luces, &num_luces);
            int num_primitivas_malla = modelo_prims.size();
            cout << "   -Malla: " << num_primitivas_malla << " primitivas" << endl;
        } else {
            // Inicializar escena básica sin modelo
            inicializarEscena(&d_primitivas, &num_primitivas, &d_luces, &num_luces);
        }

        // Verificar errores después de inicializar escena
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            cerr << "Error al inicializar escena: " << cudaGetErrorString(err) << endl;
            return 1;
        }

        Camara* d_camara;
        cudaMalloc(&d_camara, sizeof(Camara));
        
        inicializarCamara<<<1, 1>>>(d_camara, ancho_imagen, alto_imagen);
        cudaDeviceSynchronize();
        
        // Verificar errores después de la inicialización de la cámara
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            cerr << "Error al inicializar cámara: " << cudaGetErrorString(err) << endl;
            return 1;
        }
        
        // Verificar reserva de memoria
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            cerr << "Error al reservar memoria para imagen: " << cudaGetErrorString(err) << endl;
            return 1;
        }

        // Inicializar estados aleatorios para samples per pixel
        curandState* rand_states = inicializarEstadosAleatorios(ancho_imagen, alto_imagen);

        // Configurar y lanzar kernel
        dim3 blockSize(16, 16);
        dim3 gridSize((ancho_imagen + blockSize.x - 1) / blockSize.x, 
                      (alto_imagen + blockSize.y - 1) / blockSize.y);

        SceneBounds scene_bounds;
        scene_bounds.min = Vector3d(-6.0f, -4.0f, -10.0f);
        scene_bounds.max = Vector3d( 6.0f,  6.0f,  2.0f);

        if (modo == 2) {
            bool ok = renderizarModoReconstruccion(
                ancho_imagen, alto_imagen, samples_per_pixel, nombre_archivo,
                ruta_modelo_entrenado,
                d_camara,
                d_primitivas, num_primitivas,
                d_luces, num_luces,
                d_primitivas_malla, num_primitivas_malla,
                bvh_modelo_tiny,
                rand_states,
                scene_bounds
            );

            cudaFree(d_camara);
            limpiarGPU(d_primitivas, d_luces, rand_states);
            return ok ? 0 : 1;
        }

        tcnn::json config_ganadora;
        //config_ganadora = ejecutarGridSearch(ancho_imagen, alto_imagen, d_camara, *transientRenderer, 
        //                d_primitivas, num_primitivas, d_luces, num_luces,
        //                d_primitivas_malla, num_primitivas_malla,
        //                bvh_modelo.getNodosGPU(), bvh_modelo.getPrimitivasGPU(), bvh_modelo.getNumNodos(),
        //                scene_bounds);

        
        cout << "\n================================================" << endl;
        cout << "====              RENDERIZANDO              ====" << endl;
        cout << "================================================" << endl;

        // Buffer imagen
        Color* d_imagen_data; 
        cudaMalloc(&d_imagen_data, num_pixels * sizeof(Color));
        ImagenGPU imagen(ancho_imagen, alto_imagen, d_imagen_data);

        // Buffers para nferencia
        DatosMLP* d_buffer_inference_inputs; 
        cudaMalloc(&d_buffer_inference_inputs, num_pixels * sizeof(DatosMLP));
        Color* d_buffer_radiance_predicha; cudaMalloc(&d_buffer_radiance_predicha, num_pixels * sizeof(Color));
        Color* d_buffer_throughput; cudaMalloc(&d_buffer_throughput, num_pixels * sizeof(Color));

        // Buffers para bootstrapping (Entrenamiento)
        RegistroEntrenamiento* d_buffer_registros_raw; cudaMalloc(&d_buffer_registros_raw, num_pixels * sizeof(RegistroEntrenamiento));
        DatosMLP* d_buffer_train_final; cudaMalloc(&d_buffer_train_final, num_pixels * sizeof(DatosMLP));
        
        // Buffer separado para la inferencia de los tails para evitar colisiones
        DatosMLP* d_buffer_tail_inputs; cudaMalloc(&d_buffer_tail_inputs, num_pixels * sizeof(DatosMLP));
        Color* d_buffer_tail_predicha; cudaMalloc(&d_buffer_tail_predicha, num_pixels * sizeof(Color));

        DatosMLP* d_buffer_train_shuffled; 
        cudaMalloc(&d_buffer_train_shuffled, num_pixels * sizeof(DatosMLP));

        unsigned int* d_mlp_counter = nullptr;
        cudaMalloc(&d_mlp_counter, sizeof(unsigned int));

        // Acumuladores CPU
        vector<Color> acumulador_cpu(ancho_imagen * alto_imagen, Color(0,0,0));
        vector<Color> frame_cpu(ancho_imagen * alto_imagen);

        // Parámetros de la red
        uint32_t n_in = 15;
        uint32_t n_out = 3;
        uint32_t batch_size_mlp = 16384;

        auto mlp = std::make_unique<ColorMLP>(n_in, n_out, batch_size_mlp, config_ganadora);
        mlp->setBounds(scene_bounds.min, scene_bounds.max);
        ofstream loss_file("loss_log.txt");
        cout << "[MLP] Red inicializada." << endl;

        // Inicializar visualizador
        Visualizador ventana(ancho_imagen, alto_imagen, "Renderizado en Tiempo Real - NRC");

        int WARMUP_SAMPLES = 48;
        
        int frames_acumulados_reales = 0;

        // Lanzar kernel principal
        auto inicio = chrono::high_resolution_clock::now();
        for (int s = 0; s < samples_per_pixel + WARMUP_SAMPLES; ++s) {
            // Fase 1 (Warmup): 100% entrenamiento
            // Fase 2 3% entrena, 97% infiere
            bool es_warmup = (s < WARMUP_SAMPLES);
            
            cudaMemset(d_mlp_counter, 0, sizeof(unsigned int));
            cudaMemset(d_imagen_data, 0, num_pixels * sizeof(Color));

            /*
            kernelRender<<<gridSize, blockSize>>>(
                d_camara, d_primitivas, num_primitivas, d_luces, num_luces,
                d_primitivas_malla, num_primitivas_malla, ancho_imagen, alto_imagen, 1,
                bvh_modelo.getNodosGPU(), bvh_modelo.getPrimitivasGPU(), bvh_modelo.getNumNodos(),
                imagen, rand_states,
                nullptr, d_buffer_registros_raw, d_mlp_counter, num_pixels,
                d_buffer_inference_inputs, d_buffer_throughput, 
                !es_warmup, true
            );
            */
            
            // Escenario con BVH Tiny
            kernelRender_tiny<<<gridSize, blockSize>>>(
                d_camara, d_primitivas, num_primitivas, d_luces, num_luces,
                d_primitivas_malla, num_primitivas_malla, ancho_imagen, alto_imagen, 1,
                bvh_modelo_tiny.obtenerDatosGPU(), bvh_modelo_tiny.getPrimitivasGPU(), bvh_modelo_tiny.getNumPrimitivas(),
                imagen, rand_states,
                nullptr, d_buffer_registros_raw, d_mlp_counter, num_pixels,
                d_buffer_inference_inputs, d_buffer_throughput, 
                !es_warmup, true
            );
            

            // Inferencia para imagen final
            mlp->inference(d_buffer_inference_inputs, d_buffer_radiance_predicha, num_pixels);
            kernelComposite<<<gridSize, blockSize>>>(d_imagen_data, d_buffer_radiance_predicha, d_buffer_throughput, ancho_imagen, alto_imagen);

            // ===================== Entrenamiento (BOOTSTRAPPING) =========================
            unsigned int n_train_host = 0; // Variable para almacenar el número de datos válidos para entrenamiento
            cudaMemcpy(&n_train_host, d_mlp_counter, sizeof(unsigned int), cudaMemcpyDeviceToHost);

            // Limitar a la capacidad máxima 
            n_train_host = min((unsigned int)65536, n_train_host);

            // Ajustar para que sea perfectamente divisible en 4 batches
            n_train_host = (n_train_host / 4) * 4; 
            int s_batches = 4; // Número de batches para dividir los datos de entrenamiento
            int l_records_per_batch = n_train_host / s_batches; // Datos por batch

            if (n_train_host > 1024) {
                int threads = 256;
                int blocks = (n_train_host + threads - 1) / threads;

                // Preparar Tail e Inferir
                kernelPrepararInferenciaTail<<<blocks, threads>>>(d_buffer_registros_raw, d_buffer_tail_inputs, n_train_host);
                mlp->inference(d_buffer_tail_inputs, d_buffer_tail_predicha, n_train_host);
                
                // Calcular Targets Finales (Sufijo + Predicción)
                kernelCalcularTargets<<<blocks, threads>>>(d_buffer_registros_raw, d_buffer_tail_predicha, d_buffer_train_final, n_train_host);
                
                // Shuffle con curand
                kernelShuffle<<<blocks, threads>>>(
                    d_buffer_train_final, 
                    d_buffer_train_shuffled, 
                    n_train_host,
                    rand_states
                );
                cudaDeviceSynchronize(); // Asegurar que el shuffle termine antes de entrenar
                
                // Distribuir sobre los 's' batches de 'l' datos cada uno
                for(int k = 0; k < s_batches; k++) {
                    // Desplazamos el puntero para coger un trozo distinto del array mezclado
                    DatosMLP* batch_ptr = d_buffer_train_shuffled + (k * l_records_per_batch);
                    
                    float loss = mlp->train_step(batch_ptr, l_records_per_batch);
                    
                    // Guardar loss
                    if (k == 0) loss_file << loss << endl; 
                }    
            }
            // ==============================================================================

            // Acumulación
            if (!es_warmup) {
                // Al terminar el warmup, reseteamos el acumulador para eliminar el ruido inicial
                if (s == WARMUP_SAMPLES) fill(acumulador_cpu.begin(), acumulador_cpu.end(), Color(0,0,0));
                
                frames_acumulados_reales++;
                cudaMemcpy(frame_cpu.data(), d_imagen_data, num_pixels * sizeof(Color), cudaMemcpyDeviceToHost);
                for(int i=0; i < num_pixels; i++) acumulador_cpu[i] = acumulador_cpu[i] + frame_cpu[i];
                
                if (s % 2 == 0) ventana.actualizar(acumulador_cpu, frames_acumulados_reales);
            }

            const char* fase = es_warmup ? "WARMUP" : "TRAIN+INFER";
            cout << "\rSample: " << (es_warmup ? s : frames_acumulados_reales) 
                 << " [" << fase << "]" << flush;
        }
        cout << endl;

        auto fin = chrono::high_resolution_clock::now();
        auto duracion = chrono::duration_cast<chrono::milliseconds>(fin - inicio);

        cout << "\n=================================================" << endl;
        cout << "====         PATH TRACING COMPLETADO         ====" << endl;
        cout << "=================================================" << endl;
        cout << "Tiempo de renderizado GPU: " << duracion.count() << " ms" << endl;
        loss_file.close();
        mlp->save_model("mlp_model.json");

        cout << "\n==============================================" << endl;
        cout << "====         GUARDANDO RESULTADOS         ====" << endl;
        cout << "==============================================" << endl;

        // Copiar datos de GPU a la imagen
        Imagen imagenCPU(ancho_imagen, alto_imagen);
        float inv_samples = 1.0f / (float)frames_acumulados_reales;
        for (int i = 0; i < ancho_imagen * alto_imagen; i++) {
            imagenCPU.datos[i] = acumulador_cpu[i] * inv_samples;
        }

        // Aplicar tone mapping
        Imagen imagen_final = imagenCPU.filmic();

        // Guardar imagen
        if (guardarPNG(imagen_final, nombre_archivo.c_str())) {
            cout << "Imagen guardada como: " << nombre_archivo << endl;
        } else {
            cout << "Error al guardar la imagen" << endl;
            return 1;
        }

        // Limpiar recursos GPU
        cudaFree(d_imagen_data);
        cudaFree(d_camara);
        cudaFree(d_mlp_counter);
        cudaFree(d_buffer_inference_inputs);
        cudaFree(d_buffer_radiance_predicha);
        cudaFree(d_buffer_throughput);
        cudaFree(d_buffer_registros_raw);
        cudaFree(d_buffer_train_final);
        cudaFree(d_buffer_tail_inputs);
        cudaFree(d_buffer_tail_predicha);
        cudaFree(d_buffer_train_shuffled);
        limpiarGPU(d_primitivas, d_luces, rand_states);
        
    } catch (const exception& e) {
        cerr << "ERROR: " << e.what() << endl;
        return 1;
    }


    return 0;
}