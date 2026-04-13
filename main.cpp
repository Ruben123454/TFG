// ################
// Autores: 
// Mir Ramos, Rubén 869039
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
#include <cmath>
#include <cuda_runtime.h>
#include "imagen.h"
#include "imagen_utils.h"
#include "imagen_gpu.h"
#include "camara.h"
#include "primitiva.h"
#include "color.h"
#include "vector3d.h"
#include "cargarModelo.h"
//#include "bvh.h"
#include "transient.h"
#include "mlp.h"
#include "mlp_types.h"
#include "visualizador.h"
#include "tinybvh_wrapper.h"
#include "RenderKernel.h"

using namespace std;

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


// Liberar memoria GPU
void limpiarGPU(Primitiva* d_primitivas, LuzPuntual* d_luces) {
    if (d_primitivas) {
        cudaFree(d_primitivas);
    }
    if (d_luces) {
        cudaFree(d_luces);
    }
}

bool moverCamara(GLFWwindow* window, Camara* d_camara, float delta_time_s, float velocidad=2.5f) {
    if (!window || !d_camara) {
        return false;
    }

    const bool key_w = glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS;
    const bool key_a = glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS;
    const bool key_s = glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS;
    const bool key_d = glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS;

    const bool key_up = glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS;
    const bool key_down = glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS; 
    const bool key_left = glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS;
    const bool key_right = glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS;

    if (!key_w && !key_a && !key_s && !key_d && !key_up && !key_down && !key_left && !key_right) {
        return false;
    }

    alignas(Camara) unsigned char cam_mem[sizeof(Camara)];
    Camara* cam_h = reinterpret_cast<Camara*>(cam_mem);

    cudaError_t err = cudaMemcpy(cam_h, d_camara, sizeof(Camara), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        return false;
    }

    if (key_up || key_down || key_left || key_right) {
        Vector3d front = cam_h->frente.normalized();
        double fy = std::max(-1.0, std::min(1.0, front.y));

        double yaw = atan2(front.x, -front.z);
        double pitch = asin(fy);

        const double rot_speed = 0.5; // rad/s
        const double rot_step = rot_speed * static_cast<double>(delta_time_s);

        if (key_left)  yaw -= rot_step;
        if (key_right) yaw += rot_step;
        if (key_up)    pitch += rot_step;
        if (key_down)  pitch -= rot_step;

        const double pitch_limit = 1.55334306; // ~89 grados
        if (pitch > pitch_limit) pitch = pitch_limit;
        if (pitch < -pitch_limit) pitch = -pitch_limit;

        Vector3d new_front(
            sin(yaw) * cos(pitch),
            sin(pitch),
            -cos(yaw) * cos(pitch)
        );
        new_front = new_front.normalized();

        Vector3d world_up(0.0, 1.0, 0.0);
        Vector3d new_left = new_front.cross(world_up).normalized();
        if (new_left.lengthSquared() < 1e-10) {
            new_left = cam_h->izquierda.normalized();
        }
        Vector3d new_up = new_left.cross(new_front).normalized();

        cam_h->frente = new_front;
        cam_h->izquierda = new_left;
        cam_h->arriba = new_up;
    }

    Vector3d forward = cam_h->frente.normalized();
    if (forward.lengthSquared() < 1e-10) {
        forward = Vector3d(0.0, 0.0, -1.0);
    }

    Vector3d right = (-cam_h->izquierda).normalized();
    if (right.lengthSquared() < 1e-10) {
        right = Vector3d(1.0, 0.0, 0.0);
    }

    Vector3d delta(0.0, 0.0, 0.0);
    double paso = static_cast<double>(velocidad) * static_cast<double>(delta_time_s);

    if (key_w) delta = delta + forward * paso;
    if (key_s) delta = delta - forward * paso;
    if (key_d) delta = delta - right * paso;
    if (key_a) delta = delta + right * paso;

    if (delta.lengthSquared() < 1e-12) {
        err = cudaMemcpy(d_camara, cam_h, sizeof(Camara), cudaMemcpyHostToDevice);
        return err == cudaSuccess;
    }

    cam_h->posicion = cam_h->posicion + delta;

    err = cudaMemcpy(d_camara, cam_h, sizeof(Camara), cudaMemcpyHostToDevice);
    return err == cudaSuccess;
}

// Renderizar usando la red neuronal para inferencia (modo reconstrucción)
bool renderizarModoReconstruccion(int ancho_imagen, int alto_imagen, string nombre_archivo,
                                   const string& ruta_modelo_entrenado,
                                   int samples_per_pixel,
                                   Camara* d_camara, Primitiva* d_primitivas, int num_primitivas,
                                   LuzPuntual* d_luces, int num_luces, TinyBVH& bvh_modelo_tiny,
                                   Primitiva* d_primitivas_malla, int num_primitivas_malla,
                                   const SceneBounds& scene_bounds,
                                   TransientRender* transientRenderer, double t_final, bool activar_transient, 
                                   Visualizador* ventana, RenderGuiState& runtime_state) {
    int num_pixels = ancho_imagen * alto_imagen;
    if (samples_per_pixel < 1) samples_per_pixel = 1;
    
    cout << "\n================================================" << endl;
    cout << "====         MODO RECONSTRUCCIÓN            ====" << endl;
    cout << "================================================" << endl;

    tcnn::json config_ganadora;
    uint32_t n_in = 16;
    uint32_t n_out = 3;
    uint32_t batch_size_mlp = 65536;
    auto mlp = std::make_unique<ColorMLP>(n_in, n_out, batch_size_mlp, config_ganadora);
    mlp->setBounds(scene_bounds.min, scene_bounds.max, scene_bounds.t_min, scene_bounds.t_max);

    if (!mlp->load_model(ruta_modelo_entrenado)) {
        cerr << "[MLP] No se pudo cargar el modelo neuronal." << endl;
        return false;
    }

    // Configurar grid y blocks
    dim3 blockSize(16, 16);
    dim3 gridSize((ancho_imagen + blockSize.x - 1) / blockSize.x, 
                  (alto_imagen + blockSize.y - 1) / blockSize.y);

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

    if (ventana == nullptr) {
        cerr << "[UI] Ventana no válida en modo reconstrucción." << endl;
        cudaFree(d_imagen_data);
        cudaFree(d_buffer_inference_inputs);
        cudaFree(d_buffer_radiance_predicha);
        cudaFree(d_buffer_throughput);
        return false;
    }

    runtime_state.warmupActive = false;
    runtime_state.pauseRendering = false;
    runtime_state.resetAccumulation = false;

    ventana->actualizar(acumulador_cpu, 1);

    // Buffer para contadores
    unsigned int* d_mlp_counter = nullptr;
    cudaMalloc(&d_mlp_counter, sizeof(unsigned int));

    // Limpiar buffers
    cudaMemset(d_mlp_counter, 0, sizeof(unsigned int));
    cudaMemset(d_imagen_data, 0, num_pixels * sizeof(Color));

    cout << "\nLanzando rayos y recopilando datos..." << endl;
    cout << "Muestras por píxel en reconstrucción: " << samples_per_pixel << endl;

    auto inicio = chrono::high_resolution_clock::now();
    auto render_inicio = chrono::high_resolution_clock::now();
    auto last_input_time = chrono::high_resolution_clock::now();
    int sample_actual = 0;
    int iteracion = 0;

    while (sample_actual < samples_per_pixel && ventana->procesarEventos()) {
        iteracion++;

        auto now_input = chrono::high_resolution_clock::now();
        float dt_input = chrono::duration<float>(now_input - last_input_time).count();
        last_input_time = now_input;

        if (moverCamara(ventana->window, d_camara, dt_input)) {
            runtime_state.resetAccumulation = true;
        }

        if (runtime_state.resetAccumulation) {
            fill(acumulador_cpu.begin(), acumulador_cpu.end(), Color(0,0,0));
            sample_actual = 0;
            runtime_state.resetAccumulation = false;
            render_inicio = chrono::high_resolution_clock::now();
            if(activar_transient) {
                transientRenderer->limpiarAcumulado();
            }
        }

        if(runtime_state.configUpdate) {
            samples_per_pixel = runtime_state.samplesPerPixel;
            runtime_state.configUpdate = false;
        }

        if (runtime_state.pauseRendering) {
            int muestras_gui = sample_actual > 0 ? sample_actual : 1;
            ventana->actualizar(acumulador_cpu, muestras_gui);
            std::this_thread::sleep_for(std::chrono::milliseconds(8));
            continue;
        }

        auto frame_inicio = chrono::high_resolution_clock::now();
        cudaMemset(d_mlp_counter, 0, sizeof(unsigned int));
        cudaMemset(d_imagen_data, 0, num_pixels * sizeof(Color));
        cudaMemset(d_buffer_inference_inputs, 0, num_pixels * sizeof(DatosMLP));
        cudaMemset(d_buffer_throughput, 0, num_pixels * sizeof(Color));

        // Lanzar kernel de renderizado que recopila datos
        launchKernelRenderTiny(
            gridSize, blockSize,
            d_camara, d_primitivas, num_primitivas, d_luces, num_luces,
            d_primitivas_malla, num_primitivas_malla, ancho_imagen, alto_imagen, 1, sample_actual,
            bvh_modelo_tiny.obtenerDatosGPU(), bvh_modelo_tiny.getPrimitivasGPU(), bvh_modelo_tiny.getNumPrimitivas(),
            imagen, *transientRenderer,
            nullptr,
            nullptr,
            d_mlp_counter, num_pixels,
            d_buffer_inference_inputs, d_buffer_throughput,
            true,   // usar_red_inferencia = true
            false,  // entrenar_red = false (solo inferencia)
            true // Modo reconstruccion: obtener datos solo del primer rebote para inferir
        );

        cudaDeviceSynchronize();

        // Inferencia para generar predicción
        mlp->inference(d_buffer_inference_inputs, d_buffer_radiance_predicha, num_pixels);

        // Componer imagen final: luz directa + (throughput * predicción)
        launchKernelComposite(gridSize, blockSize, d_imagen_data, d_buffer_radiance_predicha, d_buffer_throughput, ancho_imagen, alto_imagen, true);

        // Si está activo el transient, depositar también la contribución NRC en los bins temporales
        if (activar_transient) {
            launchKernelTransientComposite(
                gridSize, blockSize,
                d_buffer_inference_inputs, d_buffer_radiance_predicha, d_buffer_throughput,
                *transientRenderer, ancho_imagen, alto_imagen, true
            );
        }

        cudaDeviceSynchronize();

        cudaMemcpy(frame_cpu.data(), d_imagen_data, num_pixels * sizeof(Color), cudaMemcpyDeviceToHost);
        for (int i = 0; i < num_pixels; i++) {
            acumulador_cpu[i] = acumulador_cpu[i] + frame_cpu[i];
        }

        sample_actual++;

        auto frame_fin = chrono::high_resolution_clock::now();
        runtime_state.lastFrameMs = chrono::duration<float, std::milli>(frame_fin - frame_inicio).count();
        runtime_state.lastFrameMs = chrono::duration<float, std::milli>(frame_fin - frame_inicio).count();
        runtime_state.totalRenderMs = chrono::duration<float, std::milli>(frame_fin - render_inicio).count();

        if (iteracion % 2 == 0) {
            ventana->actualizar(acumulador_cpu, sample_actual);
        }

        cout << "\rSample reconstrucción: " << (sample_actual + 1) << "/" << samples_per_pixel << flush;
    }
    if (glfwWindowShouldClose(ventana->window)) {
        cout << "\nVentana cerrada por el usuario." << endl;
    }
    cout << endl;

    auto fin = chrono::high_resolution_clock::now();
    auto duracion = chrono::duration_cast<chrono::milliseconds>(fin - inicio);

    cout << "\n=================================================" << endl;
    cout << "====    RECONSTRUCCIÓN COMPLETADA            ====" << endl;
    cout << "=================================================" << endl;
    cout << "Tiempo total: " << duracion.count() << " ms" << endl;

    // Crear imagen final promediada por número de muestras
    Imagen imagen_final(ancho_imagen, alto_imagen);
    float inv_samples = 1.0f / (float)samples_per_pixel;
    for (int i = 0; i < num_pixels; i++) {
        imagen_final.datos[i] = acumulador_cpu[i] * inv_samples;
    }

    cout << "\n==============================================" << endl;
    cout << "====         GUARDANDO RESULTADOS         ====" << endl;
    cout << "==============================================" << endl;

    // Aplicar tone mapping
    Imagen imagen_final_tone_mapping = imagen_final.reinhard().gamma();

    // Guardar imagen
    if (guardarPNG(imagen_final_tone_mapping, nombre_archivo.c_str())) {
        cout << "Imagen guardada como: " << nombre_archivo << endl;
    } else {
        cout << "Error al guardar la imagen" << endl;
    }

    if (activar_transient) {
        for (int i = 0; i < transientRenderer->num_frames; ++i) {
            vector<Color> buffer_host = transientRenderer->obtenerFrameHost(i);

            Imagen img_temp(ancho_imagen, alto_imagen);
            for (int k = 0; k < ancho_imagen * alto_imagen; k++) {
                Color c = buffer_host[k] * inv_samples;
                img_temp.datos[k] = c;
            }

            string base_nombre = nombre_archivo;
            size_t pos_ext = base_nombre.rfind('.');
            if (pos_ext != string::npos) base_nombre = base_nombre.substr(0, pos_ext);
            string nombre = "../transient/" + base_nombre + "_" + to_string(i) + ".png";

            Imagen res = img_temp.reinhard().gamma();
            res = res * 2.0f;
            res = res.clamping();
            guardarPNG(res, nombre.c_str());
        }
        cout << "Imágenes transient guardadas en carpeta 'transient/'" << endl;
    }

    runtime_state.accumulatedSamples = sample_actual;
    runtime_state.warmupActive = false;
    runtime_state.pauseRendering = true;
    runtime_state.renderingComplete = true;
    runtime_state.requestSave = false;

    while (runtime_state.renderingComplete && ventana->procesarEventos()) {
        int muestras_gui = sample_actual > 0 ? sample_actual : 1;
        ventana->actualizar(acumulador_cpu, muestras_gui);
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }

    if (runtime_state.requestSave) {
        nombre_archivo = runtime_state.outputFileName;
        if (guardarPNG(imagen_final_tone_mapping, nombre_archivo.c_str())) {
            cout << "Imagen guardada como: " << nombre_archivo << endl;
        } else {
            cout << "Error al guardar la imagen" << endl;
        }
    } else {
        cout << "Imagen no guardada." << endl;
    }

    // Limpiar
    cudaFree(d_imagen_data);
    cudaFree(d_buffer_inference_inputs);
    cudaFree(d_buffer_radiance_predicha);
    cudaFree(d_buffer_throughput);
    cudaFree(d_mlp_counter);
    return true;
}

int main() {
    // Habilitar mapeo de memoria host
    cudaSetDeviceFlags(cudaDeviceMapHost);

    try {
        cout << "================================================" << endl;
        cout << "====       INICIALIZANDO INTERFAZ         ====" << endl;
        cout << "================================================" << endl;
        
        // Crear visualizador con configuración inicial
        std::unique_ptr<Visualizador> ventana_config = std::make_unique<Visualizador>(1024, 768, "Renderizador NRC - Configuración");
        RenderGuiState gui_state;

        // Esperar a que el usuario configure los parámetros
        bool config_done = false;
        while (!config_done && ventana_config->procesarEventos()) {
            int fbw = 0, fbh = 0;
            glfwGetFramebufferSize(ventana_config->window, &fbw, &fbh);
            glViewport(0, 0, fbw, fbh);
            glClearColor(0.08f, 0.08f, 0.10f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);

            ventana_config->gui.beginFrame();
            ventana_config->gui.drawConfigScreen(gui_state);
            ventana_config->gui.endFrame(fbw, fbh);
            glfwSwapBuffers(ventana_config->window);

            if (gui_state.readyToRender) {
                config_done = true;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(16));
        }

        if (!config_done) {
            cout << "Configuración cancelada por el usuario." << endl;
            return 0;
        }

        int ancho_imagen = gui_state.imageWidth;
        int alto_imagen = gui_state.imageHeight;
        int samples_per_pixel = gui_state.samplesPerPixel;
        bool cargar_modelo = gui_state.loadModel;
        string nombre_archivo = gui_state.outputFileName;
        string ruta_modelo_entrenado = gui_state.mlpModelPath;
        int num_pixels = ancho_imagen * alto_imagen;
        bool activar_transient = gui_state.activarTransient;
        int num_frames_transient = gui_state.transientFrames;
        int WARMUP_SAMPLES = 200;

        // Reutilizar la misma ventana (evita reinicio de interfaz)
        std::unique_ptr<Visualizador> ventana = std::move(ventana_config);
        ventana->configurarRenderTarget(ancho_imagen, alto_imagen, "Renderizado en Tiempo Real - NRC");
        RenderGuiState& runtime_state = ventana->uiState();
        runtime_state = gui_state;

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
                        0.8f, -90.0f, -90.0f, 0.0f, Vector3d(0.6f, -1.0f, -4.6f),
                        Color(0.8f, 0.4f, 0.15f), Color(0.0f, 0.0f, 0.0f), Color(0.0f, 0.0f, 0.0f), Color(0.0f, 0.0f, 0.0f), 1.5f);
            */
            /*
            cargarModelo("../modelos/conejo/scene.gltf", modelo_prims, 
                    0.8f, -90.0f, -90.0f, 0.0f, Vector3d(-3.3f, -1.0f, -4.0f),
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
            num_primitivas_malla = modelo_prims.size();
            cout << "   -Malla: " << num_primitivas_malla << " primitivas" << endl;

            runtime_state.bvh = true;
            runtime_state.bvhNodes = bvh_modelo_tiny.getNumNodos();
            runtime_state.bvhPrimitives = bvh_modelo_tiny.getNumPrimitivas();
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
        
        launchInicializarCamara(d_camara, ancho_imagen, alto_imagen);
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


        // Configurar y lanzar kernel
        dim3 blockSize(16, 16);
        dim3 gridSize((ancho_imagen + blockSize.x - 1) / blockSize.x, 
                      (alto_imagen + blockSize.y - 1) / blockSize.y);


        // TRANSIENT RENDERING
        // Crear TransientRenderer para medir tiempo
        // Escala de nanosegundos
        double t_start = 2.8e-8;
        double t_final = 9.8e-8;
        if(cargar_modelo) {
            /*
            t_start = 6.0e-8;
            t_final = 11.0e-8;
            */
            t_start = 1.5e-8;
            t_final = 10.7e-8;
        }
        int n_frames = num_frames_transient;
        double frame_duration = (t_final - t_start) / n_frames;
        double sigma = frame_duration; // Sigma igual a la duración del frame para suavizado
        
        TransientRender* transientRenderer;
        if (activar_transient) {
            transientRenderer = new TransientRender(ancho_imagen, alto_imagen, t_start, t_final, n_frames, sigma);
            cout << "[Transient] Render transitorio ACTIVADO (" << n_frames << " frames)" << endl;
        } else {
            transientRenderer = new TransientRender(1, 2, 1, 2, 1, 1); // Básico para evitar errores
            cout << "[Transient] Render transitorio DESACTIVADO" << endl;
        }
        
        SceneBounds scene_bounds;
        scene_bounds.min = Vector3d(-6.0f, -4.0f, -10.0f);
        scene_bounds.max = Vector3d( 6.0f,  6.0f,  2.0f);
        scene_bounds.t_min = 0.0f;
        scene_bounds.t_max = t_final * 1.5f;

        // MODO RECONSTRUCCIÓN: Solo inferencia de MLP
        if (gui_state.renderMode == 1) {
            if (!renderizarModoReconstruccion(ancho_imagen, alto_imagen, nombre_archivo, ruta_modelo_entrenado,
                                         samples_per_pixel,
                                         d_camara, d_primitivas, num_primitivas, d_luces, num_luces,
                                         bvh_modelo_tiny, d_primitivas_malla, num_primitivas_malla,
                                         scene_bounds, transientRenderer, t_final, activar_transient,
                                         ventana.get(), runtime_state)) {
                return 1;
            }
            cudaFree(d_camara);
            return 0;
        }

        // MODO ENTRENAMIENTO: Path Tracing + MLP Training
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
        Color* d_buffer_radiance_predicha; 
        cudaMalloc(&d_buffer_radiance_predicha, num_pixels * sizeof(Color));
        Color* d_buffer_throughput; 
        cudaMalloc(&d_buffer_throughput, num_pixels * sizeof(Color));

        // Buffers para bootstrapping (Entrenamiento)
        RegistroEntrenamiento* d_buffer_registros_raw; 
        cudaMalloc(&d_buffer_registros_raw, num_pixels * sizeof(RegistroEntrenamiento));
        DatosMLP* d_buffer_train_final; 
        cudaMalloc(&d_buffer_train_final, num_pixels * sizeof(DatosMLP));
        
        // Buffer separado para la inferencia de los tails para evitar colisiones
        DatosMLP* d_buffer_tail_inputs; 
        cudaMalloc(&d_buffer_tail_inputs, num_pixels * sizeof(DatosMLP));
        Color* d_buffer_tail_predicha; 
        cudaMalloc(&d_buffer_tail_predicha, num_pixels * sizeof(Color));

        unsigned int* d_mlp_counter = nullptr;
        cudaMalloc(&d_mlp_counter, sizeof(unsigned int));

        // Acumuladores CPU
        vector<Color> acumulador_cpu(ancho_imagen * alto_imagen, Color(0,0,0));
        vector<Color> frame_cpu(ancho_imagen * alto_imagen);

        // Parámetros de la red
        uint32_t n_in = 16;
        uint32_t n_out = 3;
        uint32_t batch_size_mlp = 65536;

        auto mlp = std::make_unique<ColorMLP>(n_in, n_out, batch_size_mlp, config_ganadora);
        mlp->setBounds(scene_bounds.min, scene_bounds.max, scene_bounds.t_min, scene_bounds.t_max);
        cout << "[MLP] Red inicializada." << endl;
        ofstream loss_file("loss_log.txt");

        int frames_acumulados_reales = 0;

        // Mostrar la ventana de render
        runtime_state.warmupSamplesTotal = std::max(0, WARMUP_SAMPLES);
        runtime_state.warmupSamplesDone = 0;
        runtime_state.warmupActive = (WARMUP_SAMPLES > 0);
        ventana->actualizar(acumulador_cpu, 1);

        // Lanzar kernel principal
        auto inicio = chrono::high_resolution_clock::now();
        auto render_inicio = chrono::high_resolution_clock::now();
        auto last_input_time = chrono::high_resolution_clock::now();
        for (int s = 0; s < samples_per_pixel + WARMUP_SAMPLES; ++s) {
            if (!ventana->procesarEventos()) {
                cout << "\nVentana cerrada por el usuario." << endl;
                break;
            }

            auto now_input = chrono::high_resolution_clock::now();
            float dt_input = chrono::duration<float>(now_input - last_input_time).count();
            last_input_time = now_input;

            if (moverCamara(ventana->window, d_camara, dt_input)) {
                runtime_state.resetAccumulation = true;
            }

            if (runtime_state.resetAccumulation) {
                fill(acumulador_cpu.begin(), acumulador_cpu.end(), Color(0,0,0));
                frames_acumulados_reales = 0;
                runtime_state.resetAccumulation = false;
                s = WARMUP_SAMPLES;
                render_inicio = chrono::high_resolution_clock::now();
                if(activar_transient) {
                    transientRenderer->limpiarAcumulado();
                }
            }

            if (runtime_state.pauseRendering) {
                int muestras_gui = frames_acumulados_reales > 0 ? frames_acumulados_reales : 1;
                ventana->actualizar(acumulador_cpu, muestras_gui);
                std::this_thread::sleep_for(std::chrono::milliseconds(8));
                s--;
                continue;
            }

            if(runtime_state.configUpdate) {
                samples_per_pixel = runtime_state.samplesPerPixel;
                runtime_state.configUpdate = false;
            }

            auto frame_inicio = chrono::high_resolution_clock::now();
            
            // Fase 1 (Warmup): 100% entrenamiento
            // Fase 2 3% entrena, 97% infiere
            bool es_warmup = (WARMUP_SAMPLES > 0) && (s < WARMUP_SAMPLES);
            runtime_state.warmupActive = es_warmup;

            if (WARMUP_SAMPLES > 0) {
                runtime_state.warmupSamplesTotal = WARMUP_SAMPLES;
                runtime_state.warmupSamplesDone = (s < WARMUP_SAMPLES) ? (s + 1) : WARMUP_SAMPLES;
            } else {
                runtime_state.warmupSamplesTotal = 0;
                runtime_state.warmupSamplesDone = 0;
            }

            // Limpiar todo lo acumulado durante warmup
            if (WARMUP_SAMPLES > 0 && s == WARMUP_SAMPLES) {
                fill(acumulador_cpu.begin(), acumulador_cpu.end(), Color(0,0,0));
                if (activar_transient && transientRenderer) {
                    transientRenderer->limpiarAcumulado();
                }
            }

            cudaMemset(d_mlp_counter, 0, sizeof(unsigned int));
            cudaMemset(d_imagen_data, 0, num_pixels * sizeof(Color));
            // Limpiar buffers de inferencia para evitar residuos
            cudaMemset(d_buffer_inference_inputs, 0, num_pixels * sizeof(DatosMLP));
            cudaMemset(d_buffer_throughput, 0, num_pixels * sizeof(Color));
            /*
            // Escenario con BVH propio
            kernelRender<<<gridSize, blockSize>>>(
                d_camara, d_primitivas, num_primitivas, d_luces, num_luces,
                d_primitivas_malla, num_primitivas_malla, ancho_imagen, alto_imagen, 1,
                bvh_modelo.getNodosGPU(), bvh_modelo.getPrimitivasGPU(), bvh_modelo.getNumNodos(),
                imagen, *transientRenderer, rand_states,
                nullptr, d_buffer_registros_raw, d_mlp_counter, num_pixels,
                d_buffer_inference_inputs, d_buffer_throughput, 
                !es_warmup, true, s
            );
            */
            
            // Escenario con BVH Tiny
            launchKernelRenderTiny(
                gridSize, blockSize,
                d_camara, d_primitivas, num_primitivas, d_luces, num_luces,
                d_primitivas_malla, num_primitivas_malla, ancho_imagen, alto_imagen, 1, s,
                bvh_modelo_tiny.obtenerDatosGPU(), bvh_modelo_tiny.getPrimitivasGPU(), bvh_modelo_tiny.getNumPrimitivas(),
                imagen, *transientRenderer,
                nullptr, d_buffer_registros_raw, d_mlp_counter, num_pixels,
                d_buffer_inference_inputs, d_buffer_throughput, 
                !es_warmup, true, false
            );
            
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                cerr << "Error en kernelRender_tiny (launch) en sample " << s << ": "
                     << cudaGetErrorString(err) << endl;
                return 1;
            }

            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                cerr << "Error en kernelRender_tiny (exec) en sample " << s << ": "
                     << cudaGetErrorString(err) << endl;
                return 1;
            }


            if(!es_warmup){
                // Inferencia para imagen final
                mlp->inference(d_buffer_inference_inputs, d_buffer_radiance_predicha, num_pixels);
                launchKernelComposite(gridSize, blockSize, d_imagen_data, d_buffer_radiance_predicha, d_buffer_throughput, ancho_imagen, alto_imagen, false);

                // Si está activo el transient, depositar también la contribución NRC en los bins temporales
                if (activar_transient && !es_warmup) {
                    launchKernelTransientComposite(
                        gridSize, blockSize,
                        d_buffer_inference_inputs, d_buffer_radiance_predicha, d_buffer_throughput,
                        *transientRenderer, ancho_imagen, alto_imagen, false
                    );
                }
            }

            // ===================== Entrenamiento (BOOTSTRAPPING) =========================
            unsigned int n_train_host = 0; // Variable para almacenar el número de datos válidos para entrenamiento
            cudaMemcpy(&n_train_host, d_mlp_counter, sizeof(unsigned int), cudaMemcpyDeviceToHost);

            // Limitar a la capacidad máxima 
            n_train_host = min((unsigned int)65536, n_train_host);

            // Ajustar para que sea perfectamente divisible en 4 batches
            n_train_host = (n_train_host / 4) * 4; 

            if (n_train_host > 1024) {
                int threads = 256;
                int blocks = (n_train_host + threads - 1) / threads;

                // Preparar Tail e Inferir
                launchKernelPrepararInferenciaTail(dim3(blocks), dim3(threads), d_buffer_registros_raw, d_buffer_tail_inputs, n_train_host);
                mlp->inference(d_buffer_tail_inputs, d_buffer_tail_predicha, n_train_host);
                
                // Calcular Targets Finales (Sufijo + Predicción)
                launchKernelCalcularTargets(dim3(blocks), dim3(threads), d_buffer_registros_raw, d_buffer_tail_predicha, d_buffer_train_final, n_train_host);
                
                cudaDeviceSynchronize();

                runtime_state.trainingLoss = mlp->train_step(d_buffer_train_final, n_train_host);
                
                // Guardar loss
                loss_file << runtime_state.trainingLoss << endl; 
                 
            }
            // ==============================================================================

            // Acumulación
            if (!es_warmup) {
                frames_acumulados_reales++;
                cudaMemcpy(frame_cpu.data(), d_imagen_data, num_pixels * sizeof(Color), cudaMemcpyDeviceToHost);
                for(int i=0; i < num_pixels; i++) acumulador_cpu[i] = acumulador_cpu[i] + frame_cpu[i];

                auto frame_fin = chrono::high_resolution_clock::now();
                runtime_state.lastFrameMs = chrono::duration<float, std::milli>(frame_fin - frame_inicio).count();
                runtime_state.totalRenderMs = chrono::duration<float, std::milli>(frame_fin - render_inicio).count();
                
                if (s % 2 == 0) ventana->actualizar(acumulador_cpu, frames_acumulados_reales);
            } else {
                // Mantener UI viva durante el warmup (solo entrenamiento)
                if (s % 2 == 0) ventana->actualizar(acumulador_cpu, 1);
            }


            auto sample_fin = chrono::high_resolution_clock::now();
            runtime_state.totalExecutionMs = chrono::duration<float, std::milli>(sample_fin - inicio).count();

            const char* fase = es_warmup ? "WARMUP" : "TRAIN+INFER";
            cout << "\rSample: " << (es_warmup ? s : frames_acumulados_reales) 
                 << " [" << fase << "]" << flush;
        }
        cout << endl;

        auto fin = chrono::high_resolution_clock::now();
        auto duracion = chrono::duration_cast<chrono::milliseconds>(fin - inicio);
        auto duracion_render = chrono::duration_cast<chrono::milliseconds>(fin - render_inicio);
        runtime_state.totalRenderMs = duracion_render.count();
        runtime_state.totalExecutionMs = duracion.count();

        cout << "\n=================================================" << endl;
        cout << "====         PATH TRACING COMPLETADO         ====" << endl;
        cout << "=================================================" << endl;
        cout << "Tiempo de renderizado GPU: " << duracion.count() << " ms" << endl;
        loss_file.close();

        if(runtime_state.saveMlpModel) {
            mlp->save_model("mlp_model.json");
        }

        cout << "\n==============================================" << endl;
        cout << "====         GUARDANDO RESULTADOS         ====" << endl;
        cout << "==============================================" << endl;

        // TRANSIENT 
        if (activar_transient) {
            const float transient_exposure_min = 10.0f;
            const float transient_exposure_max = 20.0f;
            for (int i = 0; i < transientRenderer->num_frames; ++i) {
                // Traer datos de GPU a CPU de forma segura
                vector<Color> buffer_host = transientRenderer->obtenerFrameHost(i);

                float alpha = 0.0f;
                if (transientRenderer->num_frames > 1) {
                    alpha = (float)i / (float)(transientRenderer->num_frames - 1);
                }
                float exposure = transient_exposure_min + alpha * (transient_exposure_max - transient_exposure_min);
                
                // Crear una imagen temporal en CPU para guardar
                Imagen img_temp(ancho_imagen, alto_imagen);
                // Copiar datos del vector al formato que use tu clase Imagen
                for(int k=0; k<ancho_imagen*alto_imagen; k++) {
                    // Normalizar por el número de muestras
                    Color c = buffer_host[k] / samples_per_pixel;
                    c = c * exposure;
                    img_temp.datos[k] = c;
                }
                
                // Quitar extensión del nombre si la tiene
                string base_nombre = nombre_archivo;
                size_t pos_ext = base_nombre.rfind('.');
                if (pos_ext != string::npos) base_nombre = base_nombre.substr(0, pos_ext);
                string nombre = "../transient/" + base_nombre + "_" + to_string(i) + ".png";
                Imagen res = img_temp.gamma();
                guardarPNG(res, nombre.c_str());
            }
            cout << "Imágenes transient guardadas en carpeta 'transient/'" << endl;
        } else {
            cout << "Render transitorio desactivado, no se guardan frames transient." << endl;
        }

        // Crear imagen final promediando muestras
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

        if (activar_transient) {
            std::ofstream outFile("transient_volume.bin", std::ios::binary);
            for (int i = 0; i < transientRenderer->num_frames; ++i) {
                vector<Color> buffer_host = transientRenderer->obtenerFrameHost(i);
                for(int k=0; k < ancho_imagen * alto_imagen; k++) {
                    // Guardar la intensidad
                    float luminancia = (buffer_host[k].r + buffer_host[k].g + buffer_host[k].b) / 3.0f;
                    outFile.write(reinterpret_cast<const char*>(&luminancia), sizeof(float));
                }
            }
            outFile.close();
            cout << "Volumen binario guardado para Python." << endl;
        }

        cout << "\n=================================================" << endl;
        cout << "====      RENDERIZADO FINALIZADO - ESPERANDO ====" << endl;
        cout << "=================================================" << endl;

        // Mantener el render visible y decidir guardado desde la propia UI
        RenderGuiState& post_state = ventana->uiState();
        post_state = runtime_state;
        post_state.accumulatedSamples = frames_acumulados_reales;
        post_state.warmupActive = false;
        post_state.pauseRendering = true;
        post_state.renderingComplete = true;
        post_state.requestSave = false;

        while (post_state.renderingComplete && ventana->procesarEventos()) {
            int muestras_gui = frames_acumulados_reales > 0 ? frames_acumulados_reales : 1;
            ventana->actualizar(acumulador_cpu, muestras_gui);
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
        }

        // Guardar imagen si el usuario lo solicitó
        if (post_state.requestSave) {
            nombre_archivo = runtime_state.outputFileName;
            if (guardarPNG(imagen_final, nombre_archivo.c_str())) {
                cout << "Imagen guardada como: " << nombre_archivo << endl;
            } else {
                cout << "Error al guardar la imagen" << endl;
            }
        } else {
            cout << "Imagen no guardada." << endl;
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
        limpiarGPU(d_primitivas, d_luces);

        // Limpiar transient renderer
        if (transientRenderer) {
            transientRenderer->liberar();
            delete transientRenderer;
        }
        
    } catch (const exception& e) {
        cerr << "ERROR: " << e.what() << endl;
        return 1;
    }


    return 0;
}