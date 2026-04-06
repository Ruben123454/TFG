// ################
// Autores: 
// Mir Ramos, Rubén 869039
// Lopez Torralba, Alejandro 845154
//
// main.cpp
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
#include "color.h"
#include "vector3d.h"
#include "cargarModelo.h"
#include "bvh.h"
#include "mlp.h"
#include "mlp_types.h"
#include "gridsearch.h"
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


// Inicializar estados aleatorios para cada thread
// Liberar memoria GPU
void limpiarGPU(Primitiva* d_primitivas, LuzPuntual* d_luces) {
    if (d_primitivas) {
        cudaFree(d_primitivas);
    }
    if (d_luces) {
        cudaFree(d_luces);
    }
}

bool moverCamara(GLFWwindow* window, Camara* d_camara, float delta_time_s, float velocidad = 2.5f) {
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


bool renderizarModoReconstruccion(
    int ancho_imagen,
    int alto_imagen,
    int samples_per_pixel,
    string nombre_archivo,
    const string& ruta_modelo,
    Camara* d_camara,
    const Primitiva* d_primitivas,
    int num_primitivas,
    const LuzPuntual* d_luces,
    int num_luces,
    const Primitiva* d_primitivas_malla,
    int num_primitivas_malla,
    const TinyBVH& bvh_modelo_tiny,
    const SceneBounds& scene_bounds,
    Visualizador* ventana,
    RenderGuiState& runtime_state
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

        cudaMemset(d_imagen_data, 0, num_pixels * sizeof(Color));
        cudaMemset(d_buffer_inference_inputs, 0, num_pixels * sizeof(DatosMLP));
        cudaMemset(d_buffer_throughput, 0, num_pixels * sizeof(Color));

        launchKernelRenderTiny(gridSize, blockSize,
            d_camara, d_primitivas, num_primitivas, d_luces, num_luces,
            d_primitivas_malla, num_primitivas_malla, ancho_imagen, alto_imagen, 1, sample_actual,
            bvh_modelo_tiny.obtenerDatosGPU(), bvh_modelo_tiny.getPrimitivasGPU(), bvh_modelo_tiny.getNumPrimitivas(),
            imagen,
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
        launchKernelComposite(gridSize, blockSize, d_imagen_data, d_buffer_radiance_predicha, d_buffer_throughput, ancho_imagen, alto_imagen);
        cudaDeviceSynchronize();

        cudaMemcpy(frame_cpu.data(), d_imagen_data, num_pixels * sizeof(Color), cudaMemcpyDeviceToHost);
        for (int i = 0; i < num_pixels; ++i) {
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

        cout << "\rSample: " << sample_actual << "/" << samples_per_pixel << " [RECONSTRUCCION]" << flush;
    }

    if (glfwWindowShouldClose(ventana->window)) {
        cout << "\nVentana cerrada por el usuario." << endl;
    }
    cout << endl;

    auto fin = chrono::high_resolution_clock::now();
    auto duracion = chrono::duration_cast<chrono::milliseconds>(fin - inicio);

    cout << "\n=================================================" << endl;
    cout << "====       RECONSTRUCCIÓN COMPLETADA       ====" << endl;
    cout << "=================================================" << endl;
    cout << "Tiempo de renderizado GPU: " << duracion.count() << " ms" << endl;

    Imagen imagenCPU(ancho_imagen, alto_imagen);
    float inv_samples = 1.0f / (float)max(1, sample_actual);
    for (int i = 0; i < num_pixels; ++i) {
        imagenCPU.datos[i] = acumulador_cpu[i] * inv_samples;
    }

    Imagen imagen_final = imagenCPU.filmic();

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
        if (guardarPNG(imagen_final, nombre_archivo.c_str())) {
            cout << "Imagen guardada como: " << nombre_archivo << endl;
        } else {
            cout << "Error al guardar la imagen" << endl;
        }
    } else {
        cout << "Imagen no guardada." << endl;
    }

    cudaFree(d_imagen_data);
    cudaFree(d_buffer_inference_inputs);
    cudaFree(d_buffer_radiance_predicha);
    cudaFree(d_buffer_throughput);
    return sample_actual > 0;
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

        cout << "Configuración completada:" << endl;
        cout << "  Modo: " << (gui_state.renderMode == 0 ? "Normal" : "Reconstrucción") << endl;
        cout << "  Ancho: " << gui_state.imageWidth << endl;
        cout << "  Alto: " << gui_state.imageHeight << endl;
        cout << "  SPP: " << gui_state.samplesPerPixel << endl;
        cout << "  Cargar modelo: " << (gui_state.loadModel ? "Sí" : "No") << endl;
        cout << "  Guardar modelo MLP: " << (gui_state.saveMlpModel ? "Sí" : "No") << endl;
        cout << "  Archivo salida: " << gui_state.outputFileName << endl;

        int ancho_imagen = gui_state.imageWidth;
        int alto_imagen = gui_state.imageHeight;
        int samples_per_pixel = gui_state.samplesPerPixel;
        bool cargar_modelo = gui_state.loadModel;
        string nombre_archivo = gui_state.outputFileName;
        string ruta_modelo_entrenado = gui_state.mlpModelPath;
        int num_pixels = ancho_imagen * alto_imagen;

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

        SceneBounds scene_bounds;
        scene_bounds.min = Vector3d(-6.0f, -4.0f, -10.0f);
        scene_bounds.max = Vector3d( 6.0f,  6.0f,  2.0f);

        // Ejecutar modo reconstrucción si está seleccionado
        if (gui_state.renderMode == 1) {
            bool ok = renderizarModoReconstruccion(
                ancho_imagen, alto_imagen, samples_per_pixel, nombre_archivo,
                ruta_modelo_entrenado,
                d_camara,
                d_primitivas, num_primitivas,
                d_luces, num_luces,
                d_primitivas_malla, num_primitivas_malla,
                bvh_modelo_tiny,
                scene_bounds,
                ventana.get(),
                runtime_state
            );

            cudaFree(d_camara);
            limpiarGPU(d_primitivas, d_luces);
            
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

        int WARMUP_SAMPLES = 48;
        
        int frames_acumulados_reales = 0;

        // Mostrar la ventana de render desde el inicio para evitar pantalla vacia durante warmup
        runtime_state.warmupActive = true;
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
            bool es_warmup = (s < WARMUP_SAMPLES);
            runtime_state.warmupActive = es_warmup;
            
            cudaMemset(d_mlp_counter, 0, sizeof(unsigned int));
            cudaMemset(d_imagen_data, 0, num_pixels * sizeof(Color));

            /*
            launchKernelRender(gridSize, blockSize,
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
            launchKernelRenderTiny(gridSize, blockSize,
                d_camara, d_primitivas, num_primitivas, d_luces, num_luces,
                d_primitivas_malla, num_primitivas_malla, ancho_imagen, alto_imagen, runtime_state.samplesPerFrame, s,
                bvh_modelo_tiny.obtenerDatosGPU(), bvh_modelo_tiny.getPrimitivasGPU(), bvh_modelo_tiny.getNumPrimitivas(),
                imagen,
                nullptr, d_buffer_registros_raw, d_mlp_counter, num_pixels,
                d_buffer_inference_inputs, d_buffer_throughput, 
                !es_warmup, true
            );
            

            // Inferencia para imagen final
            mlp->inference(d_buffer_inference_inputs, d_buffer_radiance_predicha, num_pixels);
            launchKernelComposite(gridSize, blockSize, d_imagen_data, d_buffer_radiance_predicha, d_buffer_throughput, ancho_imagen, alto_imagen);

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
                // Al terminar el warmup, reseteamos el acumulador para eliminar el ruido inicial
                if (s == WARMUP_SAMPLES) fill(acumulador_cpu.begin(), acumulador_cpu.end(), Color(0,0,0));
                
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

        // Preparar imagen final
        Imagen imagenCPU(ancho_imagen, alto_imagen);
        float inv_samples = 1.0f / (float)frames_acumulados_reales;
        for (int i = 0; i < ancho_imagen * alto_imagen; i++) {
            imagenCPU.datos[i] = acumulador_cpu[i] * inv_samples;
        }

        // Aplicar tone mapping
        Imagen imagen_final = imagenCPU.filmic();

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
        cudaFree(d_buffer_train_shuffled);
        limpiarGPU(d_primitivas, d_luces);
        
    } catch (const exception& e) {
        cerr << "ERROR: " << e.what() << endl;
        return 1;
    }


    return 0;
}