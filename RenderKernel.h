#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "camara.h"
#include "primitiva.h"
#include "luzPuntual.h"
#include "imagen_gpu.h"
#include "transient.h"
#include "bvh.h"
#include "tinybvh_wrapper.h"
#include "mlp_types.h"

// Debug stats para NRC (predicción, throughput y contribución)
struct NrcDebugStats {
    uint32_t sampled = 0;

    // PT pre-composite
    uint32_t pt_nonzero = 0;
    uint32_t pt_naninf = 0;
    float pt_sum_r = 0.0f;
    float pt_sum_g = 0.0f;
    float pt_sum_b = 0.0f;
    uint32_t pt_min_r_bits = 0u;
    uint32_t pt_min_g_bits = 0u;
    uint32_t pt_min_b_bits = 0u;
    uint32_t pt_max_r_bits = 0u;
    uint32_t pt_max_g_bits = 0u;
    uint32_t pt_max_b_bits = 0u;
    float pt_sum_luma = 0.0f;

    // NRC (solo píxeles activos: throughput > 0)
    uint32_t active = 0;
    uint32_t naninf_any = 0;
    uint32_t pred_nonzero = 0;
    uint32_t contrib_nonzero = 0;

    float pred_sum_r = 0.0f;
    float pred_sum_g = 0.0f;
    float pred_sum_b = 0.0f;
    uint32_t pred_min_r_bits = 0u;
    uint32_t pred_min_g_bits = 0u;
    uint32_t pred_min_b_bits = 0u;
    uint32_t pred_max_r_bits = 0u;
    uint32_t pred_max_g_bits = 0u;
    uint32_t pred_max_b_bits = 0u;

    float th_sum_r = 0.0f;
    float th_sum_g = 0.0f;
    float th_sum_b = 0.0f;
    uint32_t th_min_r_bits = 0u;
    uint32_t th_min_g_bits = 0u;
    uint32_t th_min_b_bits = 0u;
    uint32_t th_max_r_bits = 0u;
    uint32_t th_max_g_bits = 0u;
    uint32_t th_max_b_bits = 0u;

    float contrib_sum_r = 0.0f;
    float contrib_sum_g = 0.0f;
    float contrib_sum_b = 0.0f;
    uint32_t contrib_min_r_bits = 0u;
    uint32_t contrib_min_g_bits = 0u;
    uint32_t contrib_min_b_bits = 0u;
    uint32_t contrib_max_r_bits = 0u;
    uint32_t contrib_max_g_bits = 0u;
    uint32_t contrib_max_b_bits = 0u;
    float contrib_sum_luma = 0.0f;
};

__global__ void kernelRender(const Camara* camara, const Primitiva* primitivas, int num_primitivas, const LuzPuntual* luces, int num_luces,
                            const Primitiva* primitivas_malla, int num_primitivas_malla, int ancho_img, int alto_img, int samples_per_pixel, int frame_number,
                            const NodoBVH* nodos_bvh, const Primitiva* primitivas_bvh, int num_nodos_bvh,
                            ImagenGPU imagen_directa,
                            TransientRender transientRenderer, 
                            unsigned int* dev_counter, RegistroEntrenamiento* buffer_registros, unsigned int* counter_train, int max_cap_train,
                            DatosMLP* buffer_inference, Color* buffer_throughput, bool usar_red_inferencia, bool entrenar_red);

__global__ void kernelRender_tiny(const Camara* camara, const Primitiva* primitivas, int num_primitivas, const LuzPuntual* luces, int num_luces,
                            const Primitiva* primitivas_malla, int num_primitivas_malla, int ancho_img, int alto_img, int samples_per_pixel, int frame_number,
                            TinyBVHD_GPU nodos_bvh, const Primitiva* primitivas_bvh, int num_nodos_bvh,
                            ImagenGPU imagen_directa,
                            TransientRender transientRenderer, 
                            unsigned int* dev_counter, RegistroEntrenamiento* buffer_registros, unsigned int* counter_train, int max_cap_train,
                            DatosMLP* buffer_inference, Color* buffer_throughput, bool usar_red_inferencia, bool entrenar_red);

__global__ void kernelComposite(Color* img_pt, Color* img_prediccion, Color* throughput_map,
                                int ancho, int alto);

__global__ void kernelTransientComposite(
    DatosMLP* buffer_inference, Color* buffer_prediccion, Color* buffer_throughput,
    TransientRender transientRenderer,
    int ancho, int alto
);

__global__ void inicializarCamara(Camara* d_camara, int ancho_imagen, int alto_imagen);

__global__ void kernelCalcularTargets(
    RegistroEntrenamiento* buffer_registros,
    Color* buffer_prediccion_tail,
    DatosMLP* buffer_training_final,
    int num_elementos
);

__global__ void kernelPrepararInferenciaTail(
    const RegistroEntrenamiento* __restrict__ source_registros,
    DatosMLP* __restrict__ dest_inference_inputs,
    int num_elements
);

// Wrappers host para poder invocar desde ficheros .cpp
void launchKernelRender(dim3 gridSize, dim3 blockSize,
                        const Camara* camara, const Primitiva* primitivas, int num_primitivas, const LuzPuntual* luces, int num_luces,
                        const Primitiva* primitivas_malla, int num_primitivas_malla, int ancho_img, int alto_img, int samples_per_pixel, int frame_number,
                        const NodoBVH* nodos_bvh, const Primitiva* primitivas_bvh, int num_nodos_bvh,
                        ImagenGPU imagen_directa,
                        TransientRender transientRenderer,
                        unsigned int* dev_counter, RegistroEntrenamiento* buffer_registros, unsigned int* counter_train, int max_cap_train,
                        DatosMLP* buffer_inference, Color* buffer_throughput, bool usar_red_inferencia, bool entrenar_red, bool modo_reconstruccion);

void launchKernelRenderTiny(dim3 gridSize, dim3 blockSize,
                            const Camara* camara, const Primitiva* primitivas, int num_primitivas, const LuzPuntual* luces, int num_luces,
                            const Primitiva* primitivas_malla, int num_primitivas_malla, int ancho_img, int alto_img, int samples_per_pixel, int frame_number,
                            TinyBVHD_GPU nodos_bvh, const Primitiva* primitivas_bvh, int num_nodos_bvh,
                            ImagenGPU imagen_directa,
                            TransientRender transientRenderer,
                            unsigned int* dev_counter, RegistroEntrenamiento* buffer_registros, unsigned int* counter_train, int max_cap_train,
                            DatosMLP* buffer_inference, Color* buffer_throughput, bool usar_red_inferencia, bool entrenar_red, bool modo_reconstruccion);

void launchKernelComposite(dim3 gridSize, dim3 blockSize,
                           Color* img_pt, Color* img_prediccion, Color* throughput_map,
                           int ancho, int alto,
                           bool modo_reconstruccion);

void launchKernelTransientComposite(dim3 gridSize, dim3 blockSize,
                                    DatosMLP* buffer_inference, Color* buffer_prediccion, Color* buffer_throughput,
                                    TransientRender transientRenderer,
                                    int ancho, int alto, 
                                    bool modo_reconstruccion);

void launchInicializarCamara(Camara* d_camara, int ancho_imagen, int alto_imagen);

void launchKernelCalcularTargets(dim3 gridSize, dim3 blockSize,
                                 RegistroEntrenamiento* buffer_registros,
                                 Color* buffer_prediccion_tail,
                                 DatosMLP* buffer_training_final,
                                 int num_elementos);

void launchKernelPrepararInferenciaTail(dim3 gridSize, dim3 blockSize,
                                        const RegistroEntrenamiento* source_registros,
                                        DatosMLP* dest_inference_inputs,
                                        int num_elements);

// Calcula stats de depuración
void launchKernelNrcDebugStats(dim3 gridSize, dim3 blockSize,
    const Color* buffer_pt,
    const Color* buffer_prediccion,
    const Color* buffer_throughput,
    int num_pixels,
    int sample_count,
    NrcDebugStats* d_stats);
