#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "camara.h"
#include "primitiva.h"
#include "luzPuntual.h"
#include "imagen_gpu.h"
#include "bvh.h"
#include "tinybvh_wrapper.h"
#include "mlp_types.h"

__global__ void kernelRender(const Camara* camara, const Primitiva* primitivas, int num_primitivas, const LuzPuntual* luces, int num_luces,
                            const Primitiva* primitivas_malla, int num_primitivas_malla, int ancho_img, int alto_img, int samples_per_pixel, int frame_number,
                            const NodoBVH* nodos_bvh, const Primitiva* primitivas_bvh, int num_nodos_bvh,
                            ImagenGPU imagen_directa,
                            unsigned int* dev_counter, RegistroEntrenamiento* buffer_registros, unsigned int* counter_train, int max_cap_train,
                            DatosMLP* buffer_inference, Color* buffer_throughput, bool usar_red_inferencia, bool entrenar_red);

__global__ void kernelRender_tiny(const Camara* camara, const Primitiva* primitivas, int num_primitivas, const LuzPuntual* luces, int num_luces,
                            const Primitiva* primitivas_malla, int num_primitivas_malla, int ancho_img, int alto_img, int samples_per_pixel, int frame_number,
                            TinyBVHD_GPU nodos_bvh, const Primitiva* primitivas_bvh, int num_nodos_bvh,
                            ImagenGPU imagen_directa,
                            unsigned int* dev_counter, RegistroEntrenamiento* buffer_registros, unsigned int* counter_train, int max_cap_train,
                            DatosMLP* buffer_inference, Color* buffer_throughput, bool usar_red_inferencia, bool entrenar_red);

__global__ void kernelComposite(Color* img_pt, Color* img_prediccion, Color* throughput_map,
                                int ancho, int alto);

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
                        unsigned int* dev_counter, RegistroEntrenamiento* buffer_registros, unsigned int* counter_train, int max_cap_train,
                        DatosMLP* buffer_inference, Color* buffer_throughput, bool usar_red_inferencia, bool entrenar_red);

void launchKernelRenderTiny(dim3 gridSize, dim3 blockSize,
                            const Camara* camara, const Primitiva* primitivas, int num_primitivas, const LuzPuntual* luces, int num_luces,
                            const Primitiva* primitivas_malla, int num_primitivas_malla, int ancho_img, int alto_img, int samples_per_pixel, int frame_number,
                            TinyBVHD_GPU nodos_bvh, const Primitiva* primitivas_bvh, int num_nodos_bvh,
                            ImagenGPU imagen_directa,
                            unsigned int* dev_counter, RegistroEntrenamiento* buffer_registros, unsigned int* counter_train, int max_cap_train,
                            DatosMLP* buffer_inference, Color* buffer_throughput, bool usar_red_inferencia, bool entrenar_red);

void launchKernelComposite(dim3 gridSize, dim3 blockSize,
                           Color* img_pt, Color* img_prediccion, Color* throughput_map,
                           int ancho, int alto);

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
