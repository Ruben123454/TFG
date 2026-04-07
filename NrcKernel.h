#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "mlp_types.h"

__global__ void generarIndicesAleatorios(int* indices, int max_range, int count, unsigned int seed);

__global__ void prepararDatosEntrenamiento(const DatosMLP* __restrict__ buffer,
                                           const int* __restrict__ indices_aleatorios,
                                           int batch_size,
                                           int n_in,
                                           int n_out,
                                           SceneBounds bounds,
                                           float* __restrict__ input_matrix,
                                           float* __restrict__ target_matrix);

__global__ void prepararDatosInferencia(const DatosMLP* datos, uint32_t n_elements, uint32_t n_in, float* buffer_in, SceneBounds bounds);

__global__ void guardarSalidaInferencia(float* network_output, Color* buffer_color, int n_elements);

// Wrappers host para invocar desde .cpp
void launchGenerarIndicesAleatorios(int num_blocks, int num_threads, cudaStream_t stream,
                                    int* indices, int max_range, int count, unsigned int seed);

void launchPrepararDatosEntrenamiento(int num_blocks, int num_threads, cudaStream_t stream,
                                      const DatosMLP* buffer,
                                      const int* indices_aleatorios,
                                      int batch_size,
                                      int n_in,
                                      int n_out,
                                      SceneBounds bounds,
                                      float* input_matrix,
                                      float* target_matrix);

void launchPrepararDatosInferencia(int num_blocks, int num_threads, cudaStream_t stream,
                                   const DatosMLP* datos, uint32_t n_elements, uint32_t n_in, float* buffer_in, SceneBounds bounds);

void launchGuardarSalidaInferencia(int num_blocks, int num_threads, cudaStream_t stream,
                                   float* network_output, Color* buffer_color, int n_elements);
